import numpy as np
import cv2
from cv2tools import vis

from image import artistic_enhance, load_image

a = lambda *l: np.array(l)

import random
def ru(k=1):return random.random()*k
def rb(): return 1 if ru()<0.5 else 0
def rbuf(gen):
    n=10000
    buf = gen()
    k = len(buf)
    i=n

    def call():
        nonlocal n,k,i,buf
        i+=1
        if i>=k:
            buf = gen()
            i = 0
        return buf[i]
    return call
rn = rbuf(lambda:np.random.normal(size=10000).tolist())
def bound(n, k): return max(0,min(k, n))

class GenePoint:
    def __init__(self, side):
        self.bound = side
        self.point = [ru(side), ru(side)]
        self.index = ru(1)
        self.starts = rb()

        self.t1 = 1 # temp for coordinates
        self.t2 = 0.01 # temp for index ordering
        self.t3 = 0.01 # temp for flipping

    def copy(self):
        c = GenePoint(1)
        c.bound = self.bound
        c.point[0] = self.point[0]
        c.point[1] = self.point[1]
        c.index = self.index
        c.starts = self.starts

        c.t1 = self.t1
        c.t2 = self.t2
        c.t3 = self.t3

        return c

    def mutate(self):
        t0 = 0.01
        # self.t1 = bound(self.t1+rn()*t0, 5)
        # self.t2 = bound(self.t2+rn()*t0*.1, .03)
        # self.t3 = bound(self.t3+rn()*t0*.1, .3)

        temp = self.t1
        self.point[0] = bound(self.point[0]+rn()*temp, self.bound)
        self.point[1] = bound(self.point[1]+rn()*temp, self.bound)

        index_temp = self.t2
        self.index += rn()*index_temp

        flip_prob = self.t3
        if ru()<flip_prob:
            self.starts = 1 - self.starts

    def __repr__(self):
        return ' '.join(['{:.4f}'.format(k) for k in [self.t1, self.t2, self.t3]])\
        + '{},{:4f},{:d}'.format(self.point, self.index, self.starts)

class Gene:
    def __init__(self, side=None):
        if side is not None:
            self.points = [GenePoint(side) for i in range(10)]
            self.points = sorted(self.points, key=lambda gp:gp.index)
            self.fitness = None
        else:
            # empty gene
            self.points = []
            self.fitness = None

    def get_length(self):
        return len(self.points)

    def mutate(self):
        for p in self.points:
            p.mutate()

    def mate(self, another):
        # 1. cross
        mo, fa = self.points, another.points

        # determine range
        mmin = min(mo[0].index, fa[0].index)
        mmax = max(mo[-1].index, fa[-1].index)

        # choose point within range.
        cut = ru()*(mmax-mmin)+mmin

        so = Gene()
        da = Gene()

        idx = 0
        idy = 0
        for gp in mo:
            if gp.index < cut:
                so.points.append(gp.copy())
                idx+=1
            else:
                break

        for gp in fa:
            if gp.index < cut:
                da.points.append(gp.copy())
                idy+=1
            else:
                break

        for i in range(idx, len(mo)):
            da.points.append(mo[i].copy())

        for i in range(idy, len(fa)):
            so.points.append(fa[i].copy())

        # 2. mutate
        so.mutate()
        da.mutate()

        return so,da


    def draw_on(self, canvas): # assume canvas is 8-bit
        segments = []
        segment = []
        # print(self.points)
        for gp in self.points:
            if gp.starts == 1:
                if len(segment)>=2:
                    segments.append(a(segment))
                segment = []
            segment.append(gp.point)
        if len(segment)>=2:
            segments.append(np.array(segment))

        cv2.polylines(
            canvas,
            [(points*64).astype(np.int32) for points in segments],
            isClosed = False,
            color=(0, 0, 0),
            # color = self.color,
            thickness=1,
            lineType=cv2.LINE_AA,
            shift = 6,
        )

class Env:
    # load image as target
    def load_image(self, path, is_filename=True, scale=1.0, target_width=None):
        orig = load_image(path, is_filename)

        # crop to face region
        from facial import facecrop
        orig = facecrop(orig)

        if target_width is not None:
            scale = target_width / orig.shape[1]

        self.orig = orig
        self.scale = scale

        # set expected output h and w
        self.orig_h, self.orig_w = orig.shape[0:2]
        self.target_h, self.target_w = int(self.orig_h*scale), int(self.orig_w*scale)

        self.orig_hw = [self.orig_h, self.orig_w]
        self.target_hw = [self.target_h, self.target_w]

        # resize
        target = vis.resize_perfect(self.orig, self.target_h, self.target_w, cubic=True, a=3)

        # log
        print('loading image {}, scale:{:2.2f}, orig[{}x{}], now[{}x{}]'.format(
            path, scale,
            self.orig_w, self.orig_h, self.target_w, self.target_h))

        # floatify
        target = (target/255.).astype('float32')

        target = self.preprocess(target)

        # clip to normal range
        # target = np.clip(target, 0, 1)

        self.target = target

    # process of loaded images
    def preprocess(self, image):
        # return artistic_enhance(image)
        return (image**0.4545).astype(np.float32)

    def get_blank_canvas(self):
        return np.full_like(self.target, 255, dtype='uint8')

    def set_metric(self, metric):
        self.loss_metric = metric
        if hasattr(self, 'target'):
            # loss metric precomputation
            self.precomputated = self.loss_metric.prepare(self.target)

    def populate(self):
        self.n = 100
        self.population = [Gene(self.target_w) for i in range(self.n)]

    def step(self):
        population = self.population
        half = 50
        mates = np.random.choice(population,half,replace=False)
        newpop = []
        for i in range(0, half, 2):
            fa,mo = mates[i:i+2]
            # fa,mo = np.random.choice(population,2,replace=False)
            so,da = fa.mate(mo)

            if so.get_length()>1 and da.get_length()>1:
                newpop.append(so)
                newpop.append(da)

        population = newpop + population[len(newpop):]
        population = sorted(population, key=lambda g:self.get_fitness(g))
        # low fitness first, high fitness last
        self.population = population
        # self.population = population[-self.n:] # last n

    def compare_with_target(self, img):
        return self.loss_metric.compare(img, self.precomputated)

    def get_fitness(self, g):
        if g.fitness is not None:
            return g.fitness
        else:
            f = self.eval_fitness(g)
            g.fitness = f
            return f

    def get_mean_fitness(self):
        return sum([g.fitness for g in self.population])/len(self.population)
    def get_mean_length(self):
        return sum([g.get_length() for g in self.population])/len(self.population)

    def eval_fitness(self, g):
        canvas = self.get_blank_canvas()
        g.draw_on(canvas)
        imgloss = self.compare_with_target(
            np.divide(canvas,255., dtype=np.float32)
        )

        lg = len(g.points)
        segloss = 1
        lloss = 10

        if lg>1:
            segloss = (sum([gp.starts for gp in g.points])/lg)**2

            lloss = 0
            p0 = g.points[0]
            for i in range(1,lg):
                p1 = g.points[i]
                d = dist(p0.point, p1.point)
                lloss += max(0,d - 15)
                p0 = p1

            lloss /= lg

        fitness = -imgloss -segloss*0.0001 - lloss*0.005
        return fitness

import math
def dist(p0, p1):
    return math.sqrt(
        (p0[0]-p1[0])**2+\
        (p0[1]-p1[1])**2
    )

from losses import (
        PyramidLoss, NNLoss, SSIMLoss,
        LaplacianPyramidLoss, FaceWeightedPyramidLoss, LabPyramidLoss, LPLoss,
        SLLoss
    )

env = Env()
env.load_image('jeff.jpg',target_width=128)
env.set_metric(LPLoss)
env.populate()

# g0 = Gene(100)
# g1 = Gene(100)
# g2,g3 = g0.mate(g1)
# print([g.get_length() for g in [g0,g1,g2,g3]])

def r(ep):
    for i in range(ep):
        env.step()

        if (i+1)%5==0:
            epop = env.population
            best = epop[-1]

            print('iter {}/{} '.format(i+1, ep)+\
            'max:{:.6f} min:{:.6f} avg:{:.6f} avg_len:{:.1f}'.format(
                epop[-1].fitness,
                epop[0].fitness,
                env.get_mean_fitness(),
                env.get_mean_length(),
            )+' popsize:{}'.format(len(epop))\
            +'params:[{}]'.format(best.points[0])
            )

            c = env.get_blank_canvas()
            env.population[-1].draw_on(c)
            vis.show_autoscaled(c,name='best')

            c = env.get_blank_canvas()
            env.population[0].draw_on(c)
            vis.show_autoscaled(c,name='worst')

            vis.show_autoscaled(env.target,name='targ')
            cv2.waitKey(1)
