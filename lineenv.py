import cv2
from cv2tools import vis,filt
import numpy as np

from scipy.optimize import minimize

def stochastic_points_that_connects():
    # rule:
    # 1. sample 2 endpoint
    # 2. find their middlepoint
    # 3. repeat using the 2 endpoints of each of the 2 new segments.

    # minimum dist between two connected points
    mindist = 0.05

    # given two point, return their centerpoint.
    def get_stochastic_centerpoint(p1, p2):
        c = (p1+p2)*.5
        # dist = np.linalg.norm(p1-p2)
        dist = np.sqrt(np.sum(np.square(p1-p2)))
        if dist < mindist:
            return None
            # if two point already very close to each other,
            # don't insert a centerpoint between them
        else:
            c += np.random.normal(loc=0, scale=0.2*dist, size=c.shape)
            return c

    # insert a new point between every two previously connected points.
    def insert_between(points):
        newpoints = []
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            newpoints.append(p1)
            cp = get_stochastic_centerpoint(p1, p2)
            if cp is not None:
                newpoints.append(cp)

        newpoints.append(p2)
        return newpoints

    while 1:
        points = [np.random.uniform(0,1,size=(2,)) for i in range(2)]

        for i in range(5):
            points = insert_between(points)

        # if the number of points included is larger than 4
        # we can return this as a legit polyline object
        if len(points)>4:
            return np.array(points)

        # otherwise we try again
        else:
            continue

# a group of points connected by lines
class Connected:
    def __init__(self, points=None):
        self.points = points

    def draw_on(self, canvas):
        cv2.polylines(
            canvas,
            [(self.points*16).astype(np.int32)],
            isClosed = False,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
            shift = 4,
        )

    # penalties of various forms.
    def calc_penalty(self):
        # segment direction penalty: if connecting segments are facing different direction, give penalty.

        deltas = self.points[:len(self.points)-1] - self.points[1:]

        sq_norms = np.square(deltas[:,0]) + np.square(deltas[:,1])
        # # normalize the vectors.
        # norms = np.linalg.norm(deltas,axis=1,keepdims=True)
        # deltas = deltas / (norms + 1e-2)
        #
        # dot_products = np.sum(
        #     deltas[:len(deltas)-1] * deltas[1:],
        #     axis=1,
        # )
        #
        # # print(deltas.shape)
        # # print(dot_products.shape)
        #
        # limited = np.maximum(-dot_products, 0)
        # angular_pen = limited.mean()

        min_length = 2
        max_length = w
        clipped = np.clip(sq_norms, min_length*min_length, max_length*max_length)

        pen = np.mean(np.abs(sq_norms - clipped))
        return pen

        length_pen = np.maximum(min_length - sq_norms, 0).mean()
        length_pen += np.maximum(sq_norms - max_length, 0).mean()

        return angular_pen + length_pen


class ManyConnected:
    def __init__(self, w=None, num_segs=60, iscopy=False):
        # w = width, indicate the range of coordinates of the lines

        self.list = []
        self.clist = []
        self.indices = []

        if not iscopy:

            k = 0
            for i in range(num_segs):
                # self.add(Connected(stochastic_points_that_connects() * w))
                sptc = stochastic_points_that_connects()
                self.list.append(sptc * w)

                k += len(sptc)
                self.indices.append(k)

        self.indices = self.indices[:-1]
        assert len(self.indices) == num_segs - 1

    def add(self, connected):
        self.list.append(connected.points)
        self.clist.append(connected)

    def draw_on(self, canvas):
        cv2.polylines(
            canvas,
            [(points*64).astype(np.int32) for points in self.list],
            isClosed = False,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
            shift = 6,
        )

    # into vector that could be optimized
    def to_vec(self):
        a = np.vstack(self.list)
        return a.flatten()

    def from_vec(self, v):
        vc = v.copy()
        vc.shape = len(vc)//2, 2
        self.list = np.split(vc, self.indices, axis=0)

    # penalties of various forms.
    def calc_penalty(self):
        return sum([c.calc_penalty() for c in self.clist]) / len(self.clist)


# canvas width
w = 256

def newcanvas():
    return np.ones((w,w,1), dtype='uint8')*255

mc = ManyConnected(w=w)

v = mc.to_vec()
print(type(v), v.shape)
mc.from_vec(v)

target = cv2.imread('hjt.jpg')
target = vis.resize_perfect(target, w, w, cubic=True, a=3)
target = (target/255.).astype('float32')
target = (target[:,:,1:2] + target[:,:,2:3]) *.5

target = np.clip(target*1+0.1, 0, 1)

def pyramid(img):
    a = [img]
    for i in range(5):
        a.append(cv2.pyrDown(a[-1]))
    return a

target_pyr = pyramid(target)

def multiscale_loss(canvas):
    canvas_pyr = pyramid((canvas/255.).astype('float32'))
    return sum([np.square((c-t)).mean() for c,t in zip(canvas_pyr, target_pyr)])

def singlescale_loss(canvas):
    return np.square(canvas.astype('float32')/255. - target).mean()

def to_optimize(v, indices = None):
    if indices is not None:
        mc.indices = indices

    mc.from_vec(v)
    nc = newcanvas()
    mc.draw_on(nc)
    ml = multiscale_loss(nc)
    # sl = singlescale_loss(nc)
    # pen = mc.calc_penalty()
    pen = 0

    return ml
    # return sl + pen * 0.001

if __name__ == '__main__':
    from llll import PoolSlave
    PoolSlave(to_optimize)
