# give greedy a try

import numpy as np
import cv2

from image import load_image, resize, lru_cache, laplacian_pyramid, artistic_enhance

class GreedyEnv:
    def set_reference_image(self, img):
        self.ref = img

    def load_reference_image(self, path):
        loaded = load_image(path)
        loaded = artistic_enhance(loaded)
        loaded = resize(loaded, self.pixelsize)
        loaded = np.clip(loaded*255, 0, 255)
        self.set_reference_image(loaded.astype('uint8'))

    def __init__(self):
        self.pixelsize = 256
        self.canvas = self.new_canvas()

    def show(self):
        cv2.imshow('ref', self.ref)
        cv2.imshow('canvas', self.canvas)
        cv2.waitKey(1)

    def new_canvas(self):
        return np.full((self.pixelsize, self.pixelsize, 3), 255, dtype='uint8')

ge = GreedyEnv()

ge.load_reference_image('jeff.jpg')
# ge.show()

def lp(i):
    return laplacian_pyramid(i, 5)

def laplacian_loss(i1, i2):
    lp1 = lp(i1)
    lp2 = lp(i2)
    return laplacian_loss_on_pyramids(lp1,lp2)

def weight_bgr(img):
    return img
    return np.einsum('ijk,kt->ijt', img, np.array([[.3], [.6], [.1]]))

from losses import PyramidLoss

loss = PyramidLoss

def laplacian_loss_on_pyramids(lp1, lp2):
    weights = [1] * len(lp1)
    weights[0] = 0.2
    weights[1] = 2
    weights[2] = 2
    weights[-1] = 1

    # sum to 100
    weights =  [w * (100/sum(weights)) for w in weights]
    # assert len(weights) == len(lp1)
    return sum([(weight_bgr((l1-l2)**2)).mean() * w for l1,l2,w in zip(lp1, lp2, weights)])

def vectorize(*a):
    v = []
    idx = [0]
    typ = []
    for i in a:
        t = type(i)
        if t != np.ndarray:
            v.append(i)
            idx.append(idx[-1]+1)
            typ.append(0)
        else:
            f = i.flatten()
            for j in f:
                v.append(j)
            idx.append(idx[-1]+len(f))
            typ.append(1)

    def devec(v):
        res = []
        for i in range(len(idx)-1):
            val = v[idx[i]:idx[i+1]]
            if typ[i]:
                res.append(val)
            else:
                res.append(val[0])
        return res

    return np.array(v, dtype='float32'), devec

# find one stroke that makes the canvas closer to ref.
# lp_ref = lp(ge.ref)
lp_ref = loss.prepare(ge.ref)

def place_one():
    global lp_ref
    counter = 0

    # lp_canvas = lp(ge.canvas)
    # ll_b4 = laplacian_loss_on_pyramids(lp_canvas, lp_ref)
    ll_b4 = loss.compare(ge.canvas, lp_ref)

    while 1:
        # initial parameters of a stroke
        position = np.random.uniform(0, ge.pixelsize, size=(2))
        radian = np.random.uniform(0, 2*np.pi)
        length = np.random.uniform(3,50)

        color = np.random.randint(2) # 0 or 1

        # color = np.random.uniform(0, 255, size=(3))

        # vectorize
        vec, devec = vectorize(position,radian,length,color)
        # vec, devec = vectorize(position,radian,length)

        def evaluate(vec):
            position, radian, length, color = devec(vec)
            # position, radian, length = devec(vec)

            endpoint = np.array([np.cos(radian), np.sin(radian)]) * length + position

            # tc = tuple(int(k) for k in color)
            # print(tc, type(tc[0]))

            cc = ge.canvas.copy()
            if color == 0:
                cv2.line(cc,
                    tuple((position*64).astype('int32')),
                    tuple((endpoint*64).astype('int32')),
                    # color = tc,
                    # color = (0,0,0),
                    color = (0,0,0),
                    thickness = 1,
                    lineType = cv2.LINE_AA,
                    shift = 6,
                )
            else:
                cv2.line(cc,
                    tuple((position*64).astype('int32')),
                    tuple((endpoint*64).astype('int32')),
                    # color = tc,
                    # color = (0,0,0),
                    color = (255,255,255),
                    thickness = 5,
                    lineType = cv2.LINE_AA,
                    shift = 6,
                )

            # lp_cc = lp(cc)
            # ll_after = laplacian_loss_on_pyramids(lp_cc, lp_ref)
            ll_after = loss.compare(cc, lp_ref)

            return ll_after, cc

        gradient_descent = True
        # gradient_descent = False # comment this line to use GD

        if gradient_descent:
            bestfx = 9999999
            bestvec = vec
            bestcc = None

            # do local gradient descent for n steps
            for n in range(5):
                # finite difference gradient
                dxs = [2, 2, 0.2, 3, None]
                grads = []
                fx,cc = evaluate(vec)

                if fx<bestfx:
                    bestfx = fx
                    bestvec = vec
                    bestcc = cc

                for i in range(len(vec)):
                    dx = dxs[i]

                    if dx is None: # don't descent on this parameter.
                        grads.append(0)
                        continue

                    vecp = vec.copy()
                    vecp[i] += dx
                    fxp,ccp = evaluate(vecp)

                    if fxp<bestfx:
                        bestfx = fxp
                        bestvec = vecp
                        bestcc = ccp

                    grads.append((fxp - fx) / (dx))

                # step size
                alpha = 10000

                # actual step
                step = np.array(grads) * alpha
                # print('step:', step)

                # gradient descent
                vec = vec - step

            fx,cc = evaluate(vec)

            epsilon = 1e-4
            if fx < bestfx - epsilon:
                bestfx = fx
                bestvec = vec
                bestcc = cc

            ll_after, cc = bestfx, bestcc
            position, radian, length, color = devec(bestvec)
            # position, radian, length = devec(bestvec)
        else:
            ll_after, cc = evaluate(vec)

        if ll_after<ll_b4:
            # keep
            ge.canvas = cc
            return {
                'pos':position,
                # 'endp':endpoint,
                'rad':radian,
                'len':length,
                'llbf':ll_b4,
                'llaf':ll_after,
                'cnt':counter+1,
            }
        else:
            counter+=1
            continue

import time
genesis = time.time()
def tick():
    global genesis
    t = time.time()
    n = t - genesis
    genesis = t
    return n

def place_many(c=100):
    tick()
    acc = 0
    for i in range(c):
        res = place_one()
        t = tick()
        print('before: {:.6f} after: {:.6f} tries: {:4d} time:{:.3f}s {:.2f}tries/s'.format(
            res['llbf'], res['llaf'], res['cnt'], t, res['cnt']/t
        ))

        acc+=t
        if acc>0.3:
            ge.show()
            acc = 0

    ge.show()

if __name__ == '__main__':
    ge.show()
    place_many(c=1000)
