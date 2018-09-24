import cv2
from cv2tools import vis,filt
import numpy as np

from losses import PyramidLoss

from image import artistic_enhance, load_image

# environment that optimizes lines to match an image.
class LineEnv:
    def __init__(self, grayscale=False, metric=PyramidLoss, color=(0,0,0)):
        self.target_pyramid = None
        # self.loss_metric = PyramidLoss
        self.set_metric(metric)

        self.grayscale=grayscale

        self.color=color

    def set_metric(self, metric=None):
        if metric is not None:
            self.loss_metric = metric

        if hasattr(self, 'target'):
            # loss metric precomputation
            self.precomputated = self.loss_metric.prepare(self.target)

    # process of loaded images
    def preprocess(self, image):
        # return artistic_enhance(image)
        return image**0.4545

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

        if self.grayscale:
            # b/w
            target = np.einsum('ijk,kt->ijt', target, np.array([[.3], [.6], [.1]]))

        # clip to normal range
        target = np.clip(target, 0, 1)

        self.target = target

        self.set_metric()

    def compare_with_target(self, img):
        return self.loss_metric.compare(img, self.precomputated)

    def get_blank_canvas(self):
        return np.ones((
                self.target_h,
                self.target_w,
                1 if self.grayscale else 3,
        ), dtype='uint8')*255

    def init_segments(self, num_segs=60):
        # num_segs=60

        self.segments = []
        self.indices = []

        k = 0
        for i in range(num_segs):
            # self.add(Connected(stochastic_points_that_connects() * w))
            sptc = stochastic_points_that_connects()
            self.segments.append(sptc * max(self.target_h, self.target_w))

            k += len(sptc)
            self.indices.append(k)

        self.indices = self.indices[:-1]
        assert len(self.indices) == num_segs - 1

    def draw_on(self, canvas):
        cv2.polylines(
            canvas,
            [(points*64).astype(np.int32) for points in self.segments],
            isClosed = False,
            # color=(0, 0, 0),
            color = self.color,
            thickness=1,
            lineType=cv2.LINE_AA,
            shift = 6,
        )

    def calculate_loss(self):
        blank = self.get_blank_canvas() # 8bit
        self.draw_on(blank)
        return self.compare_with_target((blank/255.).astype('float32')) # float

    # into vector that could be optimized
    def to_vec(self):
        a = np.vstack(self.segments)
        return a.flatten()

    def from_vec(self, v):
        vc = v.copy()
        vc.shape = len(vc)//2, 2
        self.segments = np.split(vc, self.indices, axis=0)

    def segments_to_pickle(self, filename=None):
        if filename is None:
            filename = input('Please enter the filename to save pickle to:')

        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.segments,f)

class LineEnv2(LineEnv):
    def init_segments(self, num_segs=300):
        # num_segs=60

        self.segments = []
        self.indices = []

        k = 0
        for i in range(num_segs):
            # self.add(Connected(stochastic_points_that_connects() * w))
            # sptc = stochastic_points_that_connects()
            start = np.random.uniform(0,1,size=(2))
            total = np.random.normal(loc=start,scale=0.05,size=(3, 2))

            sptc = total
            self.segments.append(sptc * max(self.target_h, self.target_w))

            k += len(sptc)
            self.indices.append(k)

        self.indices = self.indices[:-1]
        assert len(self.indices) == num_segs - 1


cmyk = np.array([
    [7,184,228],
    [226,69,198],
    [255,255,25],
    [30,30,30],
    ])

class LineEnvCMYK(LineEnv):
    # CMYK version
    # separated into 4 colors

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.envs = [
            LineEnv(*a, **kw, color=tuple([int(cmyk[i][j]) for j in reversed(range(3))]))
            for i in range(4)
        ]

    def init_segments(self, num_segs=300):
        # num_segs=60
        nk = [int(num_segs*f) for f in [0.2, 0.4, 0.4, 0.4]]

        self.segments = []
        self.indices = []

        k = 0
        for idx,e in enumerate(self.envs):
            e.target_h = self.target_h
            e.target_w = self.target_w
            e.init_segments(num_segs=nk[idx])
            k += len(e.to_vec())
            self.indices.append(k)

        self.indices = self.indices[:-1]

    def draw_on(self, canvas):
        # cmyk multiplicative mixing
        cs = []
        for e in self.envs:
            c = np.full_like(canvas,255, dtype=np.uint8)
            e.draw_on(c)
            cs.append(c)

        # cs = [c.astype(np.float32) for c in cs]
        # result = canvas * cs[0] * cs[1] * cs[2] * cs[3]
        # result = np.divide(result, 255*255*255*255, dtype=np.float32).astype('uint8')
        cc = canvas.astype(np.float32)
        result = cc * cs[0] * cs[1] * cs[2] * cs[3]
        result = np.divide(result, 255*255*255*255, dtype=np.float32).astype('uint8')

        canvas[:] = result[:]

    # into vector that could be optimized
    def to_vec(self):
        a = np.concatenate([e.to_vec() for e in self.envs])
        return a.flatten()

    def from_vec(self, v):
        vc = v.copy()
        # vc.shape = len(vc)//2, 2
        segs = np.split(vc, self.indices, axis=0)
        for e,s in zip(self.envs, segs):
            e.from_vec(s)

    def segments_to_pickle(self, filename=None):
        if filename is None:
            filename = input('Please enter the filename to save pickle to:')

        import pickle
        with open(filename, 'wb') as f:
            pickle.dump([e.segments for e in self.envs],f)

def stochastic_points_that_connects(mindist=0.05):
    # rule:
    # 1. sample 2 endpoint
    # 2. find their middlepoint
    # 3. repeat using the 2 endpoints of each of the 2 new segments.

    # minimum dist between two connected points
    # mindist = 0.05

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

def seg_at_coord(image, y, x): # image is float, output color is uint8
    color = int(image[int(y), int(x)].mean()*255)
    ps = np.random.normal(loc=[x,y],scale=5,size=(5,2))
    return ps, color

class StrokeEnv(LineEnv):
    def __init__(self, *a, **k):
        super().__init__(*a,**k)
        self.color_multiplier = 3.0
        # self.color_multiplier = 1.0

    def init_segments(self, num_segs=190):
        # num_segs=60

        self.segments = []
        self.colors = []
        self.indices = []

        a = self.target_w * self.target_h
        s = np.sqrt(a / num_segs)

        rows = int(self.target_h / s)
        cols = int(self.target_w / s)

        num_segs = rows*cols

        k = 0
        for i in range(rows):
            for j in range(cols):

                y = i*s + s/2
                x = j*s + s/2

                points, color = seg_at_coord(self.target,y,x)

                self.segments.append(points)
                self.colors.append(color)

                k += len(points)
                self.indices.append(k)

        self.color_index = self.indices[-1]
        self.indices = self.indices[:-1]
        assert len(self.indices) == num_segs - 1

    def draw_on(self, canvas):
        for i in range(len(self.segments)):
            points = self.segments[i]
            if self.grayscale:
                color = int(self.colors[i])
            else:
                c0 = int(self.colors[i])
                color = (c0,c0,c0)
            # print(color,type(color))
            cv2.polylines(
                canvas,
                [(points*64).astype(np.int32)],
                isClosed = False,
                color=color,
                thickness=4,
                lineType=cv2.LINE_AA,
                shift = 6,
            )

    # into vector that could be optimized
    def to_vec(self):
        a = np.vstack(self.segments)
        a = a.flatten()

        colors = np.array(self.colors)/self.color_multiplier
        return np.concatenate([a, colors])

    def from_vec(self, v):
        vc = v.copy()

        # segment coordinates in front, colors in back.

        k = self.color_index

        segcoords = vc[0:k*2]
        segcoords.shape = k, 2

        self.segments = np.split(segcoords, self.indices, axis=0)

        self.colors = np.clip((vc[k*2:] * self.color_multiplier), 0, 255).astype('uint8')

def test_line_env():
    le = LineEnv()
    le.load_image('hjt.jpg', target_width=384)
    le.init_segments()

    le.from_vec(le.to_vec())

    nc = le.get_blank_canvas()
    le.draw_on(nc)

    cv2.imshow('target', le.target)
    cv2.imshow('canvas', nc)
    # cv2.imshow('canvas2', nc2)
    cv2.waitKey(0)

def test_stroke_env():
    le = StrokeEnv()
    le.load_image('hjt.jpg', target_width=384)
    le.init_segments()

    le.from_vec(le.to_vec())

    nc = le.get_blank_canvas()
    le.draw_on(nc)

    cv2.imshow('target', le.target)
    cv2.imshow('canvas', nc)
    # cv2.imshow('canvas2', nc2)
    cv2.waitKey(0)

if __name__ == '__main__':
    test_line_env()
    test_stroke_env()
