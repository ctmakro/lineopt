import cv2
import numpy as np

from image import zeroone
from facial import heatmap

def pyramid_of_level(img, levels):
    a = [img]
    for i in range(levels):
        smaller = cv2.pyrDown(a[-1])
        if(len(smaller.shape)==2): smaller.shape+=1,
        a.append(smaller)
    return a

from cv2tools import vis, filt
class LPLoss:
    @staticmethod
    def prepare(target):
        # h,w = target.shape[0:2]
        # return vis.resize_perfect(target, h/3, w/3, a=1, cubic=False)
        return cv2.pyrDown(target**2, 2)

    @staticmethod
    def compare(incoming, prepared):
        incoming_r = LPLoss.prepare(incoming)
        # incoming_r = np.sqrt(incoming_r) # gamma correction
        return np.square((incoming_r - prepared)).mean()

class SobelLoss:
    @staticmethod
    def prepare(target):
        # h,w = target.shape[0:2]
        # return vis.resize_perfect(target, h/3, w/3, a=1, cubic=False)
        img = target[:,:,1]
        sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)
        return sobelx, sobely

    @staticmethod
    def compare(incoming, prepared):
        ix,iy = SobelLoss.prepare(incoming)
        px,py = prepared
        return (np.abs(ix*px + iy*py)).mean() * -0.005

class SLLoss:
    @staticmethod
    def prepare(target):
        return LPLoss.prepare(target), SobelLoss.prepare(target)

    @staticmethod
    def compare(incoming, prepared):
        lpt, spt = prepared
        return LPLoss.compare(incoming, lpt) + SobelLoss.compare(incoming, spt)

# class that estimates pyramid loss between 2 images.
class PyramidLoss:
    @staticmethod
    def build_pyramid(img):
        img = zeroone(img)
        return pyramid_of_level(img,5)

    @staticmethod
    def prepare(target):
        # assume target is of float32.

        # returns dict as temp storage for future computations.
        return PyramidLoss.build_pyramid(target)

    @staticmethod
    def compare(incoming, prepared):
        # assume incoming is of 8bit.
        incoming_pyramid = \
            PyramidLoss.build_pyramid(incoming)

        target_pyramid = prepared

        return sum([np.square((c-t)).mean()
            for c,t in zip(incoming_pyramid, target_pyramid)])

def bgr2lab(i):
    return cv2.cvtColor(i, cv2.COLOR_BGR2Lab)

# class that estimates pyramid loss between 2 images.
class LabPyramidLoss:
    @staticmethod
    def build_pyramid(img):
        img = zeroone(img)
        img = bgr2lab(img)
        p = pyramid_of_level(img,5)
        return p

    @staticmethod
    def prepare(target):
        # assume target is of float32.

        # returns dict as temp storage for future computations.
        return LabPyramidLoss.build_pyramid(target)

    @staticmethod
    def compare(incoming, prepared):
        # assume incoming is of 8bit.
        incoming_pyramid = \
            LabPyramidLoss.build_pyramid(incoming)

        target_pyramid = prepared

        return sum([np.square((c-t)).mean()
            for c,t in zip(incoming_pyramid, target_pyramid)])

class FaceWeightedPyramidLoss:
    @staticmethod
    def build_pyramid(img):
        img = zeroone(img)
        hm = heatmap(img)
        return pyramid_of_level(img, 5), pyramid_of_level(hm, 5)

    @staticmethod
    def prepare(target):
        # assume target is of float32.

        # returns dict as temp storage for future computations.
        return FaceWeightedPyramidLoss.build_pyramid(target)

    @staticmethod
    def compare(incoming, prepared):
        # assume incoming is of 8bit.
        incoming_pyramid = \
            PyramidLoss.build_pyramid(incoming)

        target_pyramid, heatmap_pyramid = prepared

        return sum([(np.square((c-t)) * w).mean()
            for c,t,w in zip(incoming_pyramid, target_pyramid, heatmap_pyramid)])

from image import laplacian_pyramid
from greedy import laplacian_loss_on_pyramids

class LaplacianPyramidLoss:
    @staticmethod
    def prepare(target):
        # return naive_laplacian_pyramid(target, 5)
        return laplacian_pyramid(target, 5)

    @staticmethod
    def compare(incoming, prepared):
        return laplacian_loss_on_pyramids(
            # naive_laplacian_pyramid(incoming, 5),
            laplacian_pyramid(incoming, 5),
            prepared,
        )

class SSIMLoss:
    @staticmethod
    def prepare(target):
        if target.shape[2] == 3:
            return target[:,:,1]
        else:
            k = target.copy()
            k.shape = target.shape[0:2]

    @staticmethod
    def compare(incoming, prepared):
        if incoming.shape[2] == 3:
            incoming = incoming[:,:,1]
        else:
            incoming.shape = incoming.shape[0:2]

        from skimage.measure import compare_ssim as ssim
        return 1 - ssim(
            incoming,
            prepared,
            datarange=1.0,
            multichannel=False,
            winsize=32,
        )

# CNN loss function
# please refer to https://github.com/richzhang/PerceptualSimilarity

class NNLoss:
    @staticmethod
    def prepare(target):
        from perdiff import im2tensor, dist
        return im2tensor(target)

    @staticmethod
    def compare(incoming, prepared):
        from perdiff import im2tensor, dist
        it = im2tensor(incoming)
        return dist(it, prepared).mean()
