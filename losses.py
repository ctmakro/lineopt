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
        return laplacian_pyramid(target, 5)

    @staticmethod
    def compare(incoming, prepared):
        return laplacian_loss_on_pyramids(
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
