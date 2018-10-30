import numpy as np
import cv2

from cv2tools import filt,vis
from canton import *

import sys
sys.path.append('..')

from image import load_image

from train import netg

net = netg(8631)

forelayers = net.subcans[0:17]
forenet = Can()
for k in forelayers: forenet.add(k)
forenet.chain()

net.load_weights('w.npz')

def forenet_infer(img):
    k = img.view()
    k.shape = (1,) + k.shape
    return forenet.infer(k)[0]

def jeffdemo():
    jeff = load_image('../jeff.jpg')

    h,w = jeff.shape[0:2]
    jeff = vis.resize_perfect(jeff, h/2, w/2)
    print(jeff.shape)

    vis.show_autoscaled(jeff,600)

    jefff = np.divide(jeff,255., dtype=np.float32)

    jeffff = forenet_infer(jefff)
    print(jeffff.shape)

    jeffff = np.transpose(jeffff,[2,0,1])
    jeffff.shape+=(1,)
    vis.show_batch_autoscaled(jeffff*0.5+0.5, 800)

    cv2.waitKey(0)

if __name__ == '__main__':
    jeffdemo()
