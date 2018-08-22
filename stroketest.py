import cv2
from cv2tools import vis,filt
import numpy as np

w = 256
canvas = np.ones((w, w, 1), dtype='uint8')

from lineenv import stochastic_points_that_connects

nstroke = 10
pointsarray = [stochastic_points_that_connects(mindist=0.5)*w for i in range(nstroke)]
colors = [np.random.randint(255) for i in range(nstroke)]

print(colors[0])

for i in range(nstroke):
    cv2.polylines(
        canvas,
        [(pointsarray[i]*64).astype(np.int32)],
        isClosed = False,
        color=colors[i],
        thickness=10,
        lineType=cv2.LINE_AA,
        shift = 6,
    )

cv2.imshow('yolo',canvas)
cv2.waitKey(0)
