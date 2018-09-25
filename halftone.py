from image import load_image, resize
from facial import facecrop

import cv2
import numpy as np
a = lambda *k:np.array(k)

jeff = load_image('jeff.jpg')
jeff = facecrop(jeff)

jeff = resize(jeff, 512)

height,width = jeff.shape[0:2]

bw = jeff[:,:,1:2]

bw = cv2.equalizeHist(bw)

# lowpass
from cv2tools import vis, filt
fbw = np.divide(bw, 255, dtype='float32')
# lp = vis.lanczos_filter(fbw, 2,2, a=2)
lp = vis.lanczos_filter(fbw**2.2, 2,2, a=2)

c = np.full_like(jeff, 255)

def italic_iteration(refimg, callback, angle=25, step=4.0, vstep=4.0):
    # signature for callback: (x, y, linebegin=False)
    t = theta = angle / 180 * np.pi
    xf = a([np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)])
    ixf = np.linalg.inv(xf)

    rect = a([0,0],[0,1],[1,1],[1,0]) * a(*list(reversed(refimg.shape[0:2])))

    rotated_rect = rect.dot(xf)

    maxx = max(rotated_rect[:, 0])
    minx = min(rotated_rect[:, 0])
    maxy = max(rotated_rect[:, 1])
    miny = min(rotated_rect[:, 1])

    row = miny
    while row<maxy:
        col = minx
        linebegin = True
        while col < maxx:
            # test if point in ref image before rotation
            i = a(col,row).dot(ixf)
            if i[0]>=0 and i[0]<width and i[1]>=0 and i[1]<height:
                callback(i[0], i[1], linebegin=linebegin)
                linebegin = False
            else:
                pass
            col+=step
        row+=vstep

ea = error_accumulator = 0

def deltasigma(x, y, linebegin=False):
    # get color from point in lowpass img
    global ea
    if linebegin: ea = 0
    value = lp[int(y), int(x)]
    blackness = 1-value
    ea += blackness
    if ea>1:
        c[int(y), int(x)] = 0
        ea-=1
    else:
        pass

def deltasigma_line_gen(eff=1, black_scale=1): # effectiveness = 1 => printer
    lines = []
    current_line = None
    lastpoint = None
    ea = 0

    def deltasigma_line(x, y, linebegin=False):
        nonlocal ea, current_line, lastpoint, lines
        # get color from point in lowpass img

        def newline():
            nonlocal current_line
            if current_line is not None and len(current_line)>1:
                lines.append(current_line)
            current_line = None

        def addpoint(p1):
            nonlocal current_line
            if current_line is not None:
                current_line.append(p1)
            else:
                if lastpoint is not None:
                    current_line = [lastpoint, p1]
                else:
                    current_line = [p1]
        if linebegin:
            ea = 0
            lastpoint = None

        value = lp[int(lp.shape[0]-y-1), int(x)]
        blackness = (1 - value) * black_scale
        ea += blackness
        if ea > eff:
            # ea-=1
            ea -= eff
            # black
            addpoint((x, y))
        else:
            # white
            newline()

        lastpoint = (x, y)

    return deltasigma_line, lines

def threshold_line_gen(threshold=0.5): # effectiveness = 1 => printer
    lines = []
    current_line = None
    lastpoint = None

    def threshold_line(x, y, linebegin=False):
        nonlocal current_line, lastpoint, lines
        # get color from point in lowpass img

        def newline():
            nonlocal current_line
            if current_line is not None and len(current_line)>1:
                lines.append(current_line)
            current_line = None

        def addpoint(p1):
            nonlocal current_line
            if current_line is not None:
                current_line.append(p1)
            else:
                if lastpoint is not None:
                    current_line = [lastpoint, p1]
                else:
                    current_line = [p1]

        if linebegin:
            newline()
            lastpoint = None

        value = lp[int(lp.shape[0]-y-1), int(x)]
        blackness = 1 - value
        if blackness>threshold:
            # black
            addpoint((x, y))
        else:
            # white
            newline()

        lastpoint = (x, y)

    return threshold_line, lines
# italic_iteration(lp, deltasigma)

total_lines = []
for i in range(7):
    # deltasigma_line, lines = deltasigma_line_gen(eff=0.2, black_scale=0.2)
    threshold_line, lines = threshold_line_gen(threshold=(i+1)*(1/(7+1)))
    # italic_iteration(lp, deltasigma_line, angle=[15,45,65,75][i], step=2, vstep=5)
    italic_iteration(lp, threshold_line, angle=[0,25,50,75,10,35,60,85][i], step=2, vstep=8)
    total_lines += lines

# c = np.full_like(jeff, 255)
# wrapped, treat bottom left as 0,0
def polylines(c, arr, color=(0,0,0), thickness=2):
    # flip y
    flipy = a([1,-1])
    addh = a([0, c.shape[0]])
    arr = [k * flipy + addh for k in arr]
    cv2.polylines(
        c,
        [(k * 64).astype('int32') for k in arr],
        isClosed = True,
        color = color,
        # lineType=cv2.LINE_8,
        lineType=cv2.LINE_AA,
        thickness = thickness,
        shift = 6,
    )

total_lines = [np.array(l) for l in total_lines]
polylines(c, total_lines, thickness=1)

cv2.imshow('original', jeff)
cv2.imshow('bw', bw)
cv2.imshow('lowpass', lp)
cv2.imshow('canvas', c)

cv2.waitKey(0)

import pickle
with open('jefflines.pickle', 'wb') as f:
    pickle.dump(total_lines, f)
