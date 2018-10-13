
import sys
sys.path.append('../')

# pip install svg.path

from xml.dom import minidom
from svg.path import parse_path
import svg.path as sp

import numpy as np
import cv2

from cartman_defaults import *
from cartman import bot

svgf = 'experiment/jeff_head.svg'
svgf = 'jeff_head.svg'

doc = minidom.parse(svgf)

paths = doc.getElementsByTagName('path')

class bunchlines:
    def __init__(self, path):
        self.frompath(path)

    def frompath(self, path):
        p = path
        pa = p.attributes
        def pa(k):return p.attributes[k].value

        self.stroke_width = int(pa('stroke-width'))
        self.color = pa('stroke')
        self.d = pa('d')

        self.segments = [np.array(s) for s in parse_d_np(self.d)]

def bc(c): return (c.real, c.imag) # break complex number

def parse_d_np(d):
    orders = parse_path(d)

    acc = []
    temp = None

    for order in orders:
        t = type(order)
        if t is sp.path.Move:
            temp = []
            acc.append(temp)

        if t is sp.path.Line:
            if len(temp)==0:
                temp.append(bc(order.start))
            temp.append(bc(order.end))

    return acc


bls=[]
for p in paths:
    b = bunchlines(p)
    print(b.color, b.stroke_width, len(b.segments), b.segments[0])
    bls.append(b)


# properly zero the paths and calc bounds
maxx = max([max([s[:,0].max() for s in b.segments]) for b in bls])
minx = min([min([s[:,0].min() for s in b.segments]) for b in bls])

maxy = max([max([s[:,1].max() for s in b.segments]) for b in bls])
miny = min([min([s[:,1].min() for s in b.segments]) for b in bls])

print(maxx,minx,maxy,miny)

xspan = maxx-minx
yspan = maxy-miny

print(xspan, yspan)

shift = a(minx,miny - yspan)
flipy = a(1, -1)

scale = .5 # I chose this number to fit in A4 paper

for b in bls:
    for s in b.segments:
        s *= flipy
        s -= shift
        s *= scale

# for p in paths:
#     for a in p.attributes.keys():
#         print(a, p.attributes[a].value)

def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

def visualize():
    canvas = np.full((512,512,3), 255, dtype=np.uint8)
    for b in bls:

        rgb = hex2rgb(b.color)
        bgr = tuple(reversed(rgb))
        # rgb = tuple([n/255. for n in rgb])

        cv2.polylines(
            canvas,
            [(points*64).astype(np.int32) for points in b.segments],
            isClosed = False,
            # color=(0, 0, 0),
            color = bgr,
            thickness=b.stroke_width,
            lineType=cv2.LINE_AA,
            shift = 6,
        )

    cv2.imshow('painted', canvas)
    cv2.waitKey(0)

visualize()

def measure(s):
    # return total length of a segment
    # to determine when to re-dip in paint
    d = s[1:] - s[:-1]
    d = d**2
    d = d.sum(axis=-1)
    d = np.sqrt(d)
    return d.sum()

print(measure(a([0,0], [1,1], [0,0])))

def gocart():

    b = bot()

    ps,ws = DefaultPaintStation(b), DefaultWaterStation(b)
    ss = DefaultSpongeStation(b)
    tc = DefaultToolChange(b)

    paper = a(132, 291, -29)
    def brushdown(): b.goto(z=paper[2])
    def brushup(): b.goto(z=paper[2] + 10)

    def paper_goto(arr):
        b.set_offset(paper[0], paper[1])
        b.goto(x=arr[0], y=arr[1])
        b.set_offset(0,0)

    def seg(arr): # in paper coordinate.
        paper_goto(arr[0])
        brushdown()
        for i in range(len(arr)-1):
            paper_goto(arr[i+1])
        brushup()

    def goodwash():
        for i in range(3):
            ws.wash()
            ss.wipe()

    def fa():
        b.set_speed(90000)

    b.home()

    fa()
    tc.pickup(1) # thick first
    fa()
    goodwash()

    color_to_dip = 0
    fa()
    ps.dip(color_to_dip) # yellow

    db = 200
    budget = db # 200mm worth of paint on brush

    for s in bls[0].segments:
        d = measure(s)
        if budget>d:
            fa()
            seg(s)
            budget-=d
        else:
            # no enough budget
            ps.dip(color_to_dip) # get more paint
            budget=db
            fa()
            seg(s)
            budget-=d

    fa()
    goodwash()

    fa()
    tc.putdown(1)
    tc.pickup(0) # thin brush

    color_to_dip=1

    fa()
    goodwash()
    fa()
    ps.dip(color_to_dip) # yellow

    budget = db # 100mm worth of paint on brush

    for s in bls[1].segments:
        d = measure(s)
        if budget>d:
            fa()
            seg(s)
            budget-=d
        else:
            # no enough budget
            ps.dip(color_to_dip) # get more paint
            budget=db
            fa()
            seg(s)
            budget-=d
    fa()
    goodwash()

    fa()
    tc.putdown(0)

    b.sync()
    b.home()

gocart()
