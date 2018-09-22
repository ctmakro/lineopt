'''
pick up a pen of each color and draw a series of circles.
'''

import numpy as np
from toolchange import pickup, putdown, safepoint, docks

a = lambda *k:np.array(k)

def shifted_goto(bot, shift):
    def goto(**kw):
        for idx, axis in enumerate(['x','y','z']):
            if axis in kw:
                kw[axis] = kw[axis] + shift[idx]
        return bot.goto(**kw)
    return goto

from cartman import bot

b = bot()

# bottom left corner of the paper
paper_origin = a(90, 240, 0)
paper_goto = shifted_goto(b, paper_origin)

def pendown():paper_goto(z=1)
def penup():paper_goto(z=10)

def draw_segments(s): # numpy array of shape (N, 2)
    penup()
    paper_goto(x=s[0][0],y=s[0][1])
    pendown()

    for i in range(1,len(s)):
        paper_goto(x=s[i][0],y=s[i][1])
    penup()

def circle_at(center, radius):
    nsegs = max(32, radius * 16)
    rad = np.arange(nsegs+1)/nsegs * np.pi * 2
    xs = np.cos(rad) * radius + center[0]
    ys = np.sin(rad) * radius + center[1]
    segments = a(*[[x,y] for x,y in zip(xs,ys)])
    draw_segments(segments)

def square_at(center, radius):
    r = radius
    f = [[-1,-1],[-1,1],[1,1],[1,-1],[-1,-1]]
    sq = a(*f)
    sq = sq * radius + center
    draw_segments(sq)

b.home()
b.set_speed(10000)

for j in range(4): # four colors
    pickup(b, docks[j])

    # immediately after pickup, a pendown-penup sequence
    # has to be performed to make sure the pen slides up
    # to its ready position.
    pendown()
    penup()

    b.set_speed(40000)
    for i in range(1, 10):
        center = a(60+j*20, 60+j*20)
        radius = i*2

        square_at(center,radius)
        circle_at(center,radius)

    putdown(b, docks[j]) # put back to where it's taken from.

b.wait_until_idle()
