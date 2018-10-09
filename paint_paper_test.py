from cartman_defaults import *

from cartman import bot
# b = bot(verbose=True)
b = bot(verbose=False)

tc = DefaultToolChange(b)
ps = DefaultPaintStation(b)

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

def circle(origin, radius):
    splits = int(max(18, radius*4))
    zeroone = np.arange(splits+1) / splits
    radian = zeroone*2*np.pi
    xy = np.stack([np.cos(radian), np.sin(radian)], axis=-1) * radius + origin
    return xy

b.home()
b.set_speed(90000)
tc.pickup(1)

b.set_speed(90000)
ps.dip(0)

b.set_speed(90000)

for i in [75, 55, 35]:
    seg(circle(a(75,75), i)) # in paper coordinate.

tc.putdown(1)

b.sync()
b.home()
