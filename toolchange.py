# tool change 3000

import numpy as np
a = lambda *k:np.array(k)

# x clearance
xc = 15
# y clearance
yc = 40
# how much more distance to push in the x- direction
# to ensure a snap-mount of the pen
overshoot = 6

# pick the pen up from a specific dock
def pickup(bot, dock):
    b,d = bot,dock
    safepoint(b)
    b.set_speed(5000)
    b.goto(z=d[2]) # height for docking
    b.goto(x=d[0], y=d[1]+yc) # proximity
    b.goto(y=d[1]-overshoot) # y- direction, snap
    b.goto(y=d[1]) # retract
    b.goto(x=d[0]-xc) # x- direction
    b.goto(y=d[1]+yc) # pull away
    safepoint(b)


# put the pen down at a specific dock
def putdown(bot, dock):
    b,d = bot,dock
    safepoint(b)
    b.set_speed(5000)
    b.goto(z=d[2]) # height for docking
    b.goto(x=d[0]-xc, y=d[1]+yc)  #proximity
    b.goto(y=d[1]) # y-
    b.goto(x=d[0]) # in place
    b.goto(y=d[1]+yc) # de-snap, pull away
    safepoint(b)

# position of 1st dock
dock0 = a(330, 4, 32-6-4)
dock0 = a(336, 13, 20)

# each dock is 30mm away from the previous one in x+ direction
dock_spacing = 30

# 4 dock in total
docks = [dock0+a(dock_spacing*i,0,0) for i in range(4)]

# goto a place where homing/picking/putting is safe.
# always do that before everything.
def safepoint(bot):
    bot.goto(x=dock0[0],y=dock0[1]+yc)

if __name__ == '__main__':
    from cartman import bot
    b = bot()
    b.home()
    b.set_speed(15000)

    def demo():
        for i in range(4): # one pen available in dock0
            pickup(b, docks[i])
            putdown(b, docks[i])

    for j in range(4):
        demo()

    b.wait_until_idle()
