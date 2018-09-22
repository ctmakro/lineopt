# tool change 3000

import numpy as np
a = lambda *k:np.array(k)

class ToolChange():
    def __init__(self, bot, dock0):
        self.bot = bot
        self.dock0 = dock0

        # each dock is 30mm away from the previous one in x+ direction
        dock_spacing = 30

        # 4 dock in total
        self.docks = [dock0+a(dock_spacing*i,0,0) for i in range(4)]

    # x clearance
    xc = 15
    # y clearance
    yc = 40
    # how much more distance to push in the x- direction
    # to ensure a snap-mount of the pen
    overshoot = 8

    # pick the pen up from a specific dock
    def pickup(self, dock_index):
        b,d = self.bot, self.docks[dock_index]

        self.safepoint()
        b.set_speed(5000)
        b.goto(z=d[2]) # height for docking
        b.goto(x=d[0], y=d[1]+yc) # proximity
        b.goto(y=d[1]-overshoot) # y- direction, snap
        b.goto(y=d[1]) # retract
        b.goto(x=d[0]-xc) # x- direction
        b.goto(y=d[1]+yc) # pull away
        self.safepoint()

    # put the pen down at a specific dock
    def putdown(self, dock_index):
        b,d = self.bot, self.docks[dock_index]
        self.safepoint()
        b.set_speed(5000)
        b.goto(z=d[2]) # height for docking
        b.goto(x=d[0]-xc, y=d[1]+yc)  #proximity
        b.goto(y=d[1]) # y-
        b.goto(x=d[0]) # in place
        b.goto(y=d[1]+yc) # de-snap, pull away
        self.safepoint()

    # goto a place where homing/picking/putting is safe.
    # always do that before everything.
    def safepoint(self):
        b = self.bot
        b.goto(x=dock0[0]-xc,y=dock0[1]+yc)

# position of 1st dock
dock0 = a(330, 4, 32-6-4)
dock0 = a(336, 13, 20)
dock0 = a(345, 14, 22)

if __name__ == '__main__':
    from cartman import bot
    b = bot()
    b.home()
    b.set_speed(15000)

    tc = ToolChange(b, dock0)

    for j in range(4):
        for i in range(4):
            tc.pickup(i)
            tc.putdown(i)

    b.wait_until_idle()
