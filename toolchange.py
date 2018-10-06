# tool change 3000

import numpy as np
a = lambda *k:np.array(k)

class ToolChange():
    def __init__(self, bot, dock0,
        xclearance, yclearance,
        overshoot, spacing, speed=5000, num_docks=None):

        self.bot = bot
        self.dock0 = dock0

        self.overshoot = overshoot
        self.spacing = spacing

        self.speed = speed
        self.xc = xclearance
        self.yc = yclearance

        self.num_docks = num_docks

    # pick the pen up from a specific dock
    def pickup(self, dock_index):
        self.check_index(dock_index)

        b,d = self.bot, self.dock0 + a(self.spacing * dock_index,0,0)
        self.safepoint()
        b.set_speed(self.speed)
        b.goto(z=d[2]) # height for docking
        b.goto(x=d[0], y=d[1]+self.yc) # proximity
        b.goto(y=d[1]-self.overshoot) # y- direction, snap
        b.goto(y=d[1]) # retract
        b.goto(x=d[0]-self.xc) # x- direction
        b.goto(y=d[1]+self.yc) # pull away
        self.safepoint()

    def check_index(self, i):
        if self.num_docks is not None:
            if i>=self.num_docks or i<0:
                raise Exception('specified dock index ({}) exceeds the bounds'.format(i))

    # put the pen down at a specific dock
    def putdown(self, dock_index):
        self.check_index(dock_index)

        b,d = self.bot, self.dock0 + a(self.spacing * dock_index,0,0)
        self.safepoint()
        b.set_speed(self.speed)
        b.goto(z=d[2]) # height for docking
        b.goto(x=d[0]-self.xc, y=d[1]+self.yc)  #proximity
        b.goto(y=d[1]) # y-
        b.goto(x=d[0]) # in place
        b.goto(y=d[1]+self.yc) # de-snap, pull away
        self.safepoint()

    # goto a place where homing/picking/putting is safe.
    # always do that before everything.
    def safepoint(self):
        b = self.bot
        b.goto(z=self.dock0[2])
        b.goto(x=self.dock0[0]-self.xc, y=self.dock0[1]+self.yc)

# position of 1st dock
dock0 = a(345, 13, 22)
dock0 = a(336, 13, 21)

if __name__ == '__main__':
    from cartman import bot
    b = bot()
    b.home()
    b.set_speed(15000)

    # tc = ToolChange(b, dock0)
    #
    # for j in range(4):
    #     for i in range(4):
    #         tc.pickup(i)
    #         tc.putdown(i)
    #
    # b.sync()

    tc = ToolChange(b, a(173,13,2),
        xclearance=25, yclearance=35, overshoot=1,
        num_docks=4, spacing=80, speed=10000)

    for j in range(3):
        for m in range(4):
            tc.pickup(m)
            tc.putdown(m)

    b.sync()
