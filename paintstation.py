# pickup paint from tray

import numpy as np
a = lambda *k:np.array(k)

class PaintStation:
    def __init__(self, bot, dock0, spacing, stir_radius, speed=5000, num_docks=None):
        self.bot = bot
        self.dock0 = dock0

        self.spacing = spacing
        self.stir_radius = stir_radius

        self.speed = speed
        self.num_docks = num_docks

    def check_index(self, i):
        if self.num_docks is not None:
            if i>=self.num_docks or i<0:
                raise Exception('specified dock index ({}) exceeds the bounds'.format(i))

    def dip(self, dock_index):
        self.check_index(dock_index)
        b,d = self.bot, self.dock0 + a(self.spacing * dock_index,0,0)
        r = self.stir_radius

        b.goto(z=0)
        b.goto(x=d[0], y=d[1])

        b.set_speed(self.speed)

        b.goto(z=d[2]) # into paint

        for i in range(2):
            b.goto(x=d[0]-r, y=d[1]+r)
            b.goto(x=d[0]-r, y=d[1]-r)
            b.goto(x=d[0]+r, y=d[1]-r)
            b.goto(x=d[0]+r, y=d[1]+r)

        b.goto(x=d[0], y=d[1])
        b.goto(z=0) # lift off

if __name__ == '__main__':
    from cartman import bot
    b = bot()

    from toolchange import ToolChange
    tc = ToolChange(b, a(173,13,2),
        xclearance=25, yclearance=35, overshoot=1,
        num_docks=4, spacing=80, speed=10000)

    ps = PaintStation(b, a(234,732,-28), spacing=32, stir_radius=9, num_docks=3, speed=30000)

    b.home()
    b.set_speed(30000)
    tc.pickup(1)

    b.set_speed(30000)
    ps.dip(0)

    b.set_speed(30000)
    tc.putdown(1)

    b.sync()
    b.home()
