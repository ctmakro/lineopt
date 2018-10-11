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

    def dip(self, dock_index, times=1):
        self.check_index(dock_index)
        b,d = self.bot, self.dock0 + a(self.spacing * dock_index,0,0)
        r = self.stir_radius

        b.goto(z=0)
        b.goto(x=d[0], y=d[1])

        b.set_speed(self.speed)

        b.goto(z=d[2]) # into paint

        for i in range(times):
            b.goto(x=d[0]-r, y=d[1]+r)
            b.goto(x=d[0]-r, y=d[1]-r)
            b.goto(x=d[0]+r, y=d[1]-r)
            b.goto(x=d[0]+r, y=d[1]+r)

        b.goto(x=d[0], y=d[1])
        b.goto(z=0) # lift off

class WaterStation(PaintStation):
    def __init__(self, *a, **k):
        super().__init__(spacing=100, num_docks=1, *a, **k)

    def wash(self):
        return self.dip(0, times=3)

class SpongeStation(WaterStation):
    def wipe(self):
        return self.dip(0, times=3)

def DefaultWaterStation(b):
    return WaterStation(b, a(66,753,-25), stir_radius=15, speed=10000)

def DefaultPaintStation(b):
    return PaintStation(b, a(234,732,-28), spacing=32, stir_radius=9, num_docks=3, speed=10000)

def DefaultSpongeStation(b):
    return SpongeStation(b, a(165, 753, -15), stir_radius=25, speed=10000)

if __name__ == '__main__':
    from cartman import bot
    b = bot()

    from cartman_defaults import *
    tc = DefaultToolChange(b)
    ps = DefaultPaintStation(b)
    ws = DefaultWaterStation(b)
    ss = DefaultSpongeStation(b)

    b.home()
    b.set_speed(30000)
    tc.pickup(1)

    b.set_speed(30000)
    ps.dip(0)
    # ps.dip(1)
    ws.wash()
    ss.wipe()

    ps.dip(2)
    ws.wash()
    ss.wipe()

    b.set_speed(30000)
    tc.putdown(1)

    b.sync()
    b.home()
