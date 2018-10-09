import numpy as np
a = lambda *k:np.array(k)

from paintstation import PaintStation

def DefaultPaintStation(b):
    return PaintStation(b, a(234,732,-28), spacing=32, stir_radius=9, num_docks=3, speed=30000)

from toolchange import ToolChange

def DefaultToolChange(b):
    return ToolChange(b, a(173,13,2),
        xclearance=25, yclearance=35, overshoot=1,
        num_docks=4, spacing=80, speed=30000)
