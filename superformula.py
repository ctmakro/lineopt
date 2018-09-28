import numpy as np

def r(fi, a, b, m1, m2, n1, n2, n3):
    return (
        np.abs(np.cos(m1*fi/4)/a)**n2
        +np.abs(np.sin(m2*fi/4)/b)**n3
        ) ** (-1/n1)

def cir(*arg, **kw):
    radians = np.arange(501) / 500 * 2*np.pi * 2
    radiuses = r(radians, *arg, **kw)

    xs = np.cos(radians) * radiuses
    ys = np.sin(radians) * radiuses

    return np.stack([xs, ys], axis=1)

import matplotlib
from matplotlib import pyplot as plt

def showcase(**k):
    points = cir(**k)
    plt.plot(points[:,0], points[:,1])

def sweep(n1,n2,n3):
    m = 5
    for i in range(10):
        showcase(a=1., b=1+i*.1, m1=m, m2=m,
            n1=n1, n2=n2, n3=n3)

def flower():
    allpoints = []
    for i in range(10):
        points = cir(a=1, b=1+i*.1, m1=5, m2=5, n1=2, n2=8, n3=4)
        allpoints.append(points)
        # allpoints += points
    return np.concatenate(allpoints)

if __name__ == '__main__':
    sweep(2,8,4)
    plt.show()
