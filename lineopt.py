import cv2
from cv2tools import vis,filt
import numpy as np
# from scipy.optimize import minimize
from lineenv import (LineEnv, StrokeEnv,
LineEnv2, LineEnvCMYK, LineEnv3, LineEnv4)

from losses import (
    PyramidLoss, NNLoss, SSIMLoss,
    LaplacianPyramidLoss, FaceWeightedPyramidLoss, LabPyramidLoss, LPLoss,
    SLLoss
    )
import math
import cProfile

def perf_debug():
    cProfile.run('schedule3()', sort='tottime')
    # cProfile.run('optimize_for_target(it=50)', sort='tottime')

# exploit parallelism
# parallel = True
parallel = False

ncpu = 12
if parallel:
    from llll import MultithreadedMapper, PoolMaster
    pm = PoolMaster('lineenv.py', is_filename=True, nproc=ncpu)

    def to_optimize(v):
        return pm.call(v)

    vv = mc.to_vec()
    for i in range(100):
        pm.call(vv, mc.indices) # assure indices propagated to all slaves


import argparse as ap
ap = ap.ArgumentParser()
ap.add_argument('filename')
d = ap.parse_args()

# le = StrokeEnv(grayscale=False)
# le = LineEnvCMYK()
# le = LineEnv2()
le = LineEnv4()
# le.load_image('hjt.jpg', target_width=256)
le.load_image(d.filename, target_width=256)
# le.load_image('fruits.jpg', target_width=256)
# le.load_image('jeff.jpg', target_width=256)
# le.load_image('forms.jpg', target_width=128)
# le.load_image('forms.jpg', target_width=64)
le.init_segments(num_segs=1000)

# le.set_metric(SSIMLoss)
# le.set_metric(LabPyramidLoss)
# le.set_metric(FaceWeightedPyramidLoss)
le.set_metric(SLLoss)
# le.set_metric(PyramidLoss)
# le.set_metric(LaplacianPyramidLoss)

def to_optimize(v):
    le.from_vec(v)
    return le.calculate_loss()

b1=None
b2=None
# natural gradient
def minimize_ng(fun, x, iters, cb=None, lr=50, useb=False):
    global b1,b2

    x = x.copy()

    if useb == False or b1 is None:
        buffer1 = np.zeros_like(x)
        buffer2 = np.zeros_like(x)
        print('initbuffer')
    else:
        buffer1 = b1
        buffer2 = b2

    noise_level = lr/10
    # noise_level = 1
    m1 = 0.99 # momentum
    km1 = 1-m1

    m2 = 0.95
    km2 = 1-m2

    for i in range(iters):

        curr_fitness = fun(x)

        noise = np.random.normal(size=x.shape, scale=math.sqrt(noise_level))
        newx = x + noise
        new_fitness = fun(newx)

        diff_fitness = curr_fitness - new_fitness
        # fitness decr -> diff_fitness incr -> move in noise direction

        buffer1 = buffer1*m1 +\
         (diff_fitness/noise_level * noise) * km1

        # buffer2 = buffer2 * m2 + buffer1 * km2
        # x = x + buffer2 * lr

        x = x + buffer1 * lr

        if cb is not None:
            cb(i, x, curr_fitness)

    b1 = buffer1
    b2 = buffer2
    return x

def run_ng_opt(iters, lr=50):
    x = le.to_vec()

    lossrec = None
    def cb(i,x,loss):
        nonlocal lossrec
        lossrec = loss
        if (i+1) % 50==0:
            print('({}/{}) lr:{:.4f} loss {:.6f}'.format(i+1, iters, lr, loss))
            le.from_vec(x)
            show()

    newx = minimize_ng(to_optimize, x, iters, cb=cb, lr=lr, useb=True)
    le.from_vec(newx)

    return lossrec

def schedule4():
    lr = 50
    loss = 99999999
    while 1:
        newloss = run_ng_opt(1000,lr=lr)
        if newloss >= loss:
            lr = lr * 0.98
        loss = newloss
        if lr < .1:
            break

# minimization
def minimize_cem(fun, x, iters,
    survival=0.5,
    mutation=3.0, popsize=100, cb=None,
    mating=False,
    parallelizable=False):

    initial_mean = x
    initial_stddev = np.ones_like(initial_mean) * mutation

    def populate(size, mean, stddev):
        return [np.random.normal(loc=mean, scale=stddev)
            for i in range(size)]

    def mate(father, mother): # each being a vector
        m = (father + mother) * .5
        std = np.maximum(mutation, np.abs(father - mother) * 0.5)
        return np.random.normal(loc=m, scale=std)

    def populate_mate(size, parents):
        population = []
        lp = len(parents)
        for i in range(size):
            fa,mo = np.random.choice(lp,2,replace=False)
            population.append(mate(
                    parents[fa],parents[mo]
                ))
        return population

    initial_population = [initial_mean] + populate(
        popsize, initial_mean, initial_stddev)

    population = initial_population

    trace = []

    if parallelizable == False:
        pass
    else:
        from llll import MultithreadedMapper
        mm = MultithreadedMapper()

    for i in range(iters):
        print('{}/{}'.format(i+1,iters))
        import time
        start = time.time()

        # evaluate fitness for all of population

        if not parallelizable:
            fitnesses = [fun(v) for v in population]
        else:
            fitnesses = mm.map(fun, population)

        mean_fitness = sum(fitnesses)/popsize

        # time it
        duration = time.time() - start
        time_per_instance = duration / popsize

        # sort according to fitness
        fitnesses_sorted = sorted(zip(fitnesses, population), key=lambda k:k[0])
        max_fitness = fitnesses_sorted[0][0]
        min_fitness = fitnesses_sorted[-1][0]

        tophalf = fitnesses_sorted[0:int(popsize*survival)] # top half

        # keep the params, discard the fitness score
        tophalf_population = [fp[1] for fp in tophalf]

        # mean and std of top half (for CEM)
        ap = np.array(tophalf_population)
        mean = np.mean(ap, axis=0)
        stddev = np.maximum(np.std(ap, axis=0), mutation)

        # keep the best
        best = tophalf_population[0]

        # keep the mean (better than best in some cases)
        population = [best, mean]

        if not mating: # CEM
            # fill the rest of population with offsprings of the top half
            population += populate(
                size=popsize-len(population),
                mean=mean, stddev=stddev,
            )
        else:
            tophalf_population.append(mean)

            # same but mate pairwise
            population = tophalf_population + populate_mate(
                size = popsize-len(tophalf_population),
                parents = tophalf_population,
            )

        assert len(population) == popsize

        # logging
        print('mutation: {:2.4f} fitness: mean {:2.6f} (best {:2.6f} worst {:2.6f} out of {}) {:2.4f} s ({:2.4f}s/i, {:4.4f}i/s)'.format(
            mutation,
            mean_fitness, max_fitness, min_fitness,
            popsize,
            duration, time_per_instance, 1 / time_per_instance))

        log = {
            'best':best,
            'max_fitness': max_fitness,
            'mean':mean,
            'mean_fitness': mean_fitness,
        }
        trace.append(log)
        if cb is not None:
            cb(log)

    return {'x':best, 'trace':trace}

from scipy.optimize import differential_evolution
import time

def run_de_opt():
    initial_x = le.to_vec()
    tick = time.time()

    def cb(x,**kw):
        nonlocal tick
        if time.time() - tick > 0.2:
            le.from_vec(x)
            show()
            tick = time.time()

    res = differential_evolution(
        func=to_optimize,
        bounds=[(-10,266) for i in range(len(initial_x))],
        strategy='best1bin',
        maxiter=1000,
        popsize=1,
        disp=True,
        callback=cb,
    )

    print('optr', res)

def run_cem_opt(it=2000, temp = 2.0):
    # initial_x = mc.to_vec()
    initial_x = le.to_vec()

    tick = time.time()

    def callback(dic):
        nonlocal tick
        mean = dic['mean']
        # display
        le.from_vec(mean)

        if time.time() - tick > 0.2:
            show()
            tick = time.time()

    results = [
    # minimize_cem(
    #     to_optimize, initial_x,
    #     iters = it,
    #     mutation = 3.0,
    #     popsize=100,
    #     survival = 0.10,
    #     cb = callback,
    # ),
    minimize_cem(
        to_optimize, initial_x,
        iters = it,
        # mutation = 4.0,
        # mutation = 1.0,
        mutation = temp,
        # mutation = 2.0,
        popsize = 100,
        survival = 0.10,
        cb = callback,
        parallelizable = parallel,
    ),
    # minimize_cem(
    #     to_optimize, initial_x,
    #     iters = it,
    #     mutation = 4.0,
    #     popsize=100,
    #     survival = 0.10,
    #     cb = callback,
    # ),
    # minimize_cem(
    #     to_optimize, initial_x,
    #     iters = it,
    #     mutation = 3.0,
    #     popsize=100,
    #     survival = 0.1,
    #     cb = callback,
    # ),
    # minimize_cem(
    #     to_optimize, initial_x,
    #     iters = it,
    #     mutation = 1.0,
    #     popsize=100,
    #     survival = 0.5,
    #     cb = callback,
    #     mating=True,
    # ),
    # minimize_cem(
    #     to_optimize, initial_x,
    #     iters = it,
    #     mutation = 0.5,
    #     popsize=100,
    #     survival = 0.5,
    #     cb = callback,
    #     mating=True,
    # ),
    # minimize_cem(
    #     to_optimize, initial_x,
    #     iters = it,
    #     mutation = 2.0,
    #     popsize=100,
    #     survival = 0.5,
    #     cb = callback,
    #     mating=True,
    # ),
    # minimize_cem(
    #     to_optimize, initial_x,
    #     iters = it,
    #     mutation = 3.0,
    #     popsize=100,
    #     survival = 0.5,
    #     cb = callback,
    #     mating=True,
    # ),
    ]

    # mc.from_vec(x1)
    # show()

    if 0:
        from matplotlib import pyplot as plt
        for i,res in enumerate(results):
            plt.plot([d['mean_fitness'] for d in res['trace']], label = str(i))
        plt.legend()
        plt.show()
    else:
        le.from_vec(results[0]['x'])

    return results

def schedule3(): # annealing schedule. worked well
    import time
    tick = time.time()

    c = 0
    t = 3
    i = 9999
    while 1:
        results = run_cem_opt(10, temp=t)
        c+=10
        best = results[0]['trace'][-1]['max_fitness']
        if best < i:
            i = best
        else:
            t *= 0.9

        if t<0.01: break

    tt = time.time() - tick
    print(tt,'seconds',c,'generations',tt/c,'s/gen' )

def schedule():
    run_cem_opt(500)
    le.set_metric(NNLoss)
    run_cem_opt(1000)
    le.set_metric(PyramidLoss)

def schedule2():
    run_cem_opt(500)
    le.set_metric(SSIMLoss)
    run_cem_opt(1000)
    le.set_metric(PyramidLoss)

# def randomize():
#     v = mc.to_vec()
#     vc = np.random.normal(loc=v, scale=3)
#     mc.from_vec(vc)
#     # vcc = mc.to_vec()
#     # assert ((vcc - vc)**2).sum() < 1e-10
#     show()

def show():
    nc = le.get_blank_canvas()
    le.draw_on(nc)

    cv2.imshow('target', le.target)
    cv2.imshow('canvas', nc)
    # cv2.imshow('canvas2', nc2)
    cv2.waitKey(1)

show()
