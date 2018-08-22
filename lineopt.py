import cv2
from cv2tools import vis,filt
import numpy as np
# from scipy.optimize import minimize

from lineenv import LineEnv, StrokeEnv

import cProfile

def perf_debug():
    cProfile.run('optimize_for_target(it=50)', sort='tottime')

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

le = StrokeEnv()
# le = LineEnv()
# le.load_image('hjt.jpg', target_width=256)
le.load_image('forms.jpg', target_width=256)
le.init_segments()

def to_optimize(v):
    le.from_vec(v)
    return le.calculate_loss()

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
        print('fitness: mean {:2.6f} (best {:2.6f} worst {:2.6f} out of {}) {:2.4f} s ({:2.4f}s/i, {:4.4f}i/s)'.format(
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

def run_cem_opt(it=2000):
    # initial_x = mc.to_vec()
    initial_x = le.to_vec()

    import time
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
        mutation = 2.0,
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
