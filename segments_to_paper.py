import pickle
import numpy as np

with open('jeff.pickle', 'rb') as f:
    segments = pickle.load(f)

# revert y-axis
reversion = np.array([1,-1])
segments = [s*reversion for s in segments]

# bound the points
minx = min([s[:,0].min() for s in segments])
miny = min([s[:,1].min() for s in segments])
maxx = max([s[:,0].max() for s in segments])
maxy = max([s[:,1].max() for s in segments])

# we want to transform the strokes to fit into (0,0) (100,100).
max_side = max(maxx-minx, maxy-miny)
desired_side = 150
scale = desired_side/max_side
offset = - np.array([minx, miny])

print('scale: {:.2f}'.format(scale))

# apply the transformation
segments = [(s+offset)*scale for s in segments]

# travel salesman
# https://ericphanson.com/blog/2016/the-traveling-salesman-and-10-lines-of-python/
def travel_salesman_sa(list_points):
    import random, numpy, math, copy, matplotlib.pyplot as plt
    ncities = len(list_points) # 15
    # cities = [random.sample(range(100), 2) for x in range(15)];
    cities = list_points
    tour = random.sample(range(ncities),ncities);
    for temperature in numpy.logspace(0,5,num=10000)[::-1]:
        [i,j] = sorted(random.sample(range(ncities),2));
        newTour =  tour[:i] + tour[j:j+1] +  tour[i+1:j] + tour[i:i+1] + tour[j+1:];
        if math.exp( ( sum([ math.sqrt(sum([(cities[tour[(k+1) % ncities]][d] - cities[tour[k % ncities]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]]) - sum([math.sqrt(sum([(cities[newTour[(k+1) % ncities]][d] - cities[newTour[k % ncities]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])) / temperature) > random.random():
            tour = copy.copy(newTour);
    plt.plot([cities[tour[i % ncities]][0] for i in range(ncities+1)], [cities[tour[i % ncities]][1] for i in range(ncities+1)], 'xb-');
    plt.show()

    return tour

indices = travel_salesman_sa([s[0] for s in segments])
segments = [segments[i] for i in indices]

# connect to bot
from cartman import bot
bot = bot()

import time
tick = time.time()
bot.home()
bot.set_speed(50000)

def pendown(): bot.goto(z=0)
def penup(): bot.goto(z=3)

def draw_segment(segment):
    penup()
    bot.goto(x=segment[0][0], y=segment[0][1])
    pendown()
    for i in range(1, len(segment)):
        bot.goto(x=segment[i][0], y=segment[i][1])
    penup()

# draw outer contour
draw_segment(np.array([
    [0,0],[0,desired_side],
    [desired_side,desired_side],[desired_side,0],
    [0,0],
]))

# segments
for s in segments:
    draw_segment(s)

bot.wait_until_idle()

print('time spent:', time.time()-tick)
