from superformula import flower
import numpy as np
a = lambda*k:np.array(k)

shifts = [a(0,0), a(0,1), a(1,1), a(1,0)]

total_segs = []
for i in range(4):
    segs = flower()
    # for s in segs:
    #     s += shifts[i] * 5
    segs += shifts[i] * 5
    total_segs.append([segs])

print(total_segs)

import pickle
with open('flower4.pickle', 'wb') as f:
    pickle.dump(total_segs, f)
