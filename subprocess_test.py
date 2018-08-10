import subprocess as sp
import os
curr_env = os.environ.copy()

pop = [sp.Popen(
    # ['python', 'lineenv.py'],
    ['ping','baidu.com'],
    stdin=sp.PIPE,
    # stdout=sp.DEVNULL,
    # stderr=sp.DEVNULL,
    env = curr_env,
) for i in range(24)]

import time; time.sleep(10)

for p in pop:
    print(p.poll(),'poll()')
    print(p.returncode)
