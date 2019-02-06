import multiprocessing as mp
import time
import numpy as np

def wait(time_sec):
    time.sleep(time_sec)
    return time_sec



pool = mp.Pool(processes=4)

my_array = [5,1,2,3,7,4]

results = pool.map(wait, my_array)

print(results)
print(my_array)

assert (np.array(my_array)==np.array(results)).all() == True, "order changes"
