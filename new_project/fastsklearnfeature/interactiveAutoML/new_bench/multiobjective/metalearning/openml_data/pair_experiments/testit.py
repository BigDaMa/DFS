import numpy as np

l_safety = 0.7
u_safety = 1.0

l_acc = 0.50
u_acc = 0.80

print(list(np.arange(l_safety, u_safety + ((u_safety - l_safety) / 10.0), (u_safety - l_safety) / 10.0)))

print(list(np.arange(l_acc, u_acc, (u_acc - l_acc) / 10.0)))