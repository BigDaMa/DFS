import numpy as np

rad = np.arange(300.)*np.pi/6
print(rad)
print(np.degrees(rad))

print(np.radians(np.degrees(rad)))



print (np.sum(np.radians(np.degrees(rad)) != rad))