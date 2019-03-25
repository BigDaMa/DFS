import time


start = time.time()

print("start: " + str(start))

while True:

    diff = time.time() - start

    if diff >= 60:
        break

print("done: " + str(time.time()))