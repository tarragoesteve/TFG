from scipy.io import wavfile
import gabor
from numba import jit
import numpy as np
import time


start = time.time()

stripe = 20000
filter_size= 20000

#rate samples/sec
rate, data = wavfile.read("./Data/a.wav")

lamb = []
theta = []
phi = []
sigma = []
gamma = []

@jit
def compute(input,filter):
    return np.sum(np.dot(filter,input))

#rate samples/sec

#ffmpeg -i ./1520881828282.mp3 a.wav

#lamb, theta, phi, sigma, gamma, input
start = time.time()
filters = []
filters.append(gabor.gabor_1D(1,1,1,1,1,filter_size))
end = time.time()
print(end - start)

start = time.time()
ret = []
for i in range(len(data)/stripe):
    input = data[i*stripe:(i+1)*stripe]
    for j in range(len(filters)):
        ret.append(compute(filters[j],input))
    if (i % 100 == 0): print(str(i) + "/" + str(len(data)/stripe))
print(ret)
end = time.time()
print(end - start)

