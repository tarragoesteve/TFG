from scipy.io import wavfile
import gabor

import numpy as np
stripe = 20000
filter_size=20000

#rate samples/sec

#ffmpeg -i ./1520881828282.mp3 a.wav
rate, data = wavfile.read("/home/esteve/PycharmProjects/TFG/speech/Data/a.wav")
#rate = 44100
#lamb, theta, phi, sigma, gamma, input
ret = []
for i in range(len(data)/stripe):
    input = data[i*stripe:(i+1)*stripe]
    ret.append(gabor.gabor_1D(1,1,1,1,1,input))
    if (i % 100 == 0): print(str(i) + "/" + str(len(data)/stripe))
print(ret)

