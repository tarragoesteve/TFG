import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/home/esteve/PycharmProjects/TFG/Polinomial layer/1. Creating a convolutional layer/2par/E0B200.csv", header=None)
abs = []
for coefficient in np.asarray(data):
    abs.append(np.mean([x * x for x in coefficient]))

plt.plot(abs)
plt.savefig('2degree.png')
