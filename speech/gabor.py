import cmath
import math
import numpy as np


def gabor_1D(lamb, theta, phi, sigma, gamma, input):
    ret = 0
    for i in range(len(input)):
        x = i
        x_prim = x * cmath.cos(theta)
        y_prim = -x * cmath.sin(theta)
        g = cmath.exp(-(x_prim*x_prim+ gamma*gamma * y_prim*y_prim)/2*sigma*sigma)*cmath.cos(2*math.pi*x_prim/lamb + phi)
        ret += g * input[i]
    return ret

