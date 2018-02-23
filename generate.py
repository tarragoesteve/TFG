import numpy as np

def exponents(variables, degree):
    if degree == 0:
        return [np.repeat(0,variables).tolist()]
    if variables == 0:
        return [];
    ret = []
    for newdegree in range(0,degree+1):
        for l in exponents(variables-1, newdegree):
            l.append(degree-newdegree)
            ret.append(l)
    return ret;


print( exponents(3,2) )