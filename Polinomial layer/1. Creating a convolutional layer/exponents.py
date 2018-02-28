import numpy as np

def exactdegree(variables, degree):
    if degree == 0:
        return [np.repeat(0,variables).tolist()]
    if variables == 0:
        return [];
    ret = []
    for newdegree in range(0,degree+1):
        for l in exactdegree(variables-1, newdegree):
            l.append(degree-newdegree)
            ret.append(l)
    return ret;

def uptodegree(variables, degree):
    ret = []
    for i in range(0,degree+1):
        ret.extend(exactdegree(variables,i))
    return ret
