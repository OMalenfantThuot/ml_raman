import numpy as np
def gamma_modes_pristine (natoms_sc):
    nmodes   = 3*natoms_sc
    v1 = np.zeros(nmodes)
    v2 = np.zeros(nmodes)
    for i in range(0,natoms_sc):
           j = i * 3
           if j==0:
               v1[i]   = 1.0
               v2[i+1] = 1.0
           if j%6==0:
               v1[j]   = 1.0
               v2[j+1] = 1.0
           elif (j%3==0 and j%6!=0):
               v1[j]   =-1.0
               v2[j+1] = -1.0
    v1=v1/np.linalg.norm(v1)
    v2=v2/np.linalg.norm(v2)
    return v1, v2

