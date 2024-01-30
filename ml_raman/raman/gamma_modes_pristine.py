import numpy as np
import h5py
def gamma_modes_pristine (natoms_sc):
    nmodes   = 3*natoms_sc
    #v1 = np.zeros(nmodes)
    #v2 = np.zeros(nmodes)
    #for i in range(0,natoms_sc):
           #j = i * 3
           #if j==0:
               #v1[i]   = 1.0*np.cos(-np.pi/3)
               #v1[i+1]   = 1.0*np.sin(-np.pi/3)
               #v2[i] = 1.0*np.sin(-np.pi/3)
               #v2[i+1] = -1.0*np.cos(-np.pi/3)
           #if j%6==0:
               #v1[j]   = 1.0*np.cos(-np.pi/3)
               #v1[j+1]   = 1.0*np.sin(-np.pi/3)
               #v2[j] = 1.0*np.sin(-np.pi/3)
               #v2[j+1] = -1.0*np.cos(-np.pi/3)
           #elif (j%3==0 and j%6!=0):
               #v1[j]   = -1.0*np.cos(-np.pi/3)
               #v1[j+1]   = -1.0*np.sin(-np.pi/3)
               #v2[j] = -1.0*np.sin(-np.pi/3)
               #v2[j+1] = 1.0*np.cos(-np.pi/3)
    with h5py.File('/home/dounia/projects/rrg-cotemich-ac/dounia/datasets/raman/gamma_modes_pristine.h5', 'r') as file: 
        v1_pr = np.array(file['v1'])
        v2_pr = np.array(file['v2'])
        v1 = np.tile (v1_pr, int(natoms_sc/2))
        v2 = np.tile (v2_pr, int(natoms_sc/2))
    v1=v1/np.linalg.norm(v1)
    v2=v2/np.linalg.norm(v2)
    return v1, v2

