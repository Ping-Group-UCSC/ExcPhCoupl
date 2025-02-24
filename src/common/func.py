#!/usr/bin/env python3
from __future__ import print_function, division
import numpy as np

from common.param import *

# k-point mapping functions
def wrap(k, c=np.zeros(3)):
    r = k - c - np.floor(k - c + 0.5)
    r = np.where(abs(r-0.5) < 1e-6, -0.5, r)
    return r + c

kcenter = np.array([0.5, 0.5, 0.5])

def k2ik(k):
    ktmp = wrap(k, kcenter)
    if k.ndim == 1:
        return int((np.round(ktmp[0]*qmesh[0]) % qmesh[0]) * qmesh[1] * qmesh[2] + 
                   (np.round(ktmp[1]*qmesh[1]) % qmesh[1]) * qmesh[2] + 
                   (np.round(ktmp[2]*qmesh[2]) % qmesh[2]))
    else:
        return ((np.round(ktmp[:,0]*qmesh[0]) % qmesh[0]) * qmesh[1] * qmesh[2] + 
                (np.round(ktmp[:,1]*qmesh[1]) % qmesh[1]) * qmesh[2] + 
                (np.round(ktmp[:,2]*qmesh[2]) % qmesh[2])).astype(np.int32)

def ik2k(ik):
    ikx = ik // (qmesh[1] * qmesh[2])
    iky = (ik // qmesh[2]) % qmesh[1]
    ikz = ik % qmesh[2]
    return np.array([ikx/qmesh[0], iky/qmesh[1], ikz/qmesh[2]]).reshape(3)

def k2ik_fqmesh(k,fqmesh):
  ktmp = wrap(k,kcenter)
  if k.ndim == 1:
    return int(np.round(ktmp[0]*fqmesh[0]*fqmesh[1]*fqmesh[2] + ktmp[1]*fqmesh[1]*fqmesh[2] + ktmp[2]*fqmesh[2]))
  else:
    return (np.round(ktmp[:,0]*fqmesh[0]*fqmesh[1]*fqmesh[2] + ktmp[:,1]*fqmesh[1]*fqmesh[2] + ktmp[:,2]*fqmesh[2])).astype(np.int32)

def ik2k_fqmesh(ik,fqmesh):
  ikx = ik // (fqmesh[1] * fqmesh[2])
  iky = (ik // fqmesh[2]) % fqmesh[1]
  ikz = ik % fqmesh[2]
  return np.array([ikx/fqmesh[0], iky/fqmesh[1], ikz/fqmesh[2]]).reshape(3)

def ik2ijk(ik):
   k=ik%Qmesh[2]
   j=((ik-k)/Qmesh[2])%Qmesh[1]
   i=(ik-k-j*Qmesh[2])/(Qmesh[1]*Qmesh[2])
   return np.asarray([int(i),int(j),int(k)])

def ijk2ik(i,j,k):
   ik=i*Qmesh[1]*Qmesh[2] + j*Qmesh[2] + k
   return ik

def ik2ijk_fine(ik):
   k=ik%fQmesh[2]
   j=((ik-k)/fQmesh[2])%fQmesh[1]
   i=(ik-k-j*fQmesh[2])/(fQmesh[1]*fQmesh[2])
   return np.asarray([int(i),int(j),int(k)])

def ijk2ik_fine(i,j,k):
   ik=i*fQmesh[1]*fQmesh[2] + j*fQmesh[2] + k
   return ik

#functions for broadening and occupation
def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def lorentz(x,mu,sig):
    return -(1./np.pi) * np.imag(1./(x-mu+sig*1j))

def n_BE(E,T):
    return 1./(np.exp(E/(T/au2kelvin))-1.)

# Fermi-Dirac distribution
def n_FD(E,T):
    return 1./(np.exp(E/(T/au2kelvin))+1.)

def fac_dist(E,E0,T):
    return np.exp(-(E-E0)/(T/au2kelvin))

# Interpolation function
def mapping_fine(iQx, iQy, iQz, data):
    # Compute grid indices based on coarse and fine meshes
    i = iQx * Qmesh[0] // fQmesh[0]
    j = iQy * Qmesh[1] // fQmesh[1]
    k = iQz * Qmesh[2] // fQmesh[2]

    # Normalize indices to get fractions
    ijk = np.asarray([i, j, k]).astype(np.float32) / np.asarray(Qmesh).astype(np.float32)
    xyz = np.asarray([iQx, iQy, iQz]).astype(np.float32) / np.asarray(fQmesh).astype(np.float32)

    # Fetch values from the data grid
    V000 = data[i, j, k].astype(np.float32)
    V100 = data[(i + 1) % Qmesh[0], j, k].astype(np.float32)
    V010 = data[i, (j + 1) % Qmesh[1], k].astype(np.float32)
    V001 = data[i, j, (k + 1) % Qmesh[2]].astype(np.float32)
    V101 = data[(i + 1) % Qmesh[0], j, (k + 1) % Qmesh[2]].astype(np.float32)
    V011 = data[i, (j + 1) % Qmesh[1], (k + 1) % Qmesh[2]].astype(np.float32)
    V110 = data[(i + 1) % Qmesh[0], (j + 1) % Qmesh[1], k].astype(np.float32)
    V111 = data[(i + 1) % Qmesh[0], (j + 1) % Qmesh[1], (k + 1) % Qmesh[2]].astype(np.float32)

    # Compute the differences for interpolation
    x, y, z = xyz - ijk

    # Perform trilinear interpolation
    Vxyz = (V000 * (1 - x) * (1 - y) * (1 - z) +
            V100 * x * (1 - y) * (1 - z) +
            V010 * (1 - x) * y * (1 - z) +
            V001 * (1 - x) * (1 - y) * z +
            V101 * x * (1 - y) * z +
            V011 * (1 - x) * y * z +
            V110 * x * y * (1 - z) +
            V111 * x * y * z)

    return Vxyz