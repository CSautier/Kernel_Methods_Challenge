import sys
import numpy as np
from numba import jit, prange
from utils import *

########## Recursive (slower) ##########

#@jit(nopython=True)
#def B_dyn(B, x1, x2, k, l):
#    i, j = len(x1)-1, len(x2)-1
#    if i==-1 or j==-1:
#        return 0
#    if B[i,j,k] != -1:
#        return B[i,j,k]
#    if B[i-1, j, k]==-1:
#        B_dyn(B, x1[:-1], x2, k, l)
#    if B[i, j-1, k]==-1:
#        B_dyn(B, x1, x2[:-1], k, l)
#    if B[i-1, j-1, k]==-1:
#        B_dyn(B, x1[:-1], x2[:-1], k, l)
#    if x1[i-1] == x2[j-1]:
#        if B[i-1, j-1, k-1]==-1:
#            B_dyn(B, x1[:-1], x2[:-1], k-1, l)
#        B[i,j,k] = l*B[i-1, j, k] + l*B[i, j-1, k] - l**2*B[i-1, j-1, k] + l**2*B[i-1, j-1, k-1]
#    else:
#        B[i,j,k] = l*B[i-1, j, k] + l*B[i, j-1, k] - l**2*B[i-1, j-1, k]
#    return B[i,j,k]

@jit(nopython=True)
def B_dyn(B, x1, x2, k, l):
    vals = [(len(x1), len(x2), k)]
    while vals:
        i,j,k = vals[-1]
        if B[i,j,k] != -1:
            vals.pop()
            continue
        _i, _j = i-1, j-1
        if B[_i, j, k]==-1:
            vals.append((_i,j,k))
            continue
        if B[i, _j, k]==-1:
            vals.append((i,_j,k))
            continue
        if B[_i, _j, k]==-1:
            vals.append((_i,_j,k))
            continue
        if x1[_i] == x2[_j]:
            if B[_i, _j, k-1]==-1:
                vals.append((_i,_j,k-1))
                continue
            B[i,j,k] = l*B[_i, j, k] + l*B[i, _j, k] - l**2*B[_i, _j, k] + l**2*B[_i, _j, k-1]
        else:
            B[i,j,k] = l*B[_i, j, k] + l*B[i, _j, k] - l**2*B[_i, _j, k]
        vals.pop()
    return B[i,j,k]

########## Recursive (slower) ##########

@jit(nopython=True)
def K_dyn(B, x1, x2, k, l):
    if len(x1)<k:
        return 0
    return K_dyn(B, x1[:-1], x2, k, l) + l**2*np.sum(
        np.array([B_dyn(B, x1[:-1], x2[:j], k-1, l) for j in range(len(x2)) if x2[j]==x1[-1]])
    )

# @jit(nopython=True)
# def K_dyn(B, x1, x2, k, l):
#     n1, n2 = len(x1), len(x2)
#     vals = [(n1, n2)]
#     K = -np.ones((n1+1, n2+1))
#     while vals:
#         i,j = vals[-1]
#         if K[i,j] != -1:
#             vals.pop()
#             continue
#         if K[i-1,j] == -1:
#             vals.append((i-1,j))
#             continue
#         K[i,j] = K[i-1,j] + l**2*np.sum(
#             np.array([B_dyn(B, x1[:i], x2[:p], k-1, l) for p in range(j) if x2[p]==x1[i]])
#         )
#         vals.pop()
#     return K[n1,n2]

@jit(nopython=True)
def K_k(x1, x2, k, l):
    B = -np.ones((len(x1)+1, len(x2)+1, k+1))
    B[:,:, 0] = 1
    for _k in range(1, k+1):
        B[:_k, :, _k] = 0
        B[:, :_k, _k] = 0
    return K_dyn(B, x1, x2, k, l)

@jit(nopython=True, parallel=True)
def compute_kernel(X, k, lbd):
    K = np.zeros((9000,9000))
    for j in range(6000, 9000):
        K[j,j] = K_k(X[j], X[j], k, lbd)
    for i in range(6000):
        p = 6000+1000*(i//2000)
        print(i, p)
        for j in prange(p, p+1000):
            K[i,j] = K_k(X[i], X[j], k, lbd)
    return K

def main():

    X_paths = [f'data/Xtr{i}.csv' for i in range(3)]
    X = load_no_padding(X_paths)
    X2_paths = [f'data/Xte{i}.csv' for i in range(3)]
    X2 = load_no_padding(X2_paths)
    X = X+X2

    k = 6
    lbd = 0.1

    K = compute_kernel(X, k, lbd)
    K = K + K.T - np.diag(K)*np.eye(len(K))
    np.save(f"K{k}-substring-l{int(10*lbd)}_te.npy", K)

if __name__ == "__main__":
    main()
