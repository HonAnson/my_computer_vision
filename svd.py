# home made SVD
import numpy as np
from numpy import sqrt, diag, pad, zeros_like
from numpy.linalg import eig
from einops import rearrange




def svd(M):
    """ Actually this function finds the reduced SVD of a matrix
    Decompose arbitary matrix M into U*S*V', 
    where U and V are orthonormal matrix, S is truncated diagonal matrix
    input: M - 2d numpy array
    return: U, S, V, three 2d numpy arrays 
    """
    m, n = M.shape
    epsilon = 10e-6
    #step1: get eigen values and vectors of M'*M or M*M', depending on shape of M
    M_t = rearrange(M, 'a b -> b a')  # take transpose
    if m > n:
        symm = M_t @ M
    else:
        symm = M @ M_t
    eig_val, eig_vec = eig(symm)
    mask = eig_val > epsilon
    # filter out eigen vector and values that the eigenvalues are too close to zero
    eig_val = eig_val[mask]
    eig_vec = eig_vec[:,mask]

    idx = eig_val.argsort()[::-1]       # get index of where eigen values are sorted from large to small
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    rank = len(eig_val)

    # get S matrix
    singular_val = sqrt(eig_val)
    S = diag(singular_val) # S is just a diagonal matrix of non-zero eigen values of M'M or MM' squarerotted
    # S = pad(temp1, ((0,m), (0,n)), 'constant', constant_values = 0) # pad diagonal matrix of eigen values to get S

    V = np.zeros((n, rank))
    U = np.zeros((m, rank))

    # get U and V matrix
    if m >= n:
        V = eig_vec
        for i in range(rank):
            temp = M@V
            U[:,i] = temp[:,i] / singular_val[i]
    else:
        U = eig_vec    # eigenvec is our V
        for i in range(rank):
            temp = np.transpose(M)@U
            V[:,i] = temp[:,i] / singular_val[i]


    return U, S, V
    














