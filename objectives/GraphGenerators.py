"""
@author: Adam Breuer
"""

import random
import networkx as nx
#import math
import numpy as np
from scipy import sparse




def ErdosRenyiSymBool(n, p):
    """ Generate a 2D numpy array of a logical Erdos Renyi adjacency matrix of size n with param p: 0<p<1 that is SYMMETRIC (no self loops) """
    upper_tri = np.triu(np.random.random((n,n)) < p, 1)
    return ( upper_tri | upper_tri.T )



def ErdosRenyiSymBoolSparseCSR(n, p):
    """ Generate a sparse scipy CSC array of a logical Erdos Renyi adjacency matrix of size n with param p: 0<p<1 that is SYMMETRIC (no self loops) 
    Note that output will be slightly less dense than p because of the way we construct the matrix. It will work well when p is small """
    ii = []
    jj = []
    for rr in range(n):
        #row_draws = np.where( np.random.random(n).astype(dtype) < p/2.0 )[0]
        row_draws = np.where( np.random.random(n) < p/2.0 )[0]
        ii.extend( rr*np.ones( len(row_draws) ) )
        jj.extend( row_draws )

    out = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    out = out + out.transpose()
    out[out>0] = 1
    out.setdiag(0)
    return( out )



def ErdosRenyiSymBoolSparseCSRparallel(n, p, comm, rank, size):
    """ Generate a sparse scipy CSC array of a logical Erdos Renyi adjacency matrix of size n with param p: 0<p<1 that is SYMMETRIC (no self loops) 
    Note that output will be slightly less dense than p because of the way we construct the matrix. It will work well when p is small """
    ii_local = []
    jj_local = []
    ii = None
    jj = None
    for rr in np.array_split(range(n), size)[rank]:

        row_draws = np.where( np.random.random(n) < p/2.0 )[0]
        ii_local.extend( rr*np.ones( len(row_draws) ) )
        jj_local.extend( row_draws )

    ii = comm.allgather(ii_local)
    jj = comm.allgather(jj_local)

    # Flatten
    ii = [block for sublist_proc in ii for block in sublist_proc]
    jj = [block for sublist_proc in jj for block in sublist_proc]

    out = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    out = out + out.transpose()
    out[out>0] = 1
    out.setdiag(0)
    return( out )




def SBMSparseCSR(numberC, minSize, maxSize, p):
    """ Sparse SBM Graph with within-cluster pr(edge)==p and outside cluster pr(edge)==0.
    Note that output will be slightly less dense than p because of the way we construct the matrix. 
    It will work well when p is small."""
    blocks = []
    # Make sparse blocks iteratively
    for bb in range(numberC):
        n = np.random.randint(low=minSize, high=maxSize)
        blocks.append( ErdosRenyiSymBoolSparseCSR(n, p) )
    # Put blocks into block-diagonal sparse matrix
    return(sparse.csr_matrix( sparse.block_diag(blocks) ))




def SBMSparseCSRparallel(numberC, minSize, maxSize, p, comm, rank, size):
    """ Parallel generate a Sparse SBM Graph with within-cluster pr(edge)==p and outside cluster pr(edge)==0.
    Note that output will be slightly less dense than p because of the way we construct the matrix. 
    It will work well when p is small."""
    blocks_local = []
    blocks = None
    # Split sparse blocks across processors and make them
    C_local = np.array_split(range(numberC), size)[rank]
    for cluster in C_local:
        n = np.random.randint(low=minSize, high=maxSize)
        blocks_local.append( ErdosRenyiSymBoolSparseCSR(n, p) )
    blocks = comm.allgather(blocks_local)
    blocks = [block for sublist_proc in blocks for block in sublist_proc]
    # Put blocks into block-diagonal sparse matrix
    return(sparse.csr_matrix( sparse.block_diag(blocks) ))







