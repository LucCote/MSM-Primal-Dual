"""
@author: Adam Breuer
"""


import numpy as np
from scipy import sparse



class TrafficCoverDirWeighted:  

    
    def __init__(self, A):
        """
        class VertexCover:
            INPUTS and INSTANCE VARIABLES:
                A: (2D Numpy Array) a 2D SYMMETRIC np.array where each element in {0,1}, representing the adjacency matrix of the graph
                Note that if A has self-loops (diag elements) they will be IGNORED
                groundset: a list of the ground set of elements with indices 1:nrow(A)
        """

        self.groundset = range(A.shape[0])
        self.A         = A
        assert (not sparse.issparse(A))




    def value(self, S):
        """ Speed-Optimized implementation of Max Cut objective value for SYMMETRIC graphs """
        # Fast version:
        if not len(S):
            return(0)

        # assert(len(S)==len(np.unique(S))) # DELETE ME

        S = np.sort(list(set(S)))
        try:
            return np.sum(self.A[S]) + np.sum(self.A[:,S]) - np.sum(self.A[S][:,S]) # 3rd term prevents double counting

        except:
            print('S', S)
            raise(Exception)
    



    def marginalval(self, S, T):
        # Fast approach
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        # assert(len(S)==len(np.unique(S)))
        # assert(len(T)==len(np.unique(T)))

        # if len(set(S).intersection(T)):
        #     print('!!SETS S AND T OVERLAP IN MARGINALVAL!!')
        #     S = np.sort(list(set(S) - set(T)))
        #     # raise(Exception)

        return self.value(list(set().union(S, T))) - self.value(T)





