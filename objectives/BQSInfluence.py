"""
@author: Sharon Qian
"""


import numpy as np
from scipy import sparse



class BQSInfluenceMax:  

    
    def __init__(self, A, weights=None):
        """
        class VertexCover:
            INPUTS and INSTANCE VARIABLES:
                A: (2D Numpy Array) a 2D SYMMETRIC np.array where each element \in {0,1}, representing the adjacency matrix of the graph
                Note that if A has self-loops (diag elements) they will be IGNORED
                p: float in (0,1) the probability that a neighbor influences another
                groundset: a list of the ground set of elements with indices 1:nrow(A)
        """

        self.groundset = range(A.shape[0])
        self.A         = A
        self.weights = weights
        if weights is None:
          self.weights = np.ones(len(A))



    def value(self, S):
        if not len(S):
            return(0)

        # assert(len(S)==len(np.unique(S))) # DELETE ME

        # for each node not in S, sum weight
        neigh_list = []
        for j in S:
            neigh_list += list(np.where(self.A[j,:] == 1)[0])
        else:
            return np.sum([self.weights[n] for n in set(neigh_list)])


    

    def marginalval(self, S, T):
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        # assert(len(S)==len(np.unique(S)))
        # assert(len(T)==len(np.unique(T)))

        # if len(set(S).intersection(T)):
        #     print('!!SETS S AND T OVERLAP IN MARGINALVAL!!')
        #     S = np.sort(list(set(S) - set(T)))
            # raise(Exception)

        return self.value(list(set().union(S, T))) - self.value(T)

