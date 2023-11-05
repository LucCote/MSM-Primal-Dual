"""
@author: Adam Breuer
"""


import numpy as np
from scipy import sparse



class NetCoverSparse:  

    
    def __init__(self, A):
        """
        class NetCoverSparse:
            INPUTS and INSTANCE VARIABLES:
                A: (a sparse CSR array representing a SYMMETRIC network where each element in {0,1}
                Note that if A has self-loops (diag elements) they will be IGNORED
                groundset: a list of the ground set of elements with indices 1:nrow(A)
        """

        self.groundset = range(A.shape[0])
        self.A         = A

        assert(sparse.issparse(A))
        assert( np.sum(A - A.T) == 0 )
        assert( np.all(A.diagonal() == 1) ) 




    def value(self, S):
        if not len(S):
            return(0)

        assert(len(S)==len(np.unique(S))) # DELETE ME

        S = np.sort(list(set(S)))
        return self.A[S,:].max(0).sum()


    

    def marginalval(self, S, T):
        # Fast approach
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        assert(len(S)==len(np.unique(S)))
        assert(len(T)==len(np.unique(T)))

        # if len(set(S).intersection(T)):
        #     print('!!SETS S AND T OVERLAP IN MARGINALVAL!!')
        #     S = np.sort(list(set(S) - set(T)))
        #     # raise(Exception)

        return self.value(list(set().union(S, T))) - self.value(T)
