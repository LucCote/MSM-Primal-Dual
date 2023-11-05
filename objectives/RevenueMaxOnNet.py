"""
@author: Adam Breuer
"""

import numpy as np
from scipy import sparse


class RevenueMaxOnNet:  

    
    def __init__(self, A, alpha):
        """
        Revenue Maximization (influence maximization with a concave function over influence):
            INPUTS and INSTANCE VARIABLES:
                A: (2D Numpy Array) a 2D SYMMETRIC np.array where each element is a weight, representing the adjacency matrix of the graph
                groundset: a list of the ground set of elements with indices 1:nrow(A)
                alpha: (float)  The coefficient of the square root on the value function

        """

        self.groundset = range(A.shape[0])
        self.A         = A
        self.alpha     = alpha 
        
        if sparse.issparse(A):
            assert( np.sum(A - A.T) == 0 )
            assert( np.all(A.diagonal() == 0) ) 
        else:
            assert( (A == A.T).all() )
            assert( (sum(np.diag(A)) == 0) )




    def value(self, S):
        if not len(S):
            return(0)
        S = list(set(S))

        return np.sum( ( np.sum( self.A[S], 0 ) )**self.alpha )

    


    def marginalval(self, S, T):
        """ Marginal value of adding set S to current set T for function above """
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        SuT = list(set().union(S, T))
        return( self.value(SuT) - self.value(T) ) 
