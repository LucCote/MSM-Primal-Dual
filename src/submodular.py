'''
@author: Adam Breuer
'''
from datetime import datetime
import numpy as np


def check_inputs(objective, k):
    '''
    Function to run basic tests on the inputs of one of our optimization functions:
    '''
    # objective class contains the ground set and also value, marginalval methods
    assert( hasattr(objective, 'groundset') )
    assert( hasattr(objective, 'value') )
    assert( hasattr(objective, 'marginalval') )
    # k is greater than 0
    assert( k>0 )
    # k is smaller than the number of elements in the ground set
    assert( k<len(objective.groundset) )
    # the ground set contains all integers from 0 to the max integer in the set
    assert( np.array_equal(objective.groundset, list(range(np.max(objective.groundset)+1) )) )


def greedy(objective, k):
    ''' 
    Greedy algorithm: for k steps.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    list of lists L_rounds -- each element is a list containing the solution set L at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in L_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    float dualfit -- upper bound on the optimal solution
    '''
    check_inputs(objective, k)

    queries = 0
    time0 = datetime.now()

    L = []
    N = [ele for ele in objective.groundset]
    dualfits = []

    L_rounds = []
    time_rounds = [0]
    query_rounds = [0]

    for i in range(k):
        if i%25==0:
            print('Greedy round', i, 'of', k)
        # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaning elements N
        ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
        bestVal_idx = np.argmax(ele_vals)
        L.append( N[bestVal_idx] )
        #print('G adding this value to solution:', ele_vals[bestVal_idx] )

        queries += 2*len(N)
        N.remove( N[bestVal_idx] )
        L_rounds.append([ele for ele in L])
        time_rounds.append((datetime.now() - time0).total_seconds())
        query_rounds.append(queries)
        mFj = max([objective.marginalval( [elem], L ) for elem in N])
        dualfits.append(k*mFj + objective.value(L))

    val = objective.value(L)
    time = (datetime.now() - time0).total_seconds()
    return val, queries, time, L, L_rounds, time_rounds, query_rounds, min(dualfits)

def primal_dual(objective, k):
    ''' 
    @author: Luc Cote
    Primal-Dual algorithm: for k steps.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    list of lists L_rounds -- each element is a list containing the solution set L at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in L_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    float dual -- upper bound on the optimal solution
    '''
    check_inputs(objective, k)

    queries = 0
    time0 = datetime.now()

    L = []
    N = [ele for ele in objective.groundset]

    L_rounds = []
    time_rounds = [0]
    query_rounds = [0]
    dual_hist = []

    # initialize betas, alpha, gamma, and tight set
    b = np.array([float(objective.value([ele])) for ele in N])
    db =np.array([0.0 for i in range(len(N))])
    a = float(np.max(b))
    da = 0.0
    T = np.argwhere(b >= a).flatten()
    y = 0.0
    dy = 0.0

     # track marginals and overall value
    F = [objective.marginalval( [ele], L ) for ele in N]
    FL = objective.value(L)
    queries += 2*len(N)+1

    for i in range(k+1):
        if i%25==0:
            print('primal-dual round', i, 'of', k)
        if i == k:
            print("i=k, dual derivative is ", (k*da)+dy)
        while (k*da)+dy < 0:
            # find the da limiting tight element
            js = T[0]
            for j in T:
                if db[j] > db[js]: js = j
            # find next element to become tight 
            l = js
            lval = 0
            for j in range(len(N)):
                if j in T:
                    continue
                jval = (F[j]-F[js])/(b[js]-b[j])
                if jval > lval:
                    l = j
                    lval = jval
            # take time step
            et = 0
            if lval == 0: # safeguard since we can't deal with infinities/division by 0
                et = 0
            else: 
                t = np.log(1+1/lval)
                et = np.exp(-t)
            for j in range(len(N)):
                b[j] = b[j]*et + (1-et)*F[j]
                db[j] = F[j] - b[j]
            y = y * et + (1-et)*FL
            dy = FL - y
            a = b[l]
            da = db[l]
            T = np.argwhere(b >= a).flatten()
        # make next pick
        dual_hist.append(k*a + y)
        if i == k:
            continue
        p = T[0]
        for j in T:
            if db[j] > db[p]: p = j
        L.append(N[p])
        # update rates
        F = [objective.marginalval( [ele], L ) for ele in N]
        FL = objective.value(L)
        queries += 2*len(N)+1
        dy = FL - y
        for j in range(len(N)):
            db[j] = F[j] - b[j]
        da=db[T[0]]
        for j in T:
            if db[j] > da: da=db[j]
        # tracking updates
        L_rounds.append([ele for ele in L])
        time_rounds.append((datetime.now() - time0).total_seconds())
        query_rounds.append(queries)

    val = objective.value(L)
    time = (datetime.now() - time0).total_seconds()

    return val, queries, time, L, L_rounds, time_rounds, query_rounds, ((k*a)+y), dual_hist

FPE = 0.001
def method3(objective, k, S):
    ''' 
    @author: Luc Cote
    Balkanski, Qian, and Singer paper method3: for k steps.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float sum(V) -- an upper bound on the optimal solution
    '''
    check_inputs(objective, k)
    N = [ele for ele in objective.groundset]
    N.sort(key=lambda ele: -1*objective.marginalval( [ele], S))
    v = np.array([0.0 for i in range(k)])
    start = 0
    sv = 0
    for j in range(k):
        i_s = None
        for i in range(start, len(N)):
            if objective.marginalval(N[:i+1], S) - sv >= objective.marginalval( [N[i]], S) - FPE:
                i_s = i
                break
        if objective.marginalval(N[:i_s], S) - sv >= objective.marginalval( [N[i]], S) - FPE:
            i_s = i_s - 1
            v[j] = objective.marginalval( N[:i_s+1], S) - sv
        else:
            v[j] = objective.marginalval( [N[i_s]], S)
        start = i_s
        sv += v[j]
    return sum(v)

def BQSBOUND(objective, k, S):
    ''' 
    @author: Luc Cote
    Balkanski, Qian, and Singer paper method4: for k steps.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    list S -- contains k+1 sets derived from iterations of the greedy algorithm on this k-MSM instance
    
    OUTPUTS:
    float sum(V) -- an upper bound on the optimal solution
    '''
    check_inputs(objective, k)
    time0 = datetime.now()
    N = [ele for ele in objective.groundset]
    OPT = objective.value(N)
    i = 0
    for Si in S:
        if i > 50: break
        if i > 20 and i % 5 != 0:
            continue
        OPTP, qm3 = method3(objective, k, Si)
        OPT = min(OPT, OPTP + objective.value(Si))
        i += 1
    time = (datetime.now() - time0).total_seconds()
    return OPT, time


def topk(objective, k):
    ''' 
    @author: Luc Cote
    Simplistic upper bound which returns the sum of the k highest singletons
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float sum([objective.value([ele]) for ele in N[:k]]) -- an upper bound on the optimal solution
    '''
    N = [ele for ele in objective.groundset]
    N.sort(key=lambda ele: -1*objective.value([ele]))
    return sum([objective.value([ele]) for ele in N[:k]])

def marginal(objective, k, S):
    ''' 
    @author: Luc Cote
    'Traditional' upper bound described in BQS paper which uses marginal values between greedy sets
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    list S -- contains k+1 sets derived from iterations of the greedy algorithm on this k-MSM instance
    
    OUTPUTS:
    float OPT -- an upper bound on the optimal solution
    '''
    N = [ele for ele in objective.groundset]
    OPT = objective.value(N)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            OPTP = (objective.value(S[j]) - np.power((1-1/k), j-i)*objective.value(S[i]))/(1-np.power((1-1/k), j-i))
            OPT = min(OPT, OPTP)
    return OPT

def curvature(objective, k):
    ''' 
    @author: Luc Cote
    'Traditional' upper bound described in BQS paper which approximates the curvature of a submodular instance
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float OPT -- an upper bound on the optimal solution
    '''
    N = [ele for ele in objective.groundset]
    a = N[0]
    ascore = objective.marginalval([a], N[1:])/objective.value([a])
    for i in range(len(N)):
        ap = N[i]
        apscore = objective.marginalval([ap], N[0:i]+N[i+1:])/objective.value([ap])
        if apscore < ascore:
            ascore = apscore
            a = ap
    S = []
    for i in range(k):
        next = N[0]
        nextscore = objective.marginalval([a], S + [next])
        for ele in N:
            if ele == a: continue
            if objective.marginalval([a], S + [ele]) < nextscore:
                nextscore = objective.marginalval([a], S + [ele])
                next = ele
        S.append(next)
    c = 1-objective.marginalval([a], S)/objective.value([a])
    return (1.0-np.exp(-c))/c

def upper_bounds(objective, k):
    val, queries, time, L, L_hist, time_rounds, query_rounds, dualfits = greedy(objective, k)
    S = [[]] + L_hist # include empty greedy solution
    BQSval, time = BQSBOUND(objective, k, S)
    return BQSval, topk(objective, k), marginal(objective, k, S), curvature(objective, k)