import numpy as np

def adpative_treatment_pattern(lowest_T, lasting_T, M):
    '''

        Input: lowest_T, lasting_T, M

        For each row i, if M(i,j) is the smallest among M(i, j-lowest_T:j) and no treatments on (i, j-lowest_T:j), then start the treatment on M(i,j+1) to M(i,j+lasting_T+1)
    '''

    Z = np.zeros_like(M)
    for i in range(Z.shape[0]):
        j = 0
        #print(i)
        while j < Z.shape[1]:
            flag = 0
            for k in range(1, lowest_T+1):
                if (j-k < 0 or Z[i, j-k]==1 or M[i,j] > M[i,j-k]):
                    flag = 1
                    break

            #print(i, j)
            if (flag == 0):
                for k in range(1, lasting_T+1):
                    if (j+k < Z.shape[1]):
                        Z[i, j+k] = 1
                j += lasting_T + lowest_T
            else:
                j = j + 1
    return Z

def iid_treatment(p, M):
    return np.random.rand(M.shape[0], M.shape[1]) <= p

def block_treatment_testone(m1, m2, M):
    Z = np.zeros_like(M)
    Z[:m1, m2:] = 1
    return Z

def block_treatment_testtwo(M):
    Z = np.zeros_like(M)
    ratio = np.random.rand()*0.8
    m1 = int(M.shape[0]*ratio)+1
    m2 = int(M.shape[1]*(1-ratio))-1
    Z[:m1, m2:] = 1
    return Z

def simultaneous_adoption(m1, m2, M):
    '''
        randomly select m1 units, adopt the treatment in [m2:]
    '''
    Z = np.zeros_like(M)
    treat_units = np.random.choice(range(M.shape[0]), m1, replace=False)
    Z[treat_units, m2:] = 1
    return Z, treat_units

def stagger_adoption(m1, m2, M):
    '''
        randomly select m1 units, adopt the treatment in after m2: randomly
    '''
    Z = np.zeros_like(M)
    treat_units = np.random.choice(range(M.shape[0]), m1, replace=False)
    for i in treat_units:
        j = np.random.randint(m2, high=M.shape[1])
        Z[i, j:] = 1 
    #Z[treat_units, m2:] = 1
    return Z

if (__name__ == 'main'):
    M = np.zeros((5, 5))
    print(simultaneous_adoption(2, 2, M))
