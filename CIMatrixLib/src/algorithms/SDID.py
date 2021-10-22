import numpy as np
#from sklearn.linear_model import LinearRegression
import cvxpy as cp

def SDID(O, Z, treat_units = [0], starting_time = 100):

    donor_units = []
    for i in range(O.shape[0]):
        if (i not in treat_units):
            donor_units.append(i)     
    
    ##Step 1, Compute regularization parameter
    
    D = O[:, 1:starting_time] - O[:, :starting_time-1]

    D_bar = np.mean(O[donor_units, :-1])

    z_square = np.mean((D - D_bar)**2)

    ##Step 2, Compute w^{sdid}
    Nco = len(donor_units)
    Ntr = len(treat_units)
    Tpre = starting_time
    Tpost = O.shape[1] - starting_time

    w = cp.Variable(Nco)
    w0 = cp.Variable(1)
    G = np.eye(Nco)
    A = np.ones(Nco)
    #G @ w >= 0
    #A.T @ w == 1

    mean_treat = np.mean(O[treat_units, :Tpre], axis = 0)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(w0+O[donor_units, :Tpre].T @ w - mean_treat) + z_square * Tpre * cp.sum_squares(w)), [G @ w >= 0, A.T @ w == 1])
    prob.solve()
    #print("\nThe optimal value is", prob.value) 
    #print("A solution w is")
    #print(w.value)

    w_sdid = np.zeros(O.shape[0]) 
    w_sdid[donor_units] = w.value
    w_sdid[treat_units] = 1.0 / Ntr

    ##Step 3, Compute l^{sdid}
    l = cp.Variable(Tpre)
    l0 = cp.Variable(1)
    G = np.eye(Tpre)
    A = np.ones(Tpre)
    #G @ w >= 0
    #A.T @ w == 1

    mean_treat = np.mean(O[donor_units, Tpre:], axis = 1)
    #print(mean_treat.shape)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(l0+O[donor_units, :Tpre] @ l - mean_treat)), [G @ l >= 0, A.T @ l == 1])
    prob.solve()
    #print("\nThe optimal value is", prob.value) 
    #print("A solution w is")
    #print(l.value)

    l_sdid = np.zeros(O.shape[1]) 
    l_sdid[:Tpre] = l.value
    l_sdid[Tpre:] = 1.0 / Tpost

    ##Step 4, Compute SDID estimator
    #tau = w_sdid.T @ O @ l_sdid


    n1 = O.shape[0]
    n2 = O.shape[1]

    weights = w_sdid.reshape((O.shape[0], 1)) @ l_sdid.reshape((1, O.shape[1]))

    a = np.zeros((n1, 1))
    b = np.zeros((n2, 1))
    tau = 0

    one_row = np.ones((1, n2))
    one_col = np.ones((n1, 1))
    for T1 in range(2000):
        a_new = np.sum((O-tau*Z-one_col.dot(b.T))*weights, axis=1).reshape((n1, 1)) / np.sum(weights, axis=1).reshape((n1, 1))
        b_new = np.sum((O-tau*Z-a.dot(one_row))*weights, axis=0).reshape((n2, 1)) / np.sum(weights, axis=0).reshape((n2, 1))
        if (np.sum((b_new - b)**2) < 1e-5 * np.sum(b**2) and np.sum((a_new - a)**2) < 1e-5 * np.sum(a**2)):
            break
        a = a_new
        b = b_new
        M = a.dot(one_row)+one_col.dot(b.T)
        tau = np.sum(Z*(O-M)*weights)/np.sum(Z*weights)

    return tau

    return tau