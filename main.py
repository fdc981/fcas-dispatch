import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
import scipy

def run_experiment(M_value):
    m = gp.Model()
    m.Params.LogToConsole = 0

    # number of consecutive trading intervals
    n = 23
    T = [i for i in range(n)]

    # column vectors
    p_raise = m.addMVar(n, vtype='C', name="p_raise")
    p_lower = m.addMVar(n, vtype='C', name="p_lower")

    # column vectors
    b_raise = m.addMVar(n, vtype='B', name="b_raise")
    b_lower = m.addMVar(n, vtype='B', name="b_lower")

    # in $AUD, lists
    l_raise = np.array([0.66448] * n)
    l_lower = np.array([0.15134] * n)

    # in kW
    p_min = 0
    p_max = 7

    # in kW
    soc_min = 0
    soc_max = 13.5
    initial_soc = 1

    # in hours
    delayed_delta = 5 / 60

    M = M_value # value with best results is M = p_max
    epsilon = 0.00000001 # (default error is 10e-4)

    # construct expressions for each SOC^t

    soc = [0] * n

    soc[0] = initial_soc
    for t in range(1, n):
        soc[t] = soc[t-1] + p_lower[t] * delayed_delta - p_raise[t] * delayed_delta

    m.setObjective(l_raise @ p_raise + l_lower @ p_lower, gp.GRB.MAXIMIZE)

    m.addConstr(p_min <= p_raise)
    m.addConstr(p_raise <= p_max)

    m.addConstr(p_min <= p_lower)
    m.addConstr(p_lower <= p_max)

    for t in T:
        m.addConstr(soc_min <= soc[t])
        m.addConstr(soc[t] <= soc_max)

    m.addConstr(-p_raise - M * (1 - b_raise) <= -epsilon)
    m.addConstr(-p_lower - M * (1 - b_lower) <= -epsilon)

    m.addConstr(p_raise - M * b_raise <= 0)
    m.addConstr(p_lower - M * b_lower <= 0)

    m.addConstr(b_raise + b_lower <= 1)

    m.optimize()

    return m.Runtime


if __name__ == '__main__':
    M_values = [7, 10, 50, 100, 500, 1000, 5000]
    num_trials = 100
    results = np.ndarray((len(M_values), num_trials))

    for i, M in enumerate(M_values):
        print(f'[{M}]', end=' ')
        for j in range(num_trials):
            print(j, end=' ')
            results[i][j] = run_experiment(M)
        print()

    p_values = np.ndarray((len(M_values), len(M_values)))
    for i, M1 in enumerate(M_values):
        for j, M2 in enumerate(M_values):
            if i != j:
                test_result = scipy.stats.wilcoxon(results[i, :],
                                                   results[j, :],
                                                   alternative='greater')
                p_values[i][j] = test_result.pvalue
            else:
                p_values[i][j] = 1

    results.tofile("results.txt")
    p_values.tofile("p_values.txt")
