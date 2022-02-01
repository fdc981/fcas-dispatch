"""Optimisation problems expressed as gurobipy models."""

import gurobipy as gp
import numpy as np
import src.data as data
from decimal import Decimal


def make_powerwall_model(n=12, M1=14, M2=14, epsilon=10**(-6), initial_soc=6):
    """Creates an optimisation problem for dispatch with a Tesla Powerwall."""

    assert M1 > 0
    assert M2 > 0
    assert n > 0
    assert epsilon > 0

    m = gp.Model()
    m.Params.LogToConsole = 0

    # indices of consecutive trading intervals
    T = [i for i in range(n)]

    # in kW
    p_min = 0
    p_max = 7

    # column vectors
    p_raise = m.addMVar(n, vtype='C', name="p_raise", lb=p_min, ub=p_max)
    p_lower = m.addMVar(n, vtype='C', name="p_lower", lb=p_min, ub=p_max)

    # column vectors
    b_raise = m.addMVar(n, vtype='B', name="b_raise")
    b_lower = m.addMVar(n, vtype='B', name="b_lower")

    # in $AUD, lists
    l_raise = np.array([0.66448] * n)
    l_lower = np.array([0.15134] * n)

    # in kW
    soc_min = 0.35 * 13.5
    soc_max = 13.5
    soc = m.addVars(n, vtype='C', name='soc', lb=soc_min, ub=soc_max)
    assert soc_min <= initial_soc and initial_soc <= soc_max

    # in hours
    delayed_delta = 5 / 60

    m.addConstr(soc[0] == initial_soc + p_lower[0] * delayed_delta - p_raise[0] * delayed_delta)
    for t in range(1, n):
        m.addConstr(soc[t] == soc[t-1] + p_lower[t] * delayed_delta - p_raise[t] * delayed_delta)

    m.setObjective(sum((l_raise[t] * p_raise[t] + l_lower[t] * p_lower[t] for t in T)), gp.GRB.MAXIMIZE)

    m.addConstrs((-p_raise[t] - M1 * (1 - b_raise[t]) <= -epsilon for t in T))
    m.addConstrs((p_raise[t] - M2 * b_raise[t] <= 0 for t in T))

    m.addConstrs((-p_lower[t] - M1 * (1 - b_lower[t]) <= -epsilon for t in T))
    m.addConstrs((p_lower[t] - M2 * b_lower[t] <= 0 for t in T))

    m.addConstrs((b_raise[t] + b_lower[t] <= 1 for t in T))

    return m


def make_alt_powerwall_model(n=12, M1=14, M2=14, epsilon=10**(-6), initial_soc=6):
    """Creates an optimisation problem for dispatch with a Tesla Powerwall."""
    m = gp.Model()
    m.Params.LogToConsole = 0

    # indices of consecutive trading intervals
    T = [i for i in range(n)]

    # in kW
    p_min = 0
    p_max = 7

    # column vectors
    p_raise = m.addMVar(n, vtype='C', name="p_raise", lb=p_min, ub=p_max)
    p_lower = m.addMVar(n, vtype='C', name="p_lower", lb=p_min, ub=p_max)

    # column vectors
    b_raise = m.addMVar(n, vtype='B', name="b_raise")
    b_lower = m.addMVar(n, vtype='B', name="b_lower")

    # in $AUD, lists
    l_raise = np.array([0.66448] * n)
    l_lower = np.array([0.15134] * n)

    # in kW
    soc_min = 0.35 * 13.5
    soc_max = 13.5
    assert soc_min <= initial_soc and initial_soc <= soc_max

    # in hours
    delayed_delta = 5 / 60

    soc = [0] * n
    soc[0] = initial_soc + p_lower[0] * delayed_delta - p_raise[0] * delayed_delta
    for t in range(1, n):
        soc[t] = soc[t-1] + p_lower[t] * delayed_delta - p_raise[t] * delayed_delta

    m.setObjective(sum((l_raise[t] * p_raise[t] + l_lower[t] * p_lower[t] for t in T)), gp.GRB.MAXIMIZE)

    m.addConstrs((soc_min <= soc[t] for t in T))
    m.addConstrs((soc_max >= soc[t] for t in T))

    m.addConstrs((-p_raise[t] - M1 * (1 - b_raise[t]) <= -epsilon for t in T))
    m.addConstrs((p_raise[t] - M2 * b_raise[t] <= 0 for t in T))

    m.addConstrs((-p_lower[t] - M1 * (1 - b_lower[t]) <= -epsilon for t in T))
    m.addConstrs((p_lower[t] - M2 * b_lower[t] <= 0 for t in T))

    m.addConstrs((b_raise[t] + b_lower[t] <= 1 for t in T))


    return m


def make_cooptimisation_model(
        n=12,
        M=14,
        epsilon=10**(-6),
        initial_soc=6,
        l_raise_s=None,
        l_lower_s=None,
        l_raise_d=None,
        l_lower_d=None,
        l_raise_f=None,
        l_lower_f=None,
        efficiency_in=0.92,
        efficiency_out=0.90,
        p_min=0,
        p_max=7,
        soc_min=0.35*13.5,
        soc_max=13.5,
        prices_from=None
):
    """Creates a Gurobi model for finding the dispatch maximising profit when
    participating in all contingency FCAS markets.

    Args:
        n: number of trading intervals
        M: value of big M for binary indicator constraints
        epsilon: an arbitrarily small value
        initial_soc: initial state of charge (in MWh)
        l_raise_s: array-like containing prices for slow raise FCAS (in $AUD per MWh) [1]
        l_lower_s: array-like containing prices for slow lower FCAS (in $AUD per MWh) [1]
        l_raise_d: array-like containing prices for delayed raise FCAS (in $AUD per MWh) [1]
        l_lower_d: array-like containing prices for delayed loewr FCAS (in $AUD per MWh) [1]
        l_raise_f: array-like containing prices for fast raise FCAS (in $AUD per MWh) [1]
        l_lower_f: array-like containing prices for fast loewr FCAS (in $AUD per MWh) [1]
        efficiency_in: the charging efficiency, as a proportion
        efficiency_out: the discharging efficiency, as a proportion
        p_min: minimum charge/discharge power limit (in MW)
        p_max: maximum charge/discharge power limit (in MW)
        soc_min: the minimum amount of stored energy (in MWh)
        soc_max: the maximum amount of stored energy (in MWh)
        prices_from: the start date to retrieve prices from
            `data/sa_fcas_data.csv`. If None, retrieves the first `n` prices
            from the price CSV.

    Notes:
        [1] For each of the `l_raise_*` and `l_lower_*` parameters, if it is
        `None`, then this function grabs appropriate price data from the
        `data/` directory of this project.
    """

    assert M > 0
    assert n > 0
    assert epsilon > 0

    m = gp.Model()
    m.Params.LogToConsole = 0

    T = [i for i in range(n)]

    # column vectors
    p_raise_s = m.addVars(T, vtype='C', name="p_raise_s", lb=p_min, ub=p_max)
    p_lower_s = m.addVars(T, vtype='C', name="p_lower_s", lb=p_min, ub=p_max)
    p_raise_d = m.addVars(T, vtype='C', name="p_raise_d", lb=p_min, ub=p_max)
    p_lower_d = m.addVars(T, vtype='C', name="p_lower_d", lb=p_min, ub=p_max)
    p_raise_f = m.addVars(T, vtype='C', name="p_raise_f", lb=p_min, ub=p_max)
    p_lower_f = m.addVars(T, vtype='C', name="p_lower_f", lb=p_min, ub=p_max)

    # column vectors
    b_raise_s = m.addVars(T, vtype='B', name="b_raise_s")
    b_lower_s = m.addVars(T, vtype='B', name="b_lower_s")
    b_raise_d = m.addVars(T, vtype='B', name="b_raise_d")
    b_lower_d = m.addVars(T, vtype='B', name="b_lower_d")
    b_raise_f = m.addVars(T, vtype='B', name="b_raise_f")
    b_lower_f = m.addVars(T, vtype='B', name="b_lower_f")

    # in $AUD, lists
    if l_raise_s is None:
        l_raise_s = data.get_sa_fcas_data(T, "RAISE60SECRRP", start_datetime=prices_from)
    if l_lower_s is None:
        l_lower_s = data.get_sa_fcas_data(T, "LOWER60SECRRP", start_datetime=prices_from)
    if l_raise_d is None:
        l_raise_d = data.get_sa_fcas_data(T, "RAISE5MINRRP", start_datetime=prices_from)
    if l_lower_d is None:
        l_lower_d = data.get_sa_fcas_data(T, "LOWER5MINRRP", start_datetime=prices_from)
    if l_raise_f is None:
        l_raise_f = data.get_sa_fcas_data(T, "RAISE6SECRRP", start_datetime=prices_from)
    if l_lower_f is None:
        l_lower_f = data.get_sa_fcas_data(T, "LOWER6SECRRP", start_datetime=prices_from)

    soc = m.addVars(T, vtype='C', name='soc', lb=soc_min, ub=soc_max)
    assert soc_min <= initial_soc and initial_soc <= soc_max

    # in hours
    delayed_delta = 5 / 60
    slow_delta = 1 / 60
    fast_delta = 1 / 10 / 60

    m.setObjective(sum(((l_raise_s[t] * p_raise_s[t] + l_lower_s[t] * p_lower_s[t]
                         + l_raise_d[t] * p_raise_d[t] + l_lower_d[t] * p_lower_d[t]
                         + l_raise_f[t] * p_raise_f[t] + l_lower_f[t] * p_lower_f[t]) * delayed_delta for t in T)), gp.GRB.MAXIMIZE)

    m.addConstr(soc[0] == initial_soc + p_lower_s[0] * slow_delta - p_raise_s[0] * slow_delta
                                      + p_lower_d[0] * delayed_delta - p_raise_d[0] * delayed_delta
                                      + p_lower_f[0] * fast_delta - p_raise_f[0] * fast_delta)

    for t in [i for i in T if i != 0]:
        m.addConstr(soc[t] == soc[t - 1] + p_lower_s[t] * slow_delta - p_raise_s[t] * slow_delta
                                         + p_lower_d[t] * delayed_delta - p_raise_d[t] * delayed_delta
                                         + p_lower_f[t] * delayed_delta - p_raise_f[t] * fast_delta)

    m.addConstrs((-p_raise_s[t] - M * (1 - b_raise_s[t]) <= -epsilon for t in T))
    m.addConstrs((p_raise_s[t] - M * b_raise_s[t] <= 0 for t in T))

    m.addConstrs((-p_raise_d[t] - M * (1 - b_raise_d[t]) <= -epsilon for t in T))
    m.addConstrs((p_raise_d[t] - M * b_raise_d[t] <= 0 for t in T))

    m.addConstrs((-p_raise_f[t] - M * (1 - b_raise_f[t]) <= -epsilon for t in T))
    m.addConstrs((p_raise_f[t] - M * b_raise_f[t] <= 0 for t in T))

    m.addConstrs((-p_lower_s[t] - M * (1 - b_lower_s[t]) <= -epsilon for t in T))
    m.addConstrs((p_lower_s[t] - M * b_lower_s[t] <= 0 for t in T))

    m.addConstrs((-p_lower_d[t] - M * (1 - b_lower_d[t]) <= -epsilon for t in T))
    m.addConstrs((p_lower_d[t] - M * b_lower_d[t] <= 0 for t in T))

    m.addConstrs((-p_lower_f[t] - M * (1 - b_lower_f[t]) <= -epsilon for t in T))
    m.addConstrs((p_lower_f[t] - M * b_lower_f[t] <= 0 for t in T))

    m.addConstrs((b_raise_s[t] + b_lower_s[t] + b_raise_d[t] + b_lower_d[t] + b_raise_f[t] + b_lower_f[t] <= 1 for t in T))

    return m
