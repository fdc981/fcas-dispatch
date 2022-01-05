"""Optimisation problems expressed as gurobipy models."""

import gurobipy as gp
import numpy as np
import src.data as data
from decimal import Decimal


# TODO: add the efficiency coefficients to each model

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
        efficiency_in=0.92,
        efficiency_out=0.90,
        p_min=0,
        p_max=7,
        soc_min=0.35*13.5,
        soc_max=13.5
):
    """Creates an optimisation problem for slow and delayed FCAS dispatch with
    a Tesla Powerwall.

    Args:
        n: number of trading intervals
        M: value of big M for binary indicator constraints
        epsilon: an arbitrarily small value
        initial_soc: initial state of charge (in kWh)
        l_raise_s: array-like containing prices for slow raise FCAS [1]
        l_lower_s: array-like containing prices for slow lower FCAS [1]
        l_raise_d: array-like containing prices for delayed raise FCAS [1]
        l_lower_d: array-like containing prices for delayed loewr FCAS [1]
        efficiency_in: the charging efficiency, as a proportion
        efficiency_out: the discharging efficiency, as a proportion
        p_min: minimum charge/discharge power limit (in kW)
        p_max: maximum charge/discharge power limit (in kW)
        soc_min: the minimum amount of stored energy (in kWh)
        soc_max: the maximum amount of stored energy (in kWh)

    Notes:
        [1] If the parameter `l_raise_s`, `l_lower_s`, `l_raise_d` or
        `l_lower_d` is `None`, then this function grabs appropriate price data
        from the `data/` directory of this project.
    """

    assert M > 0
    assert n > 0
    assert epsilon > 0

    m = gp.Model()
    m.Params.LogToConsole = 0

    # expressed in decimals to avoid floating point errors
    T_d = [Decimal('%.1f' % i) for i in np.arange(0, n)]
    T_s = [Decimal('%.1f' % i) for i in np.arange(0, n, 1/5)]

    # column vectors
    p_raise_s = m.addVars(T_s, vtype='C', name="p_raise_s", lb=p_min, ub=p_max)
    p_lower_s = m.addVars(T_s, vtype='C', name="p_lower_s", lb=p_min, ub=p_max)
    p_raise_d = m.addVars(T_d, vtype='C', name="p_raise_d", lb=p_min, ub=p_max)
    p_lower_d = m.addVars(T_d, vtype='C', name="p_lower_d", lb=p_min, ub=p_max)

    # column vectors
    b_raise_s = m.addVars(T_s, vtype='B', name="b_raise_s")
    b_lower_s = m.addVars(T_s, vtype='B', name="b_lower_s")
    b_raise_d = m.addVars(T_d, vtype='B', name="b_raise_d")
    b_lower_d = m.addVars(T_d, vtype='B', name="b_lower_d")

    # in $AUD, lists
    if l_raise_s is None:
        l_raise_s = data.get_sa_dispatch_data(T_s, "RAISE60SECRRP")
    if l_lower_s is None:
        l_lower_s = data.get_sa_dispatch_data(T_s, "LOWER60SECRRP")
    if l_raise_d is None:
        l_raise_d = data.get_sa_dispatch_data(T_d, "RAISE5MINRRP")
    if l_lower_d is None:
        l_lower_d = data.get_sa_dispatch_data(T_d, "LOWER5MINRRP")

    soc = m.addVars(T_s, vtype='C', name='soc', lb=soc_min, ub=soc_max)
    assert soc_min <= initial_soc and initial_soc <= soc_max

    # in hours
    delayed_delta = 5 / 60
    slow_delta = 1 / 60

    m.setObjective(sum((l_raise_s[t] * p_raise_s[t] + l_lower_s[t] * p_lower_s[t] for t in T_s))
                   + sum((l_raise_d[t] * p_raise_d[t] + l_lower_d[t] * p_lower_d[t] for t in T_d)), gp.GRB.MAXIMIZE)

    m.addConstr(soc[0] == initial_soc + p_lower_s[0] * slow_delta - p_raise_s[0] * slow_delta
                                      + p_lower_d[0] * delayed_delta - p_raise_d[0] * delayed_delta)

    for t in [i for i in T_s if i != 0]:
        m.addConstr(soc[t] == soc[t - Decimal("0.2")] + p_lower_s[t] * efficiency_in * slow_delta - p_raise_s[t] / efficiency_out * slow_delta)
    for t in [i for i in T_d if i != 0]:
        m.addConstr(soc[t] == soc[t - Decimal("1")] + p_lower_d[t] * efficiency_in * delayed_delta - p_raise_d[t] / efficiency_out * delayed_delta)

    m.addConstrs((-p_raise_s[t] - M * (1 - b_raise_s[t]) <= -epsilon for t in T_s))
    m.addConstrs((p_raise_s[t] - M * b_raise_s[t] <= 0 for t in T_s))

    m.addConstrs((-p_raise_d[t] - M * (1 - b_raise_d[t]) <= -epsilon for t in T_d))
    m.addConstrs((p_raise_d[t] - M * b_raise_d[t] <= 0 for t in T_d))

    m.addConstrs((-p_lower_s[t] - M * (1 - b_lower_s[t]) <= -epsilon for t in T_s))
    m.addConstrs((p_lower_s[t] - M * b_lower_s[t] <= 0 for t in T_s))

    m.addConstrs((-p_lower_d[t] - M * (1 - b_lower_d[t]) <= -epsilon for t in T_d))
    m.addConstrs((p_lower_d[t] - M * b_lower_d[t] <= 0 for t in T_d))

    m.addConstrs((b_raise_s[t] + b_lower_s[t] <= 1 for t in T_s))
    m.addConstrs((b_raise_d[t] + b_lower_d[t] <= 1 for t in T_d))

    return m
