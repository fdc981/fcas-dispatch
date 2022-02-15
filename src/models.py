"""Optimisation problems expressed as gurobipy models."""

import gurobipy as gp
import numpy as np
import src.data as data
from decimal import Decimal
from itertools import product


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
        prices='auto',
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
        prices: a dictionary with an array of prices associated with each of
            the contingency FCAS services. The keys for each service should be
            as follows: `"lower_6_sec", "raise_6_sec", "lower_60_sec",
            "raise_60_sec", "lower_5_min", "raise_5_min"`.
        efficiency_in: the charging efficiency, as a proportion
        efficiency_out: the discharging efficiency, as a proportion
        p_min: minimum charge/discharge power limit (in MW)
        p_max: maximum charge/discharge power limit (in MW)
        soc_min: the minimum amount of stored energy (in MWh)
        soc_max: the maximum amount of stored energy (in MWh)
        prices_from: if `prices == 'auto'`, then this is the start date to
            retrieve prices from `data/sa_fcas_data.csv`. If None, retrieves
            the first `n` prices from the price CSV.
    """

    assert M > 0
    assert n > 0
    assert epsilon > 0

    m = gp.Model()
    m.Params.LogToConsole = 0

    T = [i for i in range(n)]

    F_lower = ["lower_6_sec", "lower_60_sec", "lower_5_min"]
    F_raise = ["raise_6_sec", "raise_60_sec", "raise_5_min"]
    F = F_lower + F_raise

    delta_t = {
        "lower_6_sec": 6 / 60 / 60,
        "lower_60_sec": 1 / 60,
        "lower_5_min": 5 / 60,
        "raise_6_sec": 6 / 60 / 60,
        "raise_60_sec": 1 / 60,
        "raise_5_min": 5 / 60,
    }

    p = m.addVars(F, T, vtype="C", name="p", lb=p_min, ub=p_max)
    b = m.addVars(F, T, vtype="B", name="b")

    if prices == 'auto':
        prices = {}
        prices["raise_6_sec"] = data.get_sa_fcas_data(T, "RAISE6SECRRP", start_datetime=prices_from)
        prices["lower_6_sec"] = data.get_sa_fcas_data(T, "LOWER6SECRRP", start_datetime=prices_from)
        prices["raise_60_sec"] = data.get_sa_fcas_data(T, "RAISE60SECRRP", start_datetime=prices_from)
        prices["lower_60_sec"] = data.get_sa_fcas_data(T, "LOWER60SECRRP", start_datetime=prices_from)
        prices["raise_5_min"] = data.get_sa_fcas_data(T, "RAISE5MINRRP", start_datetime=prices_from)
        prices["lower_5_min"] = data.get_sa_fcas_data(T, "LOWER5MINRRP", start_datetime=prices_from)

    soc = m.addVars(T, vtype='C', name='soc', lb=soc_min, ub=soc_max)
    assert soc_min <= initial_soc and initial_soc <= soc_max

    m.setObjective(sum((prices[f][t] * p[f, t] for f, t in product(F, T))) / 12, gp.GRB.MAXIMIZE)

    m.addConstr(soc[0] == initial_soc + sum((p[f, 0] * delta_t[f] for f in F_lower))
                                      - sum((p[f, 0] * delta_t[f] for f in F_raise)))

    m.addConstrs((soc[t] == soc[t-1] + sum((p[f, t] * delta_t[f] for f in F_lower))
                                     - sum((p[f, t] * delta_t[f] for f in F_raise))
                                     for t in T[1:]))

    m.addConstrs((-p[f, t] - M * (1 - b[f, t]) <= -epsilon for f, t in product(F, T)))
    m.addConstrs((p[f, t] - M * b[f, t] <= 0 for f, t in product(F, T)))

    m.addConstrs((sum((b[f, t] for f in F)) <= 1 for t in T))

    return m
