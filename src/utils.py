"""Various utility and helper functions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import shutil
from src.constants import services, F
from itertools import product


def tabulate_solution(m):
    """Tabulate the solution produced by a model.

    Args:
        m: the optimized Gurobi model

    Returns:
        A DataFrame tabulating values of the solution."""
    n = len([v for v in m.getVars() if "p[raise_6_sec" in v.VarName])

    T = [i for i in range(n)]

    F_lower = ["lower_6_sec", "lower_60_sec", "lower_5_min"]
    F_raise = ["raise_6_sec", "raise_60_sec", "raise_5_min"]
    F = F_lower + F_raise

    p = {}
    b = {}

    for f in F:
        p[f] = [m.getVarByName(f"p[{f},{i}]").x for i in T]
        b[f] = [m.getVarByName(f"b[{f},{i}]").x for i in T]

    soc = [m.getVarByName(f"soc[{i}]").x for i in T]

    df = pd.DataFrame(columns=[f"{f} dispatch" for f in F] + [f"{f} enabled" for f in F] + ["soc"], index=T)

    for f in F:
        df[f"{f} dispatch"] = p[f]
        df[f"{f} enabled"] = b[f]

    df["soc"] = soc

    return df


def show_solution(m, date_index=None):
    """Plot the state of charge of a solution made from optimizing
    make_cooptimisation_model, highlighting dispatch for each of the FCAS
    markets.

    Args:
        m: the optimized model to plot.
        date_index: an array-like of dates to plot

    Returns:
        None. Shows a plot of the solution.
    """
    n = len([v for v in m.getVars() if "p[raise_6_sec" in v.VarName])

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

    if date_index is None:
        date_index = [i for i in range(0, n+1)]

    initial_response = sum([m.getVarByName(f"p[{f},0]").x * delta_t[f] for f in F_lower]) \
        - sum([m.getVarByName(f"p[{f},0]").x * delta_t[f] for f in F_raise])
    after_initial_soc = m.getVarByName("soc[0]").x
    initial_soc = after_initial_soc - initial_response

    plt.figure(figsize=(12, 7))
    plt.title("State of charge over time (enablements highlighted)")

    sol_df = tabulate_solution(m)

    plt.plot(date_index, [initial_soc] + sol_df["soc"].values.tolist())

    F_lower = ["lower_6_sec", "lower_60_sec", "lower_5_min"]
    F_raise = ["raise_6_sec", "raise_60_sec", "raise_5_min"]
    F = F_lower + F_raise

    var_names = [f"{f} enabled" for f in F]
    colors = ["maroon", "darkgreen", "midnightblue",
              "coral", "lightgreen", "dodgerblue"]

    var_color = dict(zip(var_names, colors))

    plotted = {v: False for v in var_names}

    # note: lw=0 to ensure smooth borders
    for i in range(len(date_index)-1):
        for var_name in var_names:
            if sol_df.loc[i, var_name] == 1:
                plt.axvspan(
                    date_index[i],
                    date_index[i+1],
                    alpha=0.2,
                    color=var_color[var_name],
                    lw=0,
                    label="_"*plotted[var_name] + var_name,
                )
                plotted[var_name] = True

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("State of charge (in MWh)")

    plt.show()


def extract_tables(report_path: str) -> pd.DataFrame:
    """Extract tables from a given NEMWEB CSV report.

    Args:
        report_path: the path to the CSV report.

    Returns:
        A list of DataFrames, with each element containing a table in order of
        appearance on the original CSV report.
    """
    filename = pathlib.Path(report_path).name

    if report_path[-4:] == ".zip":
        shutil.unpack_archive(report_path, extract_dir="/tmp/")
        report_path = "/tmp/" + filename.replace(".zip", ".CSV")

    with open(report_path, 'r') as f:
        lines = f.readlines()
        indices = []

        for i, line in enumerate(lines):
            if line[0] == "I":
                indices.append(i)

        indices.append(i)

    dfs = []

    for i in range(len(indices)-1):
        dfs.append(pd.read_csv(report_path,
                               skiprows=indices[i],
                               nrows=(indices[i+1] - indices[i]-1)))

    return dfs


def calc_enablement_probs(
        num_scenarios,
        enablement_scenarios,
        enablement_probabilities,
        debug=True
):
    """Calculate the weights of each of the enablement scenarios, defined as
    the conditional probability that a scenario occurs given that the sampled
    set of scenarios holds.

    Args:
        num_scenarios: the number of scenarios
        enablement_scenarios: a dictionary with a matrix of enablement
            scenarios associated with each of the contingency FCAS.
            The keys for each service should be as follows: `"lower_6_sec",
            "raise_6_sec", "lower_60_sec", "raise_60_sec", "lower_5_min",
            "raise_5_min"`. Each array should be of size `n`.
        enablement_probabilities: a dictionary with a matrix of enablement
            probabilities associated with each of the contingency FCAS.
        debug: if true, output each of the log probability values,
            scenario weights, and weight sums.
    """
    S = [i for i in range(num_scenarios)]

    F_lower = ["lower_6_sec", "lower_60_sec", "lower_5_min"]
    F_raise = ["raise_6_sec", "raise_60_sec", "raise_5_min"]
    F = F_lower + F_raise

    scenario_log_probability = np.zeros((num_scenarios))
    for f in F:
        scenario_log_probability += np.sum(np.log(enablement_probabilities[f] * enablement_scenarios[f]
                                                  + (1 - enablement_probabilities[f]) * (1 - enablement_scenarios[f])), axis=1)

    scenario_weights = [1 / np.sum(np.exp(scenario_log_probability - scenario_log_probability[j])) for j in S]

    if debug:
        print("scenario_log_probability:", scenario_log_probability)
        print("scenario_weights:", scenario_weights)
        print("weight sum:", sum(scenario_weights))

    return scenario_weights


def load_fcas_prices():
    """Loads and returns the Pandas dataframe of the SA FCAS prices table.
    Returned dataframe contains additional columns containing enablement
    probabilities, with names ending in "PROB", and a "PERIODID" column,
    indicating time of trading interval in the day.
    """
    price_df = pd.read_csv("../data/sa_fcas_data.csv", parse_dates=["SETTLEMENTDATE"])
    price_df["PERIODID"] = price_df["SETTLEMENTDATE"].apply(
        lambda x: str(x.hour).zfill(2) + str(x.minute).zfill(2)
    )

    for s in services:
        price_df[s + "PROB"] = (price_df[s + "LOCALDISPATCH"] / price_df[s + "ACTUALAVAILABILITY"]).clip(upper=1)

    return price_df


def load_fcas_price_forecast():
    """Loads and returns the Pandas dataframe of the forecast for SA FCAS
    prices.
    """
    scenario_dfs = {
        s: pd.read_csv(f"../data/{s}RRP_modified_training.csv", parse_dates=["SETTLEMENTDATE"]) for s in services
    }

    return scenario_dfs


def is_quantile_crossing(quantiles):
    return np.any(np.diff(quantiles) < 0)


def inverse_cdf(y, quantile_cutoffs, quantile_forecasts):
    assert y >= 0 and y <= 1

    qi = None
    for i in range(len(quantile_cutoffs)-1):
        if y >= quantile_cutoffs[i] and y <= quantile_cutoffs[i+1]:
            qi = i
            break

    if qi is None:
        raise Exception(f"value {y} is not a quantile")

    rise = quantile_cutoffs[qi+1] - quantile_cutoffs[qi]
    run = quantile_forecasts[qi+1] - quantile_forecasts[qi]

    slope = rise/run

    return (y - quantile_cutoffs[qi]) / slope + quantile_forecasts[qi]


def inverse_cdf_parallel(y, quantile_cutoffs, quantile_values):
    """Return the inverse cdf of the values in `y` with the cdf being a linear
    approximation of the one provided by the quantiles.

    Args:
        y: a column vector of values.
        quantile_cutoffs: row vector of quantile cutoffs. For example, if the
            k-quantile is x then the quantile cutoff is k.
        quantile_values: row vector of corresponding quantile forecasts. For
            example, is the k-quantile is x then the quantile value is x.

    Returns:
        the inverse cdf of each value in y.
    """
    assert np.all(np.logical_and(y >= 0, y <= 1))
    assert y.ndim == quantile_cutoffs.ndim == quantile_values.ndim == 2

    y_location = np.logical_and(y >= quantile_cutoffs[:, :-1],
                                y < quantile_cutoffs[:, 1:])

    loc_found, qi = np.nonzero(y_location)

    assert np.all(loc_found == np.arange(y.size))

    rise = quantile_cutoffs[0, qi + 1] - quantile_cutoffs[0, qi]
    run = quantile_values[0, qi + 1] - quantile_values[0, qi]

    slope = rise/run

    return (y.T - quantile_cutoffs[0, qi]) / slope + quantile_values[0, qi]


def inverse_cdf_matrix(y, quantile_cutoffs, quantile_values):
    """Return the inverse cdf of the values in `y` with the cdf being a linear
    approximation of the one provided by the quantiles.

    Args:
        y:
            a matrix of values. Each row corresponds to one scenario. The
            matrix should be of shape `(num_scenarios, scenario_length)`.
        quantile_cutoffs:
            matrix of quantile cutoffs. For example, if the k-quantile is x
            then the quantile cutoff is k. Each row should contain the quantile
            cutoffs for the interval forecast of one trading interval. The
            matrix should be of shape `(scenario_length, num_quantiles)`.
        quantile_values:
            matrix of corresponding quantile forecasts. For example, is the
            k-quantile is x then the quantile value is x. The matrix should be
            of shape `(scenario_length, num_quantiles)`.

    Returns:
        matrix containing the inverse cdf of each corresponding value in `y`.
    """
    assert np.all(np.logical_and(y >= 0, y <= 1))

    # check dimensions
    assert y.ndim == quantile_cutoffs.ndim == quantile_values.ndim == 2
    assert quantile_values.shape == quantile_cutoffs.shape
    assert y.shape[1] == quantile_cutoffs.shape[0]

    qi = -np.ones(y.shape, dtype=int)

    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            for q in range(quantile_cutoffs.shape[1]-1):
                if (y[r, c] >= quantile_cutoffs[c, q] and y[r, c] <= quantile_cutoffs[c, q+1]):
                    qi[r, c] = q
                    break

    if np.any(qi == -1):
        raise Exception(f"The following y-values did not fall in any quantile: {y[qi == -1]}")

    row_index = np.arange(quantile_cutoffs.shape[0]).reshape((quantile_cutoffs.shape[0], 1))

    rise = quantile_cutoffs[row_index, qi.T + 1] - quantile_cutoffs[row_index, qi.T]
    run = quantile_values[row_index, qi.T + 1] - quantile_values[row_index, qi.T]

    slope = rise/run

    return (y - quantile_cutoffs[row_index, qi.T].T) / slope.T + quantile_values[row_index, qi.T].T


def calc_scenario_consts(
        price_scenarios,
        enablement_scenarios,
        enablement_probabilities,
        T,
        scenario_combine_method
):
    """Calculate the scenario constants from the given price and enablement
    scenarios. This is a helper function for `src.make_scenario_model()`.

    Args:
        price_scenarios: a dictionary with an array or matrix of prices
            associated with each of the contingency FCAS services. The keys for
            each service should be as follows: `"lower_6_sec", "raise_6_sec",
            "lower_60_sec", "raise_60_sec", "lower_5_min", "raise_5_min"`.
            When values are arrays, then they must be of length `n`. Otherwise
            the values are matrices, which should be of shape
            `(num_scenarios, n)`.
        enablement_scenarios: a dictionary with a matrix of enablement
            scenarios associated with each of the contingency FCAS.
            The keys for each service should be as follows: `"lower_6_sec",
            "raise_6_sec", "lower_60_sec", "raise_60_sec", "lower_5_min",
            "raise_5_min"`. Each array should be of size `n`.
        enablement_probabilities: a dictionary with a matrix of enablement
            probabilities associated with each of the contingency FCAS.
        T: set of indices used for trading intervals.
        scenario_combine_method: either 'product' or 'zip'.

    Returns:
        a dictionary with an array of scenario constants associated with each
        of the contingency FCAS servies.
    """
    num_price_scenarios = price_scenarios[F[0]].shape[0]
    num_enablement_scenarios = enablement_scenarios[F[0]].shape[0]

    scenarios = {}
    en_scenario_weights = calc_enablement_probs(num_enablement_scenarios,
                                                enablement_scenarios,
                                                enablement_probabilities)
    scenario_weights = {f: [] for f in F}

    if scenario_combine_method == 'product':
        scenario_consts = {}
        for f in F:
            for t in T:
                enablement_sum = np.dot(en_scenario_weights, enablement_scenarios[f][:, t])
                price_sum = np.sum(price_scenarios[f][:, t]) / num_price_scenarios

                scenario_consts[f, t] = enablement_sum * price_sum
    elif scenario_combine_method == 'zip':
        assert all((price_scenarios[f].shape == enablement_scenarios[f].shape for f in F))

        for f in F:
            scenarios[f] = price_scenarios[f] * enablement_scenarios[f]
            scenario_weights[f].append(scenario_weights)

        print("scenarios shapes:", [scenarios[f].shape for f in F])

        scenario_consts = {(f, t): np.dot(scenario_weights[f], scenarios[f][:, t]) for f, t in product(F, T)}
    else:
        raise Exception("Note: scenarios not done")

    print("scenario consts calculated")

    return scenario_consts
