"""Various utility and helper functions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import shutil


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

    initial_response = sum([m.getVarByName(f"p[{f},0]").x * delta_t[f] for f in F])
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


def enablement_scenario_weights(
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
    for s in S:
        for f in F:
            scenario_log_probability[s] += np.log(enablement_probabilities[f] * enablement_scenarios[f][s]
                                                  + (1 - enablement_probabilities[f]) * (1 - enablement_scenarios[f][s])).sum()

    scenario_weights = [1 / sum((np.exp(scenario_log_probability[i] - scenario_log_probability[j]) for i in S)) for j in S]

    if debug:
        print("scenario_log_probability:", scenario_log_probability)
        print("scenario_weights:", scenario_weights)
        print("weight sum:", sum(scenario_weights))

    return scenario_weights
