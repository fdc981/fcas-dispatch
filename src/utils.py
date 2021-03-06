"""Various utility functions for notebook usage."""

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


def show_solution(m, date_index=[i for i in range(1, 289)]):
    """Plot the state of charge of a solution made from optimizing
    make_cooptimisation_model, highlighting dispatch for each of the FCAS
    markets.

    Args:
        m: the optimized model to plot.
        date_index: an array-like of dates to plot

    Returns:
        None. Shows a plot of the solution.
    """
    plt.figure(figsize=(12, 7))
    plt.title("State of charge over time (enablements highlighted)")

    sol_df = tabulate_solution(m)

    plt.plot(date_index, sol_df["soc"])

    F_lower = ["lower_6_sec", "lower_60_sec", "lower_5_min"]
    F_raise = ["raise_6_sec", "raise_60_sec", "raise_5_min"]
    F = F_lower + F_raise

    var_names = [f"{f} enabled" for f in F]
    colors = ["palegoldenrod", "lightgreen", "cyan",
              "darkgoldenrod", "darkgreen", "darkcyan"]

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
