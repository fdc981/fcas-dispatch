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
    n = len([v for v in m.getVars() if "p_raise_d" in v.VarName])

    T = [i for i in range(n)]

    p_raise_s = [m.getVarByName(f"p_raise_s[{i}]").x for i in T]
    p_lower_s = [m.getVarByName(f"p_lower_s[{i}]").x for i in T]

    p_raise_d = [(m.getVarByName(f"p_raise_d[{i}]").x if i % 1 == 0 else 0) for i in T]
    p_lower_d = [(m.getVarByName(f"p_lower_d[{i}]").x if i % 1 == 0 else 0) for i in T]

    p_raise_f = [(m.getVarByName(f"p_raise_f[{i}]").x if i % 1 == 0 else 0) for i in T]
    p_lower_f = [(m.getVarByName(f"p_lower_f[{i}]").x if i % 1 == 0 else 0) for i in T]

    b_raise_s = [m.getVarByName(f"b_raise_s[{i}]").x for i in T]
    b_lower_s = [m.getVarByName(f"b_lower_s[{i}]").x for i in T]

    b_raise_d = [(m.getVarByName(f"b_raise_d[{i}]").x if i % 1 == 0 else 0) for i in T]
    b_lower_d = [(m.getVarByName(f"b_lower_d[{i}]").x if i % 1 == 0 else 0) for i in T]

    b_raise_f = [(m.getVarByName(f"b_raise_f[{i}]").x if i % 1 == 0 else 0) for i in T]
    b_lower_f = [(m.getVarByName(f"b_lower_f[{i}]").x if i % 1 == 0 else 0) for i in T]

    soc = [m.getVarByName(f"soc[{i}]").x for i in T]

    df = pd.DataFrame(columns=["p_raise_f", "b_raise_f", "p_lower_f", "b_lower_f",
                               "p_raise_s", "b_raise_s", "p_lower_s", "b_lower_s",
                               "p_raise_d", "b_raise_d", "p_lower_d", "b_lower_d", "soc"], index=T)

    df["p_raise_s"] = p_raise_s
    df["p_lower_s"] = p_lower_s

    df["p_raise_d"] = p_raise_d
    df["p_lower_d"] = p_lower_d

    df["p_raise_f"] = p_raise_f
    df["p_lower_f"] = p_lower_f

    df["b_raise_s"] = b_raise_s
    df["b_lower_s"] = b_lower_s

    df["b_raise_d"] = b_raise_d
    df["b_lower_d"] = b_lower_d

    df["b_raise_f"] = b_raise_f
    df["b_lower_f"] = b_lower_f

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

    var_names = ["b_raise_f", "b_lower_f", "b_raise_s", "b_lower_s", "b_raise_d", "b_lower_d"]
    var_color = {"b_raise_f": "red",
                 "b_lower_f": "maroon",
                 "b_raise_s": "lime",
                 "b_lower_s": "darkgreen",
                 "b_raise_d": "skyblue",
                 "b_lower_d": "navy"}

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
