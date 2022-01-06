"""Data retrieval functions."""

import numpy as np
import pandas as pd
import pathlib

cols = None
sa_price_df = None

if __name__ == "src.data":
    data_path = pathlib.Path("../data/")
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        csv_filename = str(next((data_path).glob("*.CSV")))
        print("Retrieving data from", csv_filename)
    except Exception as e:
        print("No dispatch CSV files found.")
        raise e

    i_indices = []

    with open(csv_filename, 'r') as f:
        contents = f.read()
        for i, line in enumerate(contents.split('\n')):
            if len(line) > 0 and line[0] == 'I':
                i_indices.append(i)

    df = pd.read_csv(csv_filename,
                     skiprows=i_indices[1],
                     nrows=(i_indices[2] - i_indices[1]-1))

    cols = ["SETTLEMENTDATE", "RRP", "LOWERREGRRP", "RAISEREGRRP",
            "LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP",
            "LOWER5MINRRP", "RAISE5MINRRP"]

    sa_price_df = df.loc[df["REGIONID"] == "SA1", cols]

    sa_price_df["SETTLEMENTDATE"] = pd.to_datetime(sa_price_df["SETTLEMENTDATE"])

def get_sa_dispatch_data(indices: list, column_name: str, repeat: int = 1) -> dict:
    """Retrieve some daily dispatch data.

    Args:
        indices (int): the number of entries required.
        column_name (str): the column name to retrieve. See the MMS Data Model
                           Report for more information.
        repeat (int): number of repetitions of the entries.

    Returns:
        a dictionary of dispatch prices."""
    global cols
    global sa_price_df

    assert column_name in cols
    assert len(indices) % repeat == 0

    num = len(indices) // repeat
    values = np.repeat(sa_price_df[column_name].values[:num], repeat)

    assert len(values) == num * repeat

    return dict(zip(indices, values))
