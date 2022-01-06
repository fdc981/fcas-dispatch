"""Data retrieval functions."""

import pandas as pd
import pathlib

cols = None
sa_price_df = None

if __name__ == "src.data":
    data_path = pathlib.Path("../data/")
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        csv_filename = str(next((data_path).glob("*.CSV")))
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


def get_sa_dispatch_data(indices: list, column_name: str) -> list:
    """Retrieve some daily dispatch data."""
    global cols
    global sa_price_df

    assert column_name in cols

    num = len(indices)

    assert len(sa_price_df[column_name].values[:num]) == num

    pairs = zip(indices, sa_price_df[column_name].values[:num])

    return dict(pairs)
