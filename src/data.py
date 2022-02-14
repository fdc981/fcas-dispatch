"""Functions for data retrieval and processing."""

import numpy as np
import pandas as pd
import pathlib
import bs4
import re
import shutil
import os
import requests
from src.utils import extract_tables


def get_sa_fcas_data(
        indices: list,
        column_name: str,
        repeat: int = 1,
        csv_path: str = "../data/sa_fcas_data.csv",
        start_datetime=None
) -> dict:
    """Retrieve some daily dispatch data.

    Args:
        indices (int): the number of entries required.
        column_name (str): the column name to retrieve. See the MMS Data Model
                           Report for more information.
        repeat (int): number of repetitions of the entries.
        csv_path (str): the relative path string of the csv file containing SA
                        FCAS prices.
        start_datetime (None or np.datetime64 or Date): the date and time of
            the first settlement to be retrieved. If None, retrieves the first
            `len(indices)` entries from `csv_path`.

    Returns:
        a dictionary of dispatch prices.
    """
    cols = ["SETTLEMENTDATE", "RRP", "LOWERREGRRP", "RAISEREGRRP",
            "LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP",
            "LOWER5MINRRP", "RAISE5MINRRP"]
    sa_price_df = pd.read_csv(csv_path, parse_dates=["SETTLEMENTDATE"])

    if start_datetime is None:
        start = 0
    else:
        start = -1

        for i in range(len(sa_price_df)):
            if start_datetime <= sa_price_df.loc[i, "SETTLEMENTDATE"]:
                start = i
                break

        if start == -1:
            raise Exception("No data from " + str(start_datetime) + " found.")

    assert column_name in cols
    assert len(indices) % repeat == 0

    num = len(indices) // repeat
    values = np.repeat(sa_price_df[column_name].values[start:start+num], repeat)

    assert len(values) == num * repeat

    return dict(zip(indices, values))


def download_reports(link, out_path="data/", skip=True, output=True):
    """Download reports from the `link` into the path `out_path`.

    Args:
        link (str): link to reports from the NEMWEB
        out_path (str): relative path string of the output folder
        skip (bool): if true, then skip already downloaded files.
        output (bool): if true, outputs messages

    Returns:
        None.
    """
    response = requests.get(link)
    soup = bs4.BeautifulSoup(response.text)
    zip_links = soup.find_all('a', string=re.compile('.zip'))

    # Download all zip files
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    for a_tag in zip_links:
        filename = a_tag.attrs['href'].split('/')[-1]

        if (not pathlib.Path("data/" + filename.replace("zip", "CSV")).exists()
           and not pathlib.Path("data/" + filename).exists()) or not skip:
            if output:
                print("downloading", filename)
            url = "https://www.nemweb.com.au" + a_tag.attrs['href']
            success = False

            while not success:
                response = requests.get(url)

                if response.status_code == 200:
                    with open(f"{out_path}/{filename}", 'wb') as f:
                        f.write(response.content)
                    success = True
                else:
                    choice = input(f"Download request returned with status {response.status_code}. Retry (y/N)?")
                    if choice != 'y':
                        success = True

    # Extract and remove each zip file
    zip_paths = pathlib.Path(out_path).glob('*.zip')

    for path in zip_paths:
        shutil.unpack_archive(str(path), str(path.parent))
        os.remove(str(path))


def extract_sa_fcas_data(data_path='data/'):
    """Extract the SA FCAS data from the daily trading and dispatch price data
    downloaded under `data_path`.

    Args:
        data_path: the relative path string of the folder containing trading
            and dispatch data CSVs.

    Returns:
        DataFrame of all SA FCAS prices, (FCAS trapezium-adjusted)
        availabilities, and dispatch amounts.
    """
    services = ["RAISE6SEC", "LOWER6SEC", "RAISE60SEC", "LOWER60SEC",
                "RAISE5MIN", "LOWER5MIN", "RAISEREG", "LOWERREG"]

    cols = ["SETTLEMENTDATE", "RRP"] \
        + [s + "RRP" for s in services] \
        + [s + "ACTUALAVAILABILITY" for s in services] \
        + [s + "LOCALDISPATCH" for s in services if "REG" not in s]

    sa_price_df = pd.DataFrame(columns=cols)

    csv_filenames = [p for p in pathlib.Path(data_path).glob("PUBLIC_PRICES*.CSV")]
    csv_filenames.sort()

    for csv_filename in csv_filenames:
        df = extract_tables(str(csv_filename))[1]

        sa_price_df = pd.concat([sa_price_df, df.loc[df["REGIONID"] == "SA1", cols]])

    sa_price_df["SETTLEMENTDATE"] = pd.to_datetime(sa_price_df["SETTLEMENTDATE"])

    assert sa_price_df["SETTLEMENTDATE"].is_monotonic_increasing

    return sa_price_df
