import pandas as pd
from src.scenario import get_top_scenarios
from src.constants import services
import argparse


price_df = pd.read_csv("data/sa_fcas_data.csv", parse_dates=["SETTLEMENTDATE"])

for s in services:
    price_df[s + "PROB"] = (price_df[s + "LOCALDISPATCH"] / price_df[s + "ACTUALAVAILABILITY"]).clip(upper=1)

price_df["PERIODID"] = price_df["SETTLEMENTDATE"].apply(lambda x : str(x.hour).zfill(2) + str(x.minute).zfill(2))

prob_df = price_df.groupby("PERIODID")[[s + "PROB" for s in services]].mean().reset_index()

num_scenarios = 200
scenario_length = 24

res = get_top_scenarios(num_scenarios, prob_df[services[0] + 'PROB'][:scenario_length])

print(res)
