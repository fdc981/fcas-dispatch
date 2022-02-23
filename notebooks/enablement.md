---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: fcas-project
    language: python
    name: fcas-project
---

<!-- #region tags=[] -->
# Modelling the enablement probability
<!-- #endregion -->

```python tags=[]
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4
import re
import statsmodels.api as sm
import scipy.stats
import seaborn as sns
import lightgbm as lgb

%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, "../")

from src.utils import *
from src.models import *

plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=13)
```

<!-- #region tags=[] -->
## Modelling enablement probability with LOCALDISPATCH and ACTUALAVAILABILITY
<!-- #endregion -->

Dispatch is enablement of a response (https://energy-rules.aemc.gov.au/ner/367/glossary/d). If the dispatch and capacity data is already known beforehand, then we can estimate the probability of enablement; since dispatch is how much FCAS is enabled and capacity is how much FCAS was available (assuming that we also participated), then dividing dispatch by capacity provides an estimate. However, in practice the amount of FCAS to be dispatched and the total capacity reserved is unknown (although total FCAS reserves may be available via Yesterday's Bids). One way would be to estimate the distribution of both dispatch and capacity, and find some kind of 95% confidence interval (though this turns out to be mostly infeasible, see the Aside). This section seeks to explore this.


First, the prices are loaded:

```python
price_df = pd.read_csv("../data/sa_fcas_data.csv", parse_dates=["SETTLEMENTDATE"])
```

We provide an ID to each trading interval of each day, based on the hour and minute of the settlement. 

```python
price_df["PERIODID"] = price_df["SETTLEMENTDATE"].apply(lambda x : str(x.hour).zfill(2) + str(x.minute).zfill(2))

display(price_df["PERIODID"])
```

For each service, the two columns of interest, `LOCALDISPATCH` and `ACTUALAVAILABILITY`, are both assumed to be in MW respectively.


## Naive enablement


We can then provide a naive estimate of enablement probability. For a trading interval $t$, our block of $P_t$ MW to be dispatched could lie at any position along the sorted blocks of offers depending on how we price it in the offer. But more simply, it could just be a point lying anywhere within 0 and `ACTUALAVAILABILITY` of the service inclusive. If we further assume that this point has equal chance to lie on each position, the probability of enablement is `LOCALDISPATCH / ACTUALAVAILABILITY`.

Now, some `ACTUALAVAILABILITY` is less than `LOCALDISPATCH`. In these cases, we have guaranteed enablement, and so the probability will be 1. This is taken into account via the `pd.DataFrame.clip`:

```python tags=[]
services = ["RAISE5MIN", "LOWER5MIN", "RAISE60SEC", "LOWER60SEC", "RAISE6SEC", "LOWER6SEC"]
for s in services:
    price_df[s + "PROB"] = (price_df[s + "LOCALDISPATCH"] / price_df[s + "ACTUALAVAILABILITY"]).clip(upper=1)
```

The graph of probabilities can be found, for example in 2022-01-29:

```python tags=[]
sample_df = price_df[price_df["SETTLEMENTDATE"].dt.date == pd.Timestamp("2022-01-29").date()]

sample_df[["SETTLEMENTDATE"] + [s + "PROB" for s in services]].plot(
    fontsize=13,
    figsize=(17, 10),
    title="FCAS enablement probability for 2022-01-01 (estimated via dispatch/availability, clipped to 1)",
    x="SETTLEMENTDATE",
    ylabel="Probability"
)
```

Just in case of price corrections, we check that there are no duplicated entries:

```python
sample_df[sample_df["SETTLEMENTDATE"].duplicated(keep=False)]
```

Assuming that enablement is modelled using a Bernoulli variable, then, for each probability vector, if we multiply it pointwise with the price vector, we get the expected price. This can be fed into the model:

```python
expected_model = make_cooptimisation_model(
    n=288,
    soc_min=0,
    soc_max=3,
    initial_soc=1.5,
    p_min=0,
    p_max=0.5,
    prices={
        "lower_6_sec": sample_df["LOWER6SECRRP"].values * sample_df["LOWER6SECPROB"].values,
        "raise_6_sec": sample_df["RAISE6SECRRP"].values * sample_df["RAISE6SECPROB"].values,
        "lower_60_sec": sample_df["LOWER60SECRRP"].values * sample_df["LOWER60SECPROB"].values,
        "raise_60_sec": sample_df["RAISE60SECRRP"].values * sample_df["RAISE60SECPROB"].values,
        "lower_5_min": sample_df["LOWER5MINRRP"].values * sample_df["LOWER5MINPROB"].values,
        "raise_5_min": sample_df["RAISE5MINRRP"].values * sample_df["RAISE5MINPROB"].values
    }
)
```

For comparison we create a model using the original prices:

```python
deterministic_model = make_cooptimisation_model(
    n=288,
    soc_min=0,
    soc_max=3,
    initial_soc=1.5,
    p_min=0,
    p_max=0.5,
    prices={
        "lower_6_sec": sample_df["LOWER6SECRRP"].values,
        "raise_6_sec": sample_df["RAISE6SECRRP"].values,
        "lower_60_sec": sample_df["LOWER60SECRRP"].values,
        "raise_60_sec": sample_df["RAISE60SECRRP"].values,
        "lower_5_min": sample_df["LOWER5MINRRP"].values,
        "raise_5_min": sample_df["RAISE5MINRRP"].values
    }
)
```

Optimisation is then performed with these two models:

```python
expected_model.optimize()
deterministic_model.optimize()

show_solution(expected_model)
show_solution(deterministic_model)
```

We find for this day in particular, two differing solutions. '

Since the expected prices are fractions of the original prices, the expected model has lower objective value than the deterministic model.

```python
expected_model.ObjVal
```

```python
deterministic_model.ObjVal
```

We now construct 100 scenarios using the probabilities obtained above, performing a deterministic optimisation:

```python
rng = np.random.default_rng(seed=2021)
```

```python
objective_values = []
num_scenarios = 10000

for i in range(num_scenarios):
    scenario_df = pd.DataFrame(rng.binomial(1, sample_df[[s + "PROB" for s in services]]),
                               columns=[s + "ENABLED" for s in services])
    
    scenario_model = make_cooptimisation_model(
        n=288,
        soc_min=0,
        soc_max=3,
        initial_soc=1.5,
        p_min=0,
        p_max=0.5,
        prices={
            "lower_6_sec": sample_df["LOWER6SECRRP"].values * scenario_df["LOWER6SECENABLED"].values,
            "raise_6_sec": sample_df["RAISE6SECRRP"].values * scenario_df["RAISE6SECENABLED"].values,
            "lower_60_sec": sample_df["LOWER60SECRRP"].values * scenario_df["LOWER60SECENABLED"].values,
            "raise_60_sec": sample_df["RAISE60SECRRP"].values * scenario_df["RAISE60SECENABLED"].values,
            "lower_5_min": sample_df["LOWER5MINRRP"].values * scenario_df["LOWER5MINENABLED"].values,
            "raise_5_min": sample_df["RAISE5MINRRP"].values * scenario_df["RAISE5MINENABLED"].values
        }
    )
    
    scenario_model.Params.Threads = 1
    scenario_model.optimize()
    objective_values.append(scenario_model.ObjVal)
    
    if i % 100 == 0:
        print(i)
```

Unfortunately, the proportion of objective values that are worse than the stochastic model is very low, so in general the scenario models improve on the stochastic model.


This can be illustrated with a histogram:

```python
arr = [i for i in objective_values if i != -np.inf]
```

```python tags=[]
plt.figure(figsize=(10, 7))
plt.hist(arr, bins=20)
plt.axvline(x=expected_model.ObjVal, c='red', label="Expected value model objective")
plt.axvline(x=np.mean(arr), c='orange', label="Mean of scenario models")
plt.title("Histogram of objective values of scenario models")
plt.legend()
```

We attempt a scenario-based optimisation to overcome this. The following is a work-in-progress:

```python
prob_df = price_df[price_df["SETTLEMENTDATE"].between(pd.Timestamp("2022-01-29"), pd.Timestamp("2022-01-30"), inclusive='left')][[s + "PROB" for s in services]]
```

```python
rng = np.random.default_rng(seed=1)
num_scenarios = 100

enablement_scenarios = {
    "lower_6_sec": rng.binomial(1, prob_df["LOWER6SECPROB"], size=(num_scenarios, 288)),
    "raise_6_sec": rng.binomial(1, prob_df["RAISE6SECPROB"], size=(num_scenarios, 288)),
    "lower_60_sec": rng.binomial(1, prob_df["LOWER60SECPROB"], size=(num_scenarios, 288)),
    "raise_60_sec": rng.binomial(1, prob_df["RAISE60SECPROB"], size=(num_scenarios, 288)),
    "lower_5_min": rng.binomial(1, prob_df["LOWER5MINPROB"], size=(num_scenarios, 288)),
    "raise_5_min": rng.binomial(1, prob_df["RAISE5MINPROB"], size=(num_scenarios, 288))
}

enablement_probabilities = {
    "lower_6_sec": prob_df["LOWER6SECPROB"].values,
    "raise_6_sec": prob_df["RAISE6SECPROB"].values,
    "lower_60_sec": prob_df["LOWER60SECPROB"].values,
    "raise_60_sec": prob_df["RAISE60SECPROB"].values,
    "lower_5_min": prob_df["LOWER5MINPROB"].values,
    "raise_5_min": prob_df["RAISE5MINPROB"].values
}

ms = make_scenario_model(
    n=288,
    soc_min=0,
    soc_max=3,
    initial_soc=1.5,
    p_min=0,
    p_max=0.5,
    prices_from=pd.Timestamp("2022-01-29"),
    num_scenarios=num_scenarios,
    enablement_scenarios=enablement_scenarios,
    enablement_probabilities=enablement_probabilities
)
```

```python
ms.optimize()
```
```python
tabulate_solution(ms)
```

```python
ms.ObjVal
```

```python
show_solution(ms)
```

This solution chooses the max expected price, as shown below:

```python
weights = np.array(
    [2.6213691657961058e-21, 1.2226405496105753e-27, 2.544382418575854e-26, 2.1042802661064757e-28, 2.9713246270786774e-30, 3.263120741411554e-34, 5.474467450112092e-35, 2.507528407632176e-10, 1.5678248373004914e-08, 1.739426210666378e-22, 1.6536701613246296e-23, 2.232341557744545e-25, 5.931110028025331e-24, 3.375170194037694e-35, 1.014356335648746e-12, 0.9999999838835822, 2.868692778125398e-20, 3.43217324582967e-21, 1.3917907866017224e-26, 9.683827851997672e-41, 1.0770457476903445e-13, 6.848237974657089e-26, 1.8733300844391512e-36, 2.2060526551444808e-34, 2.229233627540206e-29, 9.698785371834155e-35, 4.880085814451911e-39, 5.653863496914764e-24, 7.552650664769611e-31, 1.7055550635778583e-15, 1.9958722683331736e-18, 7.408790351700401e-21, 2.5351663415529056e-30, 1.82120813363724e-12, 3.1086350742022237e-13, 8.156803293998408e-28, 3.7912515011506934e-33, 5.190375211672366e-25, 2.430725977257849e-28, 2.3009995263105903e-23, 1.42673570641854e-20, 3.8884285974129916e-30, 5.6481981783458356e-14, 3.602445897051469e-22, 4.3399735593171006e-13, 8.978037709884418e-32, 2.524009349033974e-33, 4.415496745409943e-19, 1.2688025407260119e-28, 2.275765505046974e-27, 1.6297809381162665e-24, 5.200482566262684e-35, 3.3711083510958284e-30, 9.961492513278606e-23, 1.0203632610666329e-39, 7.822921164447502e-26, 4.680102276604843e-20, 3.9508158784859893e-26, 1.577935748820397e-31, 3.0918134097690213e-21, 1.0989535555443783e-17, 5.991594744294355e-11, 2.3378724073955927e-21, 4.1751873383796333e-16, 3.678875642261077e-23, 1.4745583245814052e-24, 2.1481861616840143e-30, 1.5521364439436075e-18, 1.3204743168332643e-33, 1.1250318198158334e-38, 4.514186132846059e-19, 6.943555663765836e-35, 3.7418495987504166e-17, 7.548424651253122e-26, 3.7455464751461817e-31, 1.153893395385135e-31, 6.443077459020507e-27, 1.0292831674907895e-41, 1.0446409666712677e-16, 3.755618927225943e-30, 3.2354752281115518e-19, 1.2891477643643042e-26, 1.8581724844941088e-28, 4.45854880611292e-25, 2.0487188232724192e-30, 1.0840172691953114e-18, 3.713161696145085e-21, 1.554921672353685e-25, 4.198263511886603e-12, 2.038873217271102e-19, 9.799194722589153e-24, 1.6000460849054768e-11, 7.422873579465244e-17, 1.035552623255617e-10, 3.880488896864059e-37, 7.462377230259266e-29, 1.4332663301201078e-39, 4.598507632566949e-18, 8.128406556048925e-38, 7.521367433908631e-18]
)
```

```python
services_l = ["lower_6_sec", "raise_6_sec", "lower_60_sec", "raise_60_sec", "lower_5_min", "raise_5_min"]
services = [s.replace("_", "").upper() for s in services_l]
relevant_prices = price_df[price_df["SETTLEMENTDATE"].between(pd.Timestamp("2022-01-29"), pd.Timestamp("2022-01-30"), inclusive='left')][[s + "RRP" for s in services]]
```

```python
for s, s_l in zip(services, services_l):
    # row-wise multiplication
    scenario_prices = relevant_prices[s + "RRP"].values * enablement_scenarios[s_l]
    
    plt.plot((weights.T.reshape((1, 100)) @ scenario_prices)[0], label=s_l)
    
plt.legend()
plt.show()
```

```python

```
