---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
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
from src.models import make_cooptimisation_model

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
plt.figure(figsize=(10, 7))
plt.hist(objective_values, bins=20)
plt.axvline(x=expected_model.ObjVal, c='red', label="Expected value model objective")
plt.axvline(x=np.mean(objective_values), c='orange', label="Mean of scenario models")
plt.title("Histogram of objective values of scenario models")
plt.legend()
```

```python
k = scipy.stats.gaussian_kde(objective_values)
```
