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
# Scenario modelling
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

## Intro

For scenario-based optimisation, the following are important: the generation of many scenarios via forecasting, and the reduction of many scenarios (say in the 1000s) to just a representative few. Sections within this notebook attempt to address this.

```python
price_df = pd.read_csv("../data/sa_fcas_data.csv", parse_dates=["SETTLEMENTDATE"])
```

We provide an ID to each trading interval of each day, based on the hour and minute of the settlement. 

```python
price_df["PERIODID"] = price_df["SETTLEMENTDATE"].apply(lambda x : str(x.hour).zfill(2) + str(x.minute).zfill(2))

display(price_df["PERIODID"])
```

Also, a list of names of services and (naive) enablement probabilities for each service is provided.

```python tags=[]
services = ["RAISE5MIN", "LOWER5MIN", "RAISE60SEC", "LOWER60SEC", "RAISE6SEC", "LOWER6SEC"]

for s in services:
    price_df[s + "PROB"] = (price_df[s + "LOCALDISPATCH"] / price_df[s + "ACTUALAVAILABILITY"]).clip(upper=1)
```

### First attempts at fitting a normal distribution


In this section, an estimation of the distributions is found. For each service at PERIODID 0h10m, we can see that the data does not follow a similar distribution:

```python
services = ["RAISE5MIN", "LOWER5MIN", "RAISE60SEC", "LOWER60SEC", "RAISE6SEC", "LOWER6SEC"]
sample_df = price_df.query("PERIODID == '0010'")[[s + "LOCALDISPATCH" for s in services]]
sample_df.hist(bins=20, figsize=(10, 10))
plt.show()
```

We test whether the data is normally distributed using a QQ-plot.

```python
services = ["RAISE5MIN", "LOWER5MIN", "RAISE60SEC", "LOWER60SEC", "RAISE6SEC", "LOWER6SEC"]
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# flatten to a 1D array
axs = axs.flatten()

for i, s in enumerate(services):
    sm.qqplot(sample_df[f"{s}LOCALDISPATCH"], ax=axs[i])
    axs[i].set_title(s)
    
fig.tight_layout()
```

For the lower 5 min, lower 60 sec, raise 6 sec and lower 6 sec services, with the exception of a few outliers, the points on the QQ plot lie on a relatively straight line, so the normality assumption holds in those samples. However, the QQ plot for the raise 60 second service shows some right-skew, whilst the QQ plot for the raise 5 minute service shows left-skew of the data. Since these samples are non-symmetric, the data overall is not entirely normally distributed.

Suppose we remove the outliers. Normality???


The Lilliefors test shows that most of the data samples does not come from a normal distribution (with null hypothesis that the distribution is normal, and $\alpha = 0.05$). However, this is including the outlying values.


```python
for s in services:
    print(s + ":", sm.stats.diagnostic.lilliefors(price_df[price_df['PERIODID'] == '0010'][s + "LOCALDISPATCH"]))
```

<!-- #region tags=[] -->
## Exploration for causality

Some pairs of variables may be related. 
<!-- #endregion -->

For a FCAS, because the marginal price is the highest offer enabled after enabling a certain amount (i.e. `LOCALDISPATCH`), and the marginal price is co-optimised with energy prices for participants in ancillary service and energy markets to generate the clearing price (i.e. `RRP`), it must be that `LOCALDISPATCH` causes `RRP` in some fashion. A glance of the plots of lower 60 second dispatch and regional reference price reveals some possible correlation, where sprice spikes occur when dispatch peaks:

```python
sample_df = price_df[price_df["SETTLEMENTDATE"].between(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-10"))]
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(sample_df["SETTLEMENTDATE"], sample_df["LOWER60SEC" + "LOCALDISPATCH"], label="Lower 60 sec local dispatch")

ax2 = ax.twinx()

ax2.plot(sample_df["SETTLEMENTDATE"], sample_df["LOWER60SEC" + "RRP"], c="orange", label="Lower 60 sec regional reference price")
 
fig.tight_layout()
fig.legend()
```

```python
sample_df = price_df[price_df["SETTLEMENTDATE"].between(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-10"))]
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(sample_df["SETTLEMENTDATE"], sample_df["LOWER60SEC" + "LOCALDISPATCH"], label="Lower 60 sec local dispatch")

ax2 = ax.twinx()

#ax.plot(sample_df["SETTLEMENTDATE"], sample_df["LOWER60SEC" + "ACTUALAVAILABILITY"], c="orange", label="Lower 60 sec regional reference price")
 
fig.tight_layout()
fig.legend()
```

But there is no correlation in general:

```python
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs = axs.flatten()

for i, s in enumerate(services):
    axs[i].scatter(price_df[s + "LOCALDISPATCH"], price_df[s + "RRP"], alpha=0.1)
    axs[i].set_title(s)
    axs[i].set_xlabel(s + "LOCALDISPATCH")
    axs[i].set_ylabel(s + "RRP")
    
fig.tight_layout()
```

The graphs above are scatter plots of the respective data, however the colour of the points are made to be translucent to expose density (darker areas thus indicate a clustering of points). We observe multiple outlying values, possibly corresponding to price spikes. We see that majority values lie near `RRP == 0`, something more apparent in the 2D histograms below (with log transformation applied to histogram frequencies):

```python
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs = axs.flatten()
num_bins = 50

for i, s in enumerate(services):
    hist_2d = np.histogram2d(price_df[s + "LOCALDISPATCH"], price_df[s + "RRP"], bins=num_bins)
    img = axs[i].imshow(np.log(hist_2d[0]).transpose(), origin='lower')
    plt.colorbar(img, ax=axs[i])
    axs[i].set_title(s)
    axs[i].set_xlabel(s + "LOCALDISPATCH")
    axs[i].set_ylabel(s + "RRP")
    axs[i].set_xticks(range(0, num_bins+1, 10), hist_2d[1][range(0, num_bins+1, 10)].round(1))
    axs[i].set_yticks(range(0, num_bins+1, 10), hist_2d[2][range(0, num_bins+1, 10)].round(1))
    
fig.tight_layout()
```

So, the data points where `RRP < 20` were plotted. However, no visible correction can be inferred. We note, however, the existence of "row clusters" where the FCAS price appears to remain regardless of dispatch:

```python
sample_df = price_df[(price_df[[s + "RRP" for s in services]] < 20).all(axis=1)]
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs = axs.flatten()

for i, s in enumerate(services):
    axs[i].scatter(sample_df[s + "LOCALDISPATCH"], sample_df[s + "RRP"], alpha=0.1)
    axs[i].set_title(s)
    axs[i].set_xlabel(s + "LOCALDISPATCH")
    axs[i].set_ylabel(s + "RRP")
    
fig.tight_layout()
```

### Rolling correlations and an attempt at linear regression


If we look at adjacent trading intervals, we may observe correlations:

```python
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs = axs.flatten()

for i, s in enumerate(services):
    axs[i].scatter(price_df[price_df["PERIODID"] == "0000"][s + "RRP"],
                   price_df[price_df["PERIODID"] == "0005"][s + "RRP"], alpha=0.3)
    axs[i].set_title(s)
    axs[i].set_xlabel("At 0000")
    axs[i].set_ylabel("At 0005")
    
fig.tight_layout()
```

```python
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs = axs.flatten()

for i, s in enumerate(services):
    axs[i].scatter(price_df[price_df["PERIODID"] == "0000"][s + "LOCALDISPATCH"],
                   price_df[price_df["PERIODID"] == "0005"][s + "LOCALDISPATCH"], alpha=0.3)
    axs[i].set_title(s)
    axs[i].set_xlabel("At 0000")
    axs[i].set_ylabel("At 0005")
    
fig.tight_layout()
```

```python
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs = axs.flatten()

for i, s in enumerate(services):
    axs[i].scatter(price_df[price_df["PERIODID"] == "0000"][s + "ACTUALAVAILABILITY"],
                   price_df[price_df["PERIODID"] == "0005"][s + "ACTUALAVAILABILITY"], alpha=0.3)
    axs[i].set_title(s)
    axs[i].set_xlabel("At 0000")
    axs[i].set_ylabel("At 0005")
    
fig.tight_layout()
```

Since the data lies on a line, we may attempt linear regression. But first, we should check whether all relevant pairs of data are correlated:

```python
# Make a sorted list of all groups.
groups = [g for g in price_df.groupby(["PERIODID"])]
groups.sort(key=lambda g: g[0])
groups = [{"period": g[0], "group": g[1].reset_index(drop=True)} for g in groups]
correlations = {}

# Iterate through adjacent groups.
for current, next in zip(groups, groups[1:]):
    correlations[current["period"]] = (current["group"].corrwith(next["group"]))
```

The rolling correlations (using the Pearson correlation coefficient) for `LOCALDISPATCH` is shown below. Note that there are more downward spikes in correlation if duplicates exist.

```python
cols = "LOCALDISPATCH"
plot_index = [pd.Timestamp(c[:2] + ":" + c[2:]) for c in correlations.keys()]

plt.figure(figsize=(17, 10))
plt.plot(plot_index, [correlations[c][[s + cols for s in services]] for c in correlations.keys()])
plt.title("Rolling correlations (for adjacent trading intervals) for %s" % cols)

#ax2 = plt.gca().twinx()
#ax2.plot(plot_index + [pd.Timestamp("23:55")], price_df.groupby("PERIODID")["LOWER60SEC" + cols].mean())
plt.legend([s + cols for s in services])
```

This spike appears to be happening at a specific point in time:

```python
min_key = "0000"
min_val = 1

for c in correlations.keys():
    if correlations[c]["RAISE60SECLOCALDISPATCH"] < min_val:
        min_val = correlations[c]["RAISE60SECLOCALDISPATCH"]
        min_key = c
```

```python
min_val, min_key
```

This shows that the price spike occurs at the start of the trading day, which is 00:00 EST or 04:00 AEST. This may make sense if NEMDE finds a separate dispatch per trading day. 

The graphs of the rolling correlation at 0400 is shown below:

```python
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs = axs.flatten()

for i, s in enumerate(services):
    axs[i].scatter(price_df[price_df["PERIODID"] == "0400"][s + "LOCALDISPATCH"],
                   price_df[price_df["PERIODID"] == "0405"][s + "LOCALDISPATCH"], alpha=0.3)
    axs[i].set_title(s)
    axs[i].set_xlabel("At 0400")
    axs[i].set_ylabel("At 0405")
    
fig.tight_layout()
```

Hence, in a trading day, except from the trading intervals connecting one day and another, we have consistently high rolling correlation across all 6 ancillary services. 

With a high correlation, a linear model may be suitable. We attempt to fit a linear model for LOWER6SECLOCALDISPATCH samples from adjacent trading intervals 0000 and 0005:

```python
endog = groups[0]['group']["LOWER6SECLOCALDISPATCH"].values
exog = groups[1]['group']["LOWER6SECLOCALDISPATCH"].values

plt.scatter(endog, exog)
```

The ordinary least squares model was used for fitting, and the results are shown below.

```python
exog = sm.add_constant(exog)

lm = sm.OLS(endog, exog)
```

```python
results = lm.fit()
```

```python
results.summary()
```

The results above show that the residuals of the model are not normally distributed (this is indicated by the p-value being less than 0.05 for the Jarque-Bera test). The residual plot for this model supports this by exhibiting heteroskedasticity (more positive residuals compared to negative residuals):

```python
fig = sm.graphics.plot_regress_exog(results, "x1")
fig.tight_layout()
plt.show()
```

Hence linear modelling here is infeasible.

<!-- #region tags=[] -->
# KDE and $k$-means clustering for scenario generation and reduction (WIP)

The KDE estimates the probability distribution with no assumptions. As an example, the graph below shows the kernel density estimation on each of the FCAS dispatch data at the trading interval ending at 0010 (TODO: find the default parameters used for kernel density estimation):
<!-- #endregion -->

```python
sample_df = price_df.query("PERIODID == '0010'")[[s + "LOCALDISPATCH" for s in services]]
sample_df.plot.kde()
```

Histograms of the corresponding data are provided for comparison:

```python
sample_df.plot.hist(bins=80)
```

A KDE model is created for each LOCALDISPATCH amount at each trading period for each contingency  FCAS:

```python
localdispatch_kde = {g["period"]: {} for g in groups}

for g in groups:
    for service in services:
        localdispatch_kde[g["period"]][service] = scipy.stats.gaussian_kde(g["group"][service + "LOCALDISPATCH"])
```

By sampling from each KDE for each trading period for a single service, we can build up a scenario. Here, we sample from the `LOWER60SEC` service, and create 1000 scenarios:

```python
num_scenarios = 1000
service = "LOWER60SEC"

dispatch_scenarios = []

for _ in range(num_scenarios):
    scenario = []
    for g in groups:
        scenario.append(localdispatch_kde[g["period"]][service].resample(1)[0, 0])
    dispatch_scenarios.append(scenario)
```

We reduce the scenarios by partitioning the data into $k$ clusters with $k$-means clustering. At the end of the algorithm, a sample of $k$ scenarios, one from each partition, is returned. Here, $k = 10$. 

```python
dispatch_scenarios = np.array(dispatch_scenarios)

codebook, distortion = scipy.cluster.vq.kmeans(dispatch_scenarios, 10, iter=100)
```

The graph of all the scenarios below exhibits a trend:

```python
plt.title("Scenarios for LOWER60SECLOCALDISPATCH")
plt.plot(codebook.transpose())
plt.legend([f"Scenario {i}" for i in range(10)])
plt.show()
```

In general, the real LOWER60SECLOCALDISPATCH data deviates from the scenarios, and appears to have some structure such as remaining constant for periods of time:

```python
sample1 = price_df[price_df["SETTLEMENTDATE"].between(pd.Timestamp("2021-04-02"), pd.Timestamp("2021-04-03"))]["LOWER60SECLOCALDISPATCH"].values
sample2 = price_df[price_df["SETTLEMENTDATE"].between(pd.Timestamp("2021-04-03"), pd.Timestamp("2021-04-04"))]["LOWER60SECLOCALDISPATCH"].values
sample3 = price_df[price_df["SETTLEMENTDATE"].between(pd.Timestamp("2021-04-04"), pd.Timestamp("2021-04-05"))]["LOWER60SECLOCALDISPATCH"].values

plt.title("Comparison of scenario and real LOWER60SECLOCALDISPATCH data")
plt.plot(codebook.transpose(), alpha=0.1)
plt.plot(sample1, linewidth=1.5, label="sample1")
plt.plot(sample2, linewidth=1.5, label="sample2")
plt.plot(sample3, linewidth=1.5, label="sample3")
plt.legend()
plt.show()
```

The scenario data also appears to fall around the mean:

```python
sample = price_df.groupby("PERIODID")["LOWER60SECLOCALDISPATCH"].mean().values

plt.title("Comparison of scenario and real LOWER60SECLOCALDISPATCH data")
plt.plot(codebook.transpose(), alpha=0.1)
plt.plot(sample, linewidth=1.5)
plt.show()
```

Creating a scenario with structured data is important - a scenario with no structure could provide an inaccurate forecast.


### Scenarios for enablement probability

We now turn to creating scenarios for the enablement probability.

```python
price_df[[s + "PROB" for s in services]].plot.hist(bins=100)
```

The kernel density estimates are as provided:

```python
price_df[[s + "PROB" for s in services]].plot.kde()
```

Note that some of the probabilities from the kernel density estimate can be negative and even be greater than 1, which should be impossible. So, for this tentative version, when sampling from these kernel density estimates, if the value falls outside of the interval $[0, 1]$ then we resample.

Similar to the previous section, KDEs are created for each group of `PROB` data (grouped by `PERIODID`):

```python
prob_kde = {g["period"]: {} for g in groups}

for g in groups:
    for s in services:
        prob_kde[g["period"]][s] = scipy.stats.gaussian_kde(g["group"][s + "PROB"])
```

A large number of scenarios is then generated with these KDEs. Here, a scenario contains prices for all 6 contingency FCAS, instead of just one as in the previous section:

```python
num_scenarios = 100
prob_scenarios = np.ndarray((num_scenarios, 288 * 6))

for n in range(num_scenarios):
    for i, s in enumerate(services):
        for j, g in enumerate(groups):
            sample = -1
            while sample < 0 or sample > 1:
                sample = prob_kde[g["period"]][s].resample(1)[0, 0]
            prob_scenarios[n, j + i*288] = sample
```

This is reduced to just 10 scenarios with $k$-means clustering:

```python
prob_scenarios = np.array(prob_scenarios)

codebook, distortion = scipy.cluster.vq.kmeans(prob_scenarios, 10, iter=100)
```

Each scenario can then be graphed:

```python
for i, s in enumerate(services):
    plt.plot(codebook[0, i*288:(i+1)*288], label=s)

plt.legend()
```

# $k$-medioids

Since $k$-means returns centroids and not actual data points (which may or may not distort the shape of the scenario), we can use the alternative $k$-medioids for clustering, which use actual data points as its centroids.
