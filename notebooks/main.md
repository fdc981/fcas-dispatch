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

# Optimisation experiments

```python tags=[]
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
import scipy
import sys

%load_ext autoreload
%autoreload 2

sys.path.insert(0, "../")
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
## Initial problem model
<!-- #endregion -->

```python tags=[]
m = gp.Model()

# number of consecutive trading intervals
n = 23
T = [i for i in range(n)]
```

```python tags=[]
# in kW
p_min = 0
p_max = 7

# column vectors
p_raise = m.addVars(n, vtype='C', name="p_raise", lb = p_min, ub = p_max)
p_lower = m.addVars(n, vtype='C', name="p_lower", lb = p_min, ub = p_max)

# column vectors
b_raise = m.addVars(n, vtype='B', name="b_raise")
b_lower = m.addVars(n, vtype='B', name="b_lower")

# in $AUD, lists
l_raise = np.array([0.66448] * n)
l_lower = np.array([0.15134] * n)

# in kW
soc_min = 0.35 * 13.5
soc_max = 13.5
soc = m.addVars(n, vtype='C', name='soc', lb = soc_min, ub = soc_max)
initial_soc = 6

# in hours
delayed_delta = 5 / 60

M = 10**20 # value with best results is M = p_max
epsilon = 0.00000001 # (default error is 10e-4)
```

```python tags=[]
# construct expressions for each SOC^t

m.addConstr(soc[0] == initial_soc + p_lower[0] * delayed_delta - p_raise[0] * delayed_delta)
for t in range(1, n):
    m.addConstr(soc[t] == soc[t-1] + p_lower[t] * delayed_delta - p_raise[t] * delayed_delta)
```

```python tags=[]
m.setObjective(sum((l_raise[t] * p_raise[t] + l_lower[t] * p_lower[t] for t in T)), gp.GRB.MAXIMIZE)
```

<!-- #region tags=[] -->
First, $P_{raise}^t > 0$ could be modelled to be $P_{raise}^t \geq \epsilon$ for some small $\epsilon > 0$, that is, $- P_{raise}^t \leq - \epsilon$.

- Now, $b^t_{raise} = 0 \implies P_{raise}^t = 0$, which in big $M$ formulation is $P_{raise}^t - M \cdot b_{raise}^t \leq 0$.
- Also, $b^t_{raise} = 1 \implies P_{raise}^t > 0$, which in big $M$ formulation is $-P_{raise}^t - M \cdot (1 - b_{raise}^t) \leq -\epsilon$.
<!-- #endregion -->

```python tags=[]
m.addConstrs((-p_raise[t] - M * (1 - b_raise[t]) <= -epsilon for t in T))
m.addConstrs((-p_lower[t] - M * (1 - b_lower[t]) <= -epsilon for t in T))

m.addConstrs((p_raise[t] - M * b_raise[t] <= 0 for t in T))
m.addConstrs((p_lower[t] - M * b_lower[t] <= 0 for t in T))

m.addConstrs((b_raise[t] + b_lower[t] <= 1 for t in T))
```

```python tags=[]
# Optimize, recording results and time taken
%time m.optimize()
```

```python tags=[]
# Print out raw solution
print("Solution:")
for var in m.getVars():
    if 'soc' in var.varName:
        print(var.VarName + ":", var.X)
    if 'p_raise' in var.varName:
        print(var.VarName + ":", var.X)
    if 'p_lower' in var.varName:
        print(var.VarName + ":", var.X)
```

```python
# Print out solution in a table. Note: columns are currently unlabeled
sol = np.array(m.X).reshape((5, n)).transpose()

display(pd.DataFrame(sol))
```

<!-- #region tags=[] -->
# Experiments
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
## Powerwall model
<!-- #endregion -->

```python
from src.models import make_powerwall_model

%load_ext autoreload
%autoreload 2
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Adjusting big $M$
<!-- #endregion -->

Experiments measuring node count and runtime for the powerwall model for $M = 7$ and for $M \in \{10^i : i = 1, \ldots, 100\}$ were done, fixing the parameter $n = 19$ and $\epsilon = 10^{-6}$.

Gurobi is deterministic, so each experiment from here on was only run once. The node count and runtime were recorded.

```python
results = []

for M in [7] + [10**i for i in range(1, 101)]:
    m = make_powerwall_model(n=19, M1=M, M2=M)
    m.optimize()
    results.append([M, m.NodeCount, m.Runtime])

results = np.array(results)
df = pd.DataFrame(results, columns=["M", "Node count", "Runtime"])
```

The results are shown below. The node count appears to increase from the baseline of 56355 when $M = 7$ and when $M \in \{10^6, 10^7, 10^{10}\}$ (the results for this $M$ can be seen at row index 6).

```python
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    display(df)
```

For the following, the effects on node count and runtime when $M = 7$ and at around $M = 10^6$ are investigated further by setting $n = 23$ (from $n = 19$).

```python
results = []

for M in [7] + [10**i for i in range(1, 10)]:
    m = make_powerwall_model(n=23, M1=M, M2=M)
    m.optimize()
    results.append([M, m.NodeCount, m.Runtime])

results = np.array(results)
df = pd.DataFrame(results, columns=["M", "Node count", "Runtime"])
```

The results are shown below. There is larger node count for $M = 7$ compared to $M = 10^6$.

```python
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    display(df)
```

For the following, we steadily increase $M = 7$ up until $M = 8.9$ (that is, $M \in \{7.0, 7.1, 7.2, \ldots, 8.9\}$) to see precisely where the node count drops.

```python
results = []

for M in [7 + i/10 for i in range(20)]:
    m = make_powerwall_model(n=23, M1=M, M2=M)
    m.optimize()
    results.append([M, m.NodeCount, m.Runtime])

results = np.array(results)
df = pd.DataFrame(results, columns=["M", "Node count", "Runtime"])
```

We find that a sudden decrease to "optimal" node count happens when going from $M = 8$ to $M = 8.1$.

```python
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    display(df)
```

Finally, values of $M < 7$ were investigated. Here we fix $n = 19$.

```python
results = []

for M in [7 - i/100 for i in range(1, 50)]:
    m = make_powerwall_model(n=19, M1=M, M2=M)
    m.optimize()
    results.append([M, m.NodeCount, m.Runtime, m.ObjVal])

results = np.array(results)
df = pd.DataFrame(results, columns=["M", "Node count", "Runtime", "Objective value"])
```

We find varying node count and also worse objective value than usual.

```python
plt.figure(figsize=(16, 6), dpi=80)
plt.xticks([7 - i/50 for i in range(1, 25)])
plt.plot(df["M"], df["Node count"])

plt.gca().twinx()
plt.plot(df["M"], df["Objective value"])
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Adjusting $\epsilon$
<!-- #endregion -->

In the following $\epsilon$ was changed, fixing $M = 14$ and $n = 19$. From initial experimentation, it was found that for $\epsilon \in [10^{-323}, 10^{-9}]$, no change in node count was observed, so only results for $\epsilon \in [10^{-9}, 1]$ are presented. Note that $\epsilon = 10^{-324} = 0$ in Python.

```python
results = []

for epsilon in [10**(-i) for i in range(10)]:
    m = make_powerwall_model(n=19, epsilon=epsilon)
    m.optimize()
    results.append([epsilon, m.NodeCount, m.Runtime, m.ObjVal])

results = np.array(results)
df = pd.DataFrame(results, columns=["epsilon", "Node count", "Runtime", "Objective value"])
```

The objective value was

```python
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    display(df)
```

In the previous section higher node count was observed with $M = 7$. This is probably even if $b_{raise}^t = b_{lower}^t = 1$ then $P_{raise}^t, P_{lower}^t \leq M$, in practice .

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Initial SOC
<!-- #endregion -->

Different values of the intial state of charge, denoted $SOC^{-1}$ (), produces different results. For the following, we vary $SOC^{-1}$ between values $13.5 \times 35\%$ to $13.5$. In particular 300 linearly spaced values over the interval $[13.5 \times 0.35, 13.5]$ for $SOC^{-1}$ were tested.

```python
results = []

for initial_soc in np.linspace(13.5 * 0.35, 13.5, num=500):
    m = make_powerwall_model(n=19, initial_soc=initial_soc)
    m.optimize()
    results.append([initial_soc, m.NodeCount, m.Runtime, m.ObjVal])

results = np.array(results)
df = pd.DataFrame(results, columns=["Initial SOC", "Node count", "Runtime", "Objective value"])
```

The results below show variable node count during the optimisation with smaller $SOC^{-1}$, and smaller node count for $SOC^{-1} \geq 8$. The objective value also increases as $SOC^{-1}$ increases.

```python
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Initial SOC')
ax1.set_ylabel('Node count', color=color)
ax1.plot(df["Initial SOC"], df["Node count"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Objective value', color=color)
ax2.plot(df["Initial SOC"], df["Objective value"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Presence of SOC constraint
<!-- #endregion -->

The presence of the SOC constraint and decision variable appears to increase optimisation performance. Without it, there is a possibility of numerical errors propagating through the expanded recurrence as in the source code of `make_alt_powerwall_model` (as Gurobi does not simplify algebraic expressions). There is also the loss of convenience of not having the state of charge automatically calculated by the solver.

Below, an alternative model which excludes the SOC constraint and instead expands the recurrence in terms of $P_{raise}^t$ and $P_{lower}^t$ is compared with the usual powerwall model.

```python
from src.models import make_alt_powerwall_model
```

```python
M1 = 14
M2 = M1
epsilon = 10**(-6)
initial_soc = 6
```

Experiments were done with fixed parameters above and varying $n = 6, \ldots, 25$, and the results from optimising both models were compared.

```python
df = pd.DataFrame(columns=["n", "M1 objective", "M1 node count", "M1 work", "M2 objective", "M2 node count", "M2 work"])

for i, n in enumerate(range(6, 26)):
    m1 = make_powerwall_model(M1=M1, M2=M2, n=n, epsilon=epsilon, initial_soc=initial_soc)
    m1.optimize()
    
    m2 = make_alt_powerwall_model(M1=M1, M2=M2, n=n, epsilon=epsilon, initial_soc=initial_soc)
    m2.optimize()
    
    df = df.append({'n': n,
                    'M1 objective': m1.ObjVal,
                    'M1 node count': m1.NodeCount, 
                    'M1 work': m1.Work,
                    'M2 objective': m2.ObjVal,
                    'M2 node count': m2.NodeCount,
                    'M2 work': m2.Work}, ignore_index=True)
```

First, the objective values from each of the models appear to be similar. 

```python
plt.scatter(df["n"], df['M1 objective'])
plt.scatter(df["n"], df['M2 objective'])
plt.xticks(range(6, 26))
plt.show()
```

Indeed, the distances between corresponding objective values were less than the error tolerance $10^{-6}$.

```python
all(abs(df.loc[:, "M1 objective"] - df.loc[:, "M2 objective"]) < 10**(-6))
```

More importantly, we may observe that there is usually larger node count from optimizing the alternative model as $n$. For these results, $n = 21$ was the only exception. The graph below shows this, with the vertical axes ticks scaled by $10^6$.

```python
plt.plot(df["n"], df['M1 node count'])
plt.plot(df["n"], df['M2 node count'])
plt.xticks(range(6, 26))
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Why branch and bound only begins at $n \geq 23$
<!-- #endregion -->

During initial experimentation, the Gurobi solver only began using branch and bound when $n \geq 23$ - for $n < 23$ the solver only explores one node. We look into the SOC:

```python
n = 23

m = make_powerwall_model(n=n)
m.optimize()
```

```python
soc = [var.x for var in m.getVars() if "soc" in var.VarName]

plt.scatter(range(n), soc)
```

<!-- #region tags=[] -->
## Co-optimisation model
<!-- #endregion -->


```python
from src.models import make_cooptimisation_model
from decimal import Decimal
```

```python tags=[]
def tabulate_solution(m):
    """Tabulate the solution produced by a model. Returns a DataFrame."""
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

    df = pd.DataFrame(columns = ["p_raise_f", "b_raise_f", "p_lower_f", "b_lower_f",
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
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
### Initial experimentation
<!-- #endregion -->

```python
n = 30
m = make_cooptimisation_model(n = n)
```

```python
m.optimize()
display(m.ObjVal, m.Runtime, m.NodeCount, m.IterCount)
```

```python
df = tabulate_solution(m)
```

```python
# Verify the answer by checking whether only one binary variable is 1 at every moment
all(df["b_raise_f"] + df["b_raise_s"] + df["b_raise_d"] + df["b_lower_f"] + df["b_lower_s"] + df["b_lower_d"] <= 1)
```

```python
with pd.option_context("display.max_rows", None, "display.max_columns", None) as p:
    display(df.round(3))
```

```python
# Graph the state of charge over time.
plt.plot(df.index, df["soc"])
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
### Day-ahead optimisation
<!-- #endregion -->

```python
plt.rcParams.update({'font.size': 13})
```

```python
n = 24 * 60 // 5
m = make_cooptimisation_model(n=n, 
                              soc_min=0,
                              soc_max=3,
                              initial_soc=1.5,
                              p_min=0,
                              p_max=0.5)
```

```python
# Limit processing power.
m.setParam("Threads", 1)

m.optimize()
```

```python
print("Runtime:", m.Runtime)
print("Node count:", m.NodeCount)
```

```python
df = tabulate_solution(m)
```

```python
with pd.option_context("display.max_rows", None, "display.max_columns", None) as p:
    display(df.round(3))
```

```python
import src.data as data

date_index = data.sa_price_df["SETTLEMENTDATE"].values
```

```python
# Graph the state of charge over time.
plt.figure(figsize=(12,7))
plt.title("State of charge over time")
plt.plot(date_index, df["soc"])
plt.xlabel("Index of trading interval")
plt.ylabel("State of charge (in MWh)")
plt.grid(which='major', color='lightgrey')
plt.show()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
### Results for day-ahead optimisation
<!-- #endregion -->

```python
cols = ["b_raise_f", "b_lower_f", "b_raise_s", "b_lower_s", "b_raise_d", "b_lower_d"]
names = ["Raise 6 sec", "Lower 6 sec", "Raise 60 sec", "Lower 60 sec", "Raise 5 min", "Lower 5 min"]

plt.figure(figsize=(12,7))

plt.title("Enablement frequency")
plt.bar(names, df[cols].sum())
plt.xlabel("FCAS Product")
plt.ylabel("Frequency")
plt.gca().set_axisbelow(True)
plt.grid(which='major', axis='y', color='lightgrey')

plt.show()
```

```python
import src.data as data

cols = ["LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP", "LOWER5MINRRP", "RAISE5MINRRP"]

plt.figure(figsize=(12,7))

plt.title("FCAS prices for 2021-12-11")
plt.plot(date_index, data.sa_price_df[cols])
plt.xlabel("Date and time (MM-DD HH)")
plt.ylabel("Price ($AUD per MWh)")
plt.grid(which='major', color='lightgrey')
plt.legend(cols)

plt.show()
```

```python
plt.figure(figsize=(12,7))

plt.title("FCAS prices for 2021-12-11")
plt.plot(data.sa_price_df["SETTLEMENTDATE"], data.sa_price_df[cols].max(axis=1))
```

```python
argmax = []
for i in data.sa_price_df.index:
    argmax.append(cols[np.argmax(data.sa_price_df.loc[i, cols])])

plt.figure(figsize=(12,7))

plt.title("Number of times an FCAS product was at max price")
plt.bar(["Lower 60 sec", "Raise 6 sec", "Raise 60 sec"], pd.value_counts(argmax))
plt.xlabel("FCAS Product")
plt.ylabel("Frequency")
plt.gca().set_axisbelow(True)
plt.grid(which='major', axis='y', color='lightgrey')

plt.show()
```

```python
plt.figure(figsize=(12,7))
plt.title("State of charge over time (enablements highlighted)")
plt.plot(date_index, df["soc"])

b_raise_f_plotted = 0
b_lower_s_plotted = 0
b_raise_s_plotted = 0
b_raise_d_plotted = 0

# note: lw=0 to ensure smooth borders
for i in range(len(date_index)-1):
    if df.loc[i, "b_raise_f"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='orange', lw=0, label = "_"*b_raise_f_plotted + "Raise 6 sec")
        b_raise_f_plotted = 1
    if df.loc[i, "b_lower_s"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='green', lw=0, label = "_"*b_lower_s_plotted + "Lower 60 sec")
        b_lower_s_plotted = 1
    if df.loc[i, "b_raise_s"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='red', lw=0, label = "_"*b_raise_s_plotted + "Raise 60 sec")
        b_raise_s_plotted = 1
    if df.loc[i, "b_raise_d"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='brown', lw=0, label = "_"*b_raise_d_plotted + "Raise 5 min")
        b_raise_d_plotted = 1

plt.legend()
plt.xlabel("Index of trading interval")
plt.ylabel("State of charge (in MWh)")
        
plt.show()
```

```python
cols = ["p_lower_f", "p_raise_f", "p_lower_s", "p_raise_s", "p_lower_d", "p_raise_d"]
power_sum = df[cols].sum(axis=1)

power_sum.value_counts()
```

```python
power_sum[2]
```

```python
plt.figure(figsize=(12,7))
plt.title("State of charge over time (enablements at max price highlighted)")
plt.plot(date_index, df["soc"])

b_raise_f_plotted = 0
b_lower_s_plotted = 0
b_raise_s_plotted = 0
b_raise_d_plotted = 0

# note: lw=0 to ensure smooth borders
for i in range(len(date_index)-1):
    if df.loc[i, "b_raise_f"] == 1 and argmax[i] == "RAISE6SECRRP":
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='orange', lw=0, label = "_"*b_raise_f_plotted + "Raise 6 sec")
        b_raise_f_plotted = 1
    if df.loc[i, "b_lower_s"] == 1 and argmax[i] == "LOWER60SECRRP":
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='green', lw=0, label = "_"*b_lower_s_plotted + "Lower 60 sec")
        b_lower_s_plotted = 1
    if df.loc[i, "b_raise_s"] == 1 and argmax[i] == "RAISE60SECRRP":
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='red', lw=0, label = "_"*b_raise_s_plotted + "Raise 60 sec")
        b_raise_s_plotted = 1

plt.legend()
plt.xlabel("Index of trading interval")
plt.ylabel("State of charge (in kWh)")
        
plt.show()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
### Varying initial SOC
<!-- #endregion -->

```python
soc_params = np.linspace(0, 3, 100)
soc_obj_vals = []
n = 24 * 60 // 5

for initial_soc in soc_params:
    print(f"optimizing with initial_soc={initial_soc}")
    m = make_cooptimisation_model(n=n, 
                                  soc_min=0,
                                  soc_max=3,
                                  initial_soc=initial_soc,
                                  p_min=0,
                                  p_max=0.5)
    m.optimize()
    soc_obj_vals.append(m.ObjVal)
```

```python
plt.figure(figsize=(12,7))
plt.rc('font', size=17)
plt.title("Objective value (revenue) against initial SOC")
plt.plot(soc_params, soc_obj_vals)
plt.xlabel("Initial SOC (in MWh)")
plt.ylabel("Objective value/Revenue (in $AUD)")
plt.grid(which='major', color='lightgrey')
plt.rc('font', size=13)
```

```python
params = np.linspace(0, 20, 50)
obj_vals = []

for p_max in params:
    print(f"optimizing with p_max={p_max}")
    m = make_cooptimisation_model(n=n, 
                                  soc_min=0,
                                  soc_max=3,
                                  initial_soc=1.5,
                                  p_min=0,
                                  p_max=p_max)
    m.Params.Threads = 1
    m.Params.WorkLimit = 8
    m.optimize()
    obj_vals.append(m.ObjVal)
```

```python
plt.figure(figsize=(12,7))
plt.rc('font', size=17)
plt.title("Objective value (revenue) with different P_max")
plt.plot(params, obj_vals)
plt.xlabel("P_max (in MW)")
plt.ylabel("Objective value/Revenue (in $AUD)")
plt.grid(which='major', color='lightgrey')

plt.show()
plt.rc('font', size=13)
```

```python
socmax_params = np.linspace(1.5, 20, 50)
socmax_obj_vals = []

for soc_max in socmax_params:
    print(f"optimizing with soc_max={soc_max}")
    m = make_cooptimisation_model(n=n, 
                                  soc_min=0,
                                  soc_max=soc_max,
                                  initial_soc=1.5,
                                  p_min=0,
                                  p_max=0.5)
    m.Params.Threads = 1
    m.Params.WorkLimit = 8
    m.optimize()
    socmax_obj_vals.append(m.ObjVal)
```

```python
plt.figure(figsize=(12,7))
plt.rc('font', size=17)
plt.title("Objective value (revenue) with different SOC_max")
plt.plot(socmax_params, socmax_obj_vals)
plt.xlabel("SOC_max (in MWh)")
plt.ylabel("Objective value/Revenue (in $AUD)")
plt.grid(which='major', color='lightgrey')

plt.show()
plt.rc('font', size=13)
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
### Forecasting

Requires existing daily dispatch data. This can be downloaded with `download.py`.
<!-- #endregion -->

```python
import pathlib
```

```python
help(data_path.iterdir)
```

```python
filtered_filenames.sort()
filtered_filenames[0]
```

```python
data_path = pathlib.Path("../data/")
data_path.mkdir(parents=True, exist_ok=True)

cols = ["SETTLEMENTDATE", "RRP", "LOWERREGRRP", "RAISEREGRRP",
        "LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP",
        "LOWER5MINRRP", "RAISE5MINRRP"]

sa_price_df = pd.DataFrame(columns = cols)

filtered_filenames = [path for path in data_path.iterdir() if "CSV" in str(path) and str(path) < "../data/PUBLIC_DAILY_202112110000_20211212040503.CSV"]

print(filtered_filenames)

for csv_filename in filtered_filenames:
    print("exporting from:", csv_filename)
    i_indices = []

    with open(csv_filename, 'r') as f:
        contents = f.read()
        for i, line in enumerate(contents.split('\n')):
            if len(line) > 0 and line[0] == 'I':
                i_indices.append(i)

    df = pd.read_csv(csv_filename,
                     skiprows=i_indices[1],
                     nrows=(i_indices[2] - i_indices[1]-1))

    new_df = df.loc[df["REGIONID"] == "SA1", cols]
    new_df["SETTLEMENTDATE"] = pd.to_datetime(new_df["SETTLEMENTDATE"])
    
    sa_price_df = sa_price_df.append(new_df)
```

```python
forecast = pd.DataFrame(columns=["LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP", "LOWER5MINRRP", "RAISE5MINRRP"])

rng = np.random.default_rng(seed=1)

for col_name in forecast.columns:
    forecast[col_name] = [max(0, rng.normal(loc=sa_price_df[col_name].mean(),
                                            scale=sa_price_df[col_name].std())) for _ in range(288)]
```

```python
plt.figure(figsize=(12,7))
plt.title("Generated FCAS price forecast")
plt.plot(date_index, forecast)
plt.legend(forecast.columns)
plt.xlabel("Date and time (MM-DD HH)")
plt.ylabel("Price ($AUD per MWh)")
plt.grid(which='major', axis='y', color='lightgrey')

plt.show()
```

```python
m = make_cooptimisation_model(n=n, 
                              soc_min=0,
                              soc_max=3,
                              initial_soc=1.5,
                              p_min=0,
                              p_max=0.5,
                              l_raise_f=forecast["RAISE6SECRRP"],
                              l_lower_f=forecast["LOWER6SECRRP"],
                              l_raise_s=forecast["RAISE60SECRRP"],
                              l_lower_s=forecast["LOWER60SECRRP"],
                              l_raise_d=forecast["RAISE5MINRRP"],
                              l_lower_d=forecast["LOWER5MINRRP"])
m.Params.Threads = 1
m.optimize()
```

```python
df = tabulate_solution(m)
```

```python
plt.figure(figsize=(12,7))
plt.title("State of charge over time")
plt.plot(date_index, df["soc"])

b_lower_f_plotted = 0
b_raise_f_plotted = 0
b_lower_s_plotted = 0
b_raise_s_plotted = 0
b_lower_d_plotted = 0
b_raise_d_plotted = 0

# note: lw=0 to ensure smooth borders
for i in range(len(date_index)-1):
    if df.loc[i, "b_lower_f"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='blue', lw=0, label = "_"*b_lower_f_plotted + "Lower 6 sec")
        b_lower_f_plotted = 1
    if df.loc[i, "b_raise_f"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='orange', lw=0, label = "_"*b_raise_f_plotted + "Raise 6 sec")
        b_raise_f_plotted = 1
    if df.loc[i, "b_lower_s"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='green', lw=0, label = "_"*b_lower_s_plotted + "Lower 60 sec")
        b_lower_s_plotted = 1
    if df.loc[i, "b_raise_s"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='red', lw=0, label = "_"*b_raise_s_plotted + "Raise 60 sec")
        b_raise_s_plotted = 1
    if df.loc[i, "b_lower_d"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='purple', lw=0, label = "_"*b_lower_d_plotted + "Lower 5 min")
        b_lower_d_plotted = 1
    if df.loc[i, "b_raise_d"] == 1:
        plt.axvspan(date_index[i], date_index[i+1], alpha=0.2, color='brown', lw=0, label = "_"*b_raise_d_plotted + "Raise 5 min")
        b_raise_d_plotted = 1
        
plt.legend()
plt.xlabel("Index of trading interval")
plt.ylabel("State of charge (in MWh)")
plt.show()
```

```python
cols_f = forecast.columns

argmax = []
for i in forecast.index:
    argmax.append(cols_f[np.argmax(forecast.loc[i, cols_f])])

plt.figure(figsize=(12,7))

plt.title("Number of times an FCAS product was at max price")
plt.bar(["Lower 60 sec", "Raise 6 sec", "Raise 60 sec", "Lower 6 sec", "Lower 5 min", "Raise 5 min"], pd.value_counts(argmax))
plt.xlabel("FCAS Product")
plt.ylabel("Frequency")
plt.gca().set_axisbelow(True)
plt.grid(which='major', axis='y', color='lightgrey')

plt.show()
```

```python
df[[name for name in df.columns if "p_" in name]].sum(axis=1)
```

```python
forecast_obj_vals = []

print("seed:", end='')
for seed in range(1000):
    print(seed, end=' ')
    new_forecast = pd.DataFrame(columns=["LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP", "LOWER5MINRRP", "RAISE5MINRRP"])

    rng = np.random.default_rng(seed=seed)

    for col_name in new_forecast.columns:
        new_forecast[col_name] = [max(0, rng.normal(loc=sa_price_df[col_name].mean(),
                                                    scale=sa_price_df[col_name].std())) for _ in range(288)]
    m = make_cooptimisation_model(n=n, 
                                  soc_min=0,
                                  soc_max=3,
                                  initial_soc=1.5,
                                  p_min=0,
                                  p_max=0.5,
                                  l_raise_f=new_forecast["RAISE6SECRRP"],
                                  l_lower_f=new_forecast["LOWER6SECRRP"],
                                  l_raise_s=new_forecast["RAISE60SECRRP"],
                                  l_lower_s=new_forecast["LOWER60SECRRP"],
                                  l_raise_d=new_forecast["RAISE5MINRRP"],
                                  l_lower_d=new_forecast["LOWER5MINRRP"])
    m.Params.Threads = 1
    m.Params.WorkLimit = 10
    m.optimize()
    
    forecast_obj_vals.append(m.ObjVal)
    
print()
```

```python
plt.figure(figsize=(12,7))

plt.title("Histogram of objective values using 1000 random forecasts")
plt.hist(forecast_obj_vals)
plt.xlabel("Objective value/revenue (in $AUD)")
plt.ylabel("Frequency")
plt.gca().set_axisbelow(True)
plt.grid(which='major', axis='y', color='lightgrey')
```

```python
# Nothing interesting revealed by histogram
relevant_df = sa_price_df[["LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP", "LOWER5MINRRP", "RAISE5MINRRP"]]

plt.hist(relevant_df, range=(0, 1000), stacked=True)
```

```python

```
