---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region tags=[] -->
# Data analysis
<!-- #endregion -->

```python
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2
```

<!-- #region tags=[] -->
### Scraper JSON
<!-- #endregion -->

```python
import json
import datetime
```

```python
tables = ""
with open("results-10-12-2021.json", 'r') as f:
    tables = json.loads(f.read())
```

```python
# TODO: make string dates into actual Date objects using time.strptime

sa_price_df = pd.DataFrame(columns = ["Date", "Energy", "Raise Reg", "Lower Reg", "Raise 6 sec", "Raise 60 sec", "Raise 5 min", "Lower 6 sec", "Lower 60 sec", "Lower 5 min"])
for i, table in enumerate(tables):
    date = datetime.datetime.strptime(table[0].replace('Date: ', ''), "%d %B %Y - %H:%M")
    sa_price_df = sa_price_df.append({"Date": date,
                                      "Energy": float(table[2].split('       ')[3].replace('$', '')),
                                      "Raise Reg": float(table[3].split('       ')[3].replace('$', '')),
                                      "Lower Reg": float(table[4].split('       ')[3].replace('$', '')),
                                      "Raise 6 sec": float(table[5].split('       ')[3].replace('$', '')),
                                      "Raise 60 sec": float(table[6].split('       ')[3].replace('$', '')),
                                      "Raise 5 min": float(table[7].split('       ')[3].replace('$', '')),
                                      "Lower 6 sec": float(table[8].split('       ')[3].replace('$', '')),
                                      "Lower 60 sec": float(table[9].split('       ')[3].replace('$', '')),
                                      "Lower 5 min": float(table[10].split('       ')[3].replace('$', ''))}, ignore_index=True)
```

```python
sa_price_df
```

```python
plt.figure(figsize=(10, 6))
plt.title("Energy price")
plt.plot(sa_price_df["Date"], sa_price_df["Energy"])
plt.show()
```

```python
cols = ["Raise Reg", "Raise 6 sec", "Raise 60 sec", "Raise 5 min", "Lower Reg", "Lower 6 sec", "Lower 60 sec", "Lower 5 min"]

fig, axes = plt.subplots(2, 4, figsize=(30, 6))
i = 0

for row in axes:
    for ax in row:
        ax.set_title(cols[i] + " price")
        ax.plot(sa_price_df["Date"], sa_price_df[cols[i]])
        i += 1

fig.tight_layout()
```

<!-- #region tags=[] -->
### 5 minute dispatch data CSV
<!-- #endregion -->

First, 5 minute dispatch data obtained from `scraper.ipynb` is required.

```python
csv_filename = str(next(pathlib.Path("/tmp/").glob('*.CSV')))
```

```python
i_indices = []

with open(csv_filename, 'r') as f:
    contents = f.read()
    for i, line in enumerate(contents.split('\n')):
        if len(line) > 0 and line[0] == 'I':
            i_indices.append(i)
```

```python
dfs = []

for i in range(len(i_indices) - 1):
    dfs.append(pd.read_csv(csv_filename, skiprows=i_indices[i], nrows=(i_indices[i+1] - i_indices[i]-1)))
```

A CSV of dispatch data consists of 6 tables. The names of the tables are provided on the tables themselves as the repeating entries of the second and third columns. Below is a list of the tables that they correspond to, as given in AEMO's MMS Data Model Report, with references to the appropriate sections of the Report.

- `dfs[0]` is the DISPATCHCASESOLUTION table (see section 12.14)
- `dfs[1]` is the DISPATCH_LOCAL_PRICE table (see section 12.7)
- `dfs[2]` is the DISPATCHPRICE table (see section 12.19)
- `dfs[3]` is the DISPATCHREGIONSUM table (see seciton 12.20)
- `dfs[4]` is the DISPATCHINTERCONNECTORRES table (see section 12.16)
- `dfs[5]` is the DISPATCHCONSTRAINT table (see section 12.15)

```python
display(dfs[3])
```

We can gather dispatch data from SA. 

```python
dispatch_prices = []

for csv_filename in pathlib.Path("/tmp/").glob('*.CSV'):
    i_indices = []

    with open(csv_filename, 'r') as f:
        contents = f.read()
        for i, line in enumerate(contents.split('\n')):
            if len(line) > 0 and line[0] == 'I':
                i_indices.append(i)
                
    dispatch_prices.append(pd.read_csv(csv_filename, skiprows=i_indices[2], nrows=(i_indices[3] - i_indices[2]-1)))
```

```python
sa_price_df = pd.DataFrame(columns = ["Date", "Energy", "Raise Reg", "Lower Reg", "Raise 6 sec", "Raise 60 sec", "Raise 5 min", "Lower 6 sec", "Lower 60 sec", "Lower 5 min"])
for i, table in enumerate(dispatch_prices):
    date = datetime.datetime.strptime(table[0].replace('Date: ', ''), "%d %B %Y - %H:%M")
    sa_price_df = sa_price_df.append({"Date": date,
                                      "Energy": float(table[2].split('       ')[3].replace('$', '')),
                                      "Raise Reg": float(table[3].split('       ')[3].replace('$', '')),
                                      "Lower Reg": float(table[4].split('       ')[3].replace('$', '')),
                                      "Raise 6 sec": float(table[5].split('       ')[3].replace('$', '')),
                                      "Raise 60 sec": float(table[6].split('       ')[3].replace('$', '')),
                                      "Raise 5 min": float(table[7].split('       ')[3].replace('$', '')),
                                      "Lower 6 sec": float(table[8].split('       ')[3].replace('$', '')),
                                      "Lower 60 sec": float(table[9].split('       ')[3].replace('$', '')),
                                      "Lower 5 min": float(table[10].split('       ')[3].replace('$', ''))}, ignore_index=True)
```

```python
plt.plot(range(len(lower5minrrp)), lower5minrrp)
```

### Daily historical dispatch data CSV

```python
csv_filename = str(pathlib.Path.home() / "Downloads/data/PUBLIC_DAILY_202112140000_20211215040504.CSV")
```

```python
i_indices = []

with open(csv_filename, 'r') as f:
    contents = f.read()
    for i, line in enumerate(contents.split('\n')):
        if len(line) > 0 and line[0] == 'I':
            i_indices.append(i)
```

```python
dfs = []

for i in range(len(i_indices) - 1):
    dfs.append(pd.read_csv(csv_filename, skiprows=i_indices[i], nrows=(i_indices[i+1] - i_indices[i]-1)))
```

```python
# TODO: check whether dfs[1] or dfs[2] contains the relevant prices?

df1 = dfs[1]
df2 = dfs[2]
```

```python
df1["REGIONID"].value_counts()
```

```python
sa_price_df = df1.loc[df1["REGIONID"].str.contains("SA"),
                      ["SETTLEMENTDATE", "RRP", "LOWERREGRRP", "RAISEREGRRP", "LOWER6SECRRP", "RAISE6SECRRP", "LOWER60SECRRP", "RAISE60SECRRP", "LOWER5MINRRP", "RAISE5MINRRP"]]
```

```python
sa_price_df.columns = ["Date", "Energy", "Raise Reg", "Lower Reg", "Raise 6 sec", "Raise 60 sec", "Raise 5 min", "Lower 6 sec", "Lower 60 sec", "Lower 5 min"]
```

```python
sa_price_df["Date"] = pd.to_datetime(sa_price_df["Date"])
```

```python
plt.figure(figsize=(10, 6))
plt.title("Energy price")
plt.plot(sa_price_df["Date"], sa_price_df["Energy"])
plt.show()
```

```python
cols = ["Raise Reg", "Raise 6 sec", "Raise 60 sec", "Raise 5 min", "Lower Reg", "Lower 6 sec", "Lower 60 sec", "Lower 5 min"]

fig, axes = plt.subplots(2, 4, figsize=(30, 6))
i = 0

start_date = sa_price_df.iloc[0, 0]

for row in axes:
    for ax in row:
        ax.set_title(cols[i] + " price")
        ax.plot(sa_price_df["Date"], sa_price_df[cols[i]])
        i += 1

fig.tight_layout()
```
