"""Common constants used in the optimisation models and helper functions."""

services = ["RAISE5MIN", "LOWER5MIN", "RAISE60SEC", "LOWER60SEC", "RAISE6SEC", "LOWER6SEC"]

F_lower = ["lower_6_sec", "lower_60_sec", "lower_5_min"]
F_raise = ["raise_6_sec", "raise_60_sec", "raise_5_min"]
F = F_lower + F_raise

quantile_col_names = ["Q_QRA01", "Q_QRA1", "Q_QRA2", "Q_QRA3", "Q_QRA4", "Q_QRA5", "Q_QRA6", "Q_QRA7", "Q_QRA8", "Q_QRA9", "Q_QRA99"]

quantile_cutoffs = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
quantile_cutoffs = [0] + quantile_cutoffs + [1]
