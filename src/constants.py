"""Common constants used in the optimisation models and helper functions."""

services = ["RAISE5MIN", "LOWER5MIN", "RAISE60SEC", "LOWER60SEC", "RAISE6SEC", "LOWER6SEC"]

F_lower = ["lower_6_sec", "lower_60_sec", "lower_5_min"]
F_raise = ["raise_6_sec", "raise_60_sec", "raise_5_min"]
F = F_lower + F_raise
