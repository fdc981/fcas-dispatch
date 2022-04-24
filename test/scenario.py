import unittest
from src.scenario import get_top_scenarios
from src.constants import services
import pandas as pd
import numpy as np


class TestTopScenarioGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        price_df = pd.read_csv("data/sa_fcas_data.csv", parse_dates=["SETTLEMENTDATE"])

        for s in services:
            price_df[s + "PROB"] = (price_df[s + "LOCALDISPATCH"] / price_df[s + "ACTUALAVAILABILITY"]).clip(upper=1)

        price_df["PERIODID"] = price_df["SETTLEMENTDATE"].apply(lambda x : str(x.hour).zfill(2) + str(x.minute).zfill(2))

        prob_df = price_df.groupby("PERIODID")[[s + "PROB" for s in services]].mean().reset_index()

        self.scenario_length = 14
        self.num_scenarios = 2**self.scenario_length
        self.prob_vec = prob_df[services[0] + 'PROB'][:self.scenario_length].values
        self.top_scenarios = get_top_scenarios(self.num_scenarios, self.prob_vec)

    def test_is_monotonic_decreasing(self):
        outcome_probs = self.top_scenarios * self.prob_vec + (1 - self.top_scenarios) * (1 - self.prob_vec)
        scenario_probs = np.prod(outcome_probs, axis=1, dtype=np.float128)

        self.assertTrue(np.all(np.diff(scenario_probs) <= 0))

    def test_is_correct_shape(self):
        self.assertTrue(self.top_scenarios.shape == (self.num_scenarios, self.scenario_length))


if __name__ == "__main__":
    unittest.main()
