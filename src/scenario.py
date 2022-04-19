import numpy as np
import heapq


def int_to_scenario(n, scenario_length):
    """Convert integer n to a binary string."""
    assert type(n) in [int, np.int64], n

    bitstr = ("{:0" + str(scenario_length) + "b}").format(n)
    bitstr = [int(b) for b in bitstr]
    bitstr = np.array(bitstr, dtype=bool)

    return bitstr


def scenario_to_int(bitstr):
    """Convert binary string to integer."""
    return sum((2**i * bit for i, bit in enumerate(bitstr[::-1])))


def int_scenario_probability(index, success_probs):
    """Output probability of scenario given by index"""
    assert type(index) in [int, np.int64]

    scenario_length = len(success_probs)
    scenario = int_to_scenario(index, scenario_length)

    return np.prod(scenario * success_probs + (1 - scenario) * (1 - success_probs))


def possible_next_scenarios(int_scenario, success_probs):
    scenario_length = len(success_probs)
    for flip_pos in range(scenario_length):
        scenario = int_scenario ^ 2**flip_pos
        if int_scenario_probability(scenario, success_probs) <= int_scenario:
            yield scenario


def get_top_scenarios(n, success_probs):
    most_likely_scenario = scenario_to_int([0 if p < 0.5 else 1 for p in success_probs])

    q = [(-int_scenario_probability(most_likely_scenario, success_probs), most_likely_scenario)]

    top_scenarios = []
    encountered = set([e[1] for e in q])

    i = 0
    iter_limit = n

    while i < iter_limit and len(q) > 0:
        i += 1
        most_likely_scenario = heapq.heappop(q)
        top_scenarios.append(most_likely_scenario)
        for s in possible_next_scenarios(most_likely_scenario[1], success_probs):
            elem = (-int_scenario_probability(s, success_probs), s)
            if s not in encountered:
                heapq.heappush(q, elem)
                encountered.add(s)

    scenario_length = len(success_probs)
    return np.array([int_to_scenario(s, scenario_length) for _, s in top_scenarios])
