import numpy as np
import heapq


def int_to_scenario(n, scenario_length):
    """Convert integer n to a binary string.

    Args:
        n: an integer representing the scenario.
        scenario_length: the length of the scenario.

    Returns:
       the scenario as a Boolean numpy array.
    """
    assert type(n) in [int, np.int64], n

    bitstr = ("{:0" + str(scenario_length) + "b}").format(n)
    bitstr = [int(b) for b in bitstr]
    bitstr = np.array(bitstr, dtype=bool)

    return bitstr


def scenario_to_int(bitstr):
    """Convert binary string to integer.

    Args:
        bitstr: a Boolean numpy array.

    Returns:
        the integer representation of `bitstr`.
    """
    return sum((2**i * bit for i, bit in enumerate(bitstr[::-1])))


def int_scenario_probability(index, success_probs):
    """Output probability of scenario given by index.

    Args:
        index: the index of a scenario
        success_probs: an array of the probabilities of success for the
            Bernoulli process

    Returns:
        the probability of a scenario
    """
    assert type(index) in [int, np.int64]

    scenario_length = len(success_probs)
    scenario = int_to_scenario(index, scenario_length)

    return np.prod(scenario * success_probs + (1 - scenario) * (1 - success_probs))


def flip_scenario(int_scenario, prob, index, success_prob):
    """Flip the `index` bit of scenario `bitstr`, returning a pair
    containing the new scenario and its associated probability.

    Args:
        bitstr: the scenario as a boolean array
        prob: probability of the scenario occurring
        index: the index of the bit to be flipped
        success_prob: the probability of success

    Returns:
        a pair `(prob, scenario)` with `scenario` being the new
        scenario in bitstring form and `prob` being the associated probability.
    """
    bit = (int_scenario & 2**index) == 2**index

    prob /= (bit * success_prob + (1 - bit) * (1 - success_prob))

    bit = not bit

    prob *= (bit * success_prob + (1 - bit) * (1 - success_prob))

    if bit == 0:
        new_scenario = int_scenario - (2**index)
    else:
        new_scenario = int_scenario + (2**index)

    return (prob, new_scenario)


def possible_next_scenarios(bitstr, scenario_prob, success_probs):
    """Return a generator of the scenarios from `int_scenario` that
    may be the next possible scenario.

    Args:
        int_scenario: the scenario in integer representation
        scenario_prob: probability of the scenario
        success_probs: the success probabilities

    Return:
        a generator of the next possible scenarios from int_scenario
    """
    scenario_length = len(success_probs)
    for flip_pos in range(scenario_length):
        new_prob, new_scenario = flip_scenario(bitstr, scenario_prob, flip_pos, success_probs[flip_pos])
        yield (new_prob, new_scenario)


def get_top_scenarios(n, success_probs):
    """Get the `n` most likely outcomes (scenarios) for the Bernoulli process
    specified by the array of probabilities of success `success_probs`.

    Args:
        n: the number of top outcomes to generate
        success_probs: an array of the probabilities of success for the
            Bernoulli process

    Returns:
        a matrix with the i-th row containing the i-th top scenario.
    """
    top_scenario = scenario_to_int([0 if p < 0.5 else 1 for p in success_probs])

    q = [(-int_scenario_probability(top_scenario, success_probs), top_scenario)]

    top_scenarios = []
    encountered = set([e[1] for e in q])

    i = 0
    iter_limit = n

    while i < iter_limit and len(q) > 0:
        i += 1
        top_prob, top_scenario = heapq.heappop(q)
        top_scenarios.append(top_scenario)
        for prob, scenario in possible_next_scenarios(top_scenario, top_prob, success_probs):
            elem = (-prob, scenario)
            if scenario not in encountered:
                heapq.heappush(q, elem)
                encountered.add(scenario)

    scenario_length = len(success_probs)
    return np.array([int_to_scenario(s, scenario_length) for s in top_scenarios])
