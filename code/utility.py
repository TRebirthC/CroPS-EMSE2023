"""
This file contains some utility functions/variables for this project.
"""
import numpy as np
import numbers


"""constant variables"""
ONE_DAY_HAS_TIMESTAMPS = 86400  # timestamps for one day
PSEUDO_LABEL_UNLABEL = -1  # pseudo label for unlabelled data
MY_EPS = np.finfo("float").eps


def check_random_state(seed):
    """
    This method is used to turn seed into a np.random.RandomState instance.
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise, raise ValueError.

    Args:
        seed (none, int or instance of RandomState): The random seed.
    """
    # examples of usage:
    # rng = check_random_state(rnd_seed)
    # rng.choice(repo_id_use, nb_repo_request, replace=False)
    # or
    # rng.uniform(0, 1, 100)

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)


def cvt_day2timestamp(days: float) -> float:
    """
    This method is used to convert the days to the value of timestamps.

    Args:
        days (float, list or tuple): The value of the days.

    Returns:
        day2timestamp_list (the same as days): The converted timestamps.
    """
    days_np = np.array(days)  # force to numpy data
    day2timestamp_list = (days_np * ONE_DAY_HAS_TIMESTAMPS).tolist()
    return day2timestamp_list


def cvt_timestamp2day(timestamps):
    """
    This method is used to convert the value of timestampsthe to days.

    Args:
        timestamps (float, list or tuple): The value of timestamps.

    Returns:
        days_list (the same as timestamps): The converted days.
    """
    timestamps_np = np.array(timestamps)
    days_list = (timestamps_np / ONE_DAY_HAS_TIMESTAMPS).tolist()
    return days_list


if __name__ == '__main__':
    times = times = [86401, 186400, 286400]
    days = cvt_timestamp2day(times)
    print(days)

