import numpy as np
import scipy.stats as stats


def ind_diff_prop(
    p1: float = 0.117,
    p2: float = 0.117,
    n1: int = 1350,
    n2: int = 1350,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Calculate the confidence interval for the difference in proportions between two indipendent groups.

    Args:
        p1 (float): Positive proportion in group 1
        p2 (float): Positive proportion in group 2
        n1 (int): Sample count in group 1
        n2 (int): Sample count in group 2
        confidence_level (float): Confidence level

    Returns:
        upper_bound (float): Upper bound of the confidence interval.
        lower_bound (float): Lower bound of the confidence interval.
    """
    # Standard error formula for the difference in proportions
    SE = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))

    # Critical value (2-sided) from standard normal distribution
    Z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin_of_error = Z * SE

    # Confidence interval upper/lower bounds
    lower_bound = (p1 - p2) - margin_of_error
    upper_bound = (p1 - p2) + margin_of_error

    return upper_bound, lower_bound


def wald(A: int, B: int, C: int, D: int, N: int, Z: float = 1.96) -> tuple[float, float, float]:

    """
    Wald confidence interval for dependent proportions.

    https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635

    Args:
        A (int): Number of successes in condition 1 and 2.
        B (int): Number of successes in condition 1 and failures in condition 2.
        C (int): Number of failures in condition 1 and successes in condition 2.
        D (int): Number of failures in condition 1 and 2.
        N (int): Total number of observations.

    Returns:
        diff (float): Mean difference in proportion.
        lower_conf (float): Lower 95% confidence interval.
        upper_conf (float): Upper 95% confidence interval.
    """

    p_12 = B / N
    p_21 = C / N
    diff = p_12 - p_21

    # Variance estimation
    var = ((p_12 + p_21) - (p_12 - p_21) ** 2) / N

    lower_conf = diff - Z * (var**0.5)
    upper_conf = diff + Z * (var**0.5)

    return diff, lower_conf, upper_conf



def wald_adjusted(A: int, B: int, C: int, D: int, N: int, Z: float = 1.96) -> tuple[float, float, float]:
    """
    Adjusted Wald paired binary confidence interval function.

    https://www.researchgate.net/publication/258150348_Adjusted_Wald_Confidence_Interval_for_a_Difference_of_Binomial_Proportions_Based_on_Paired_Data

    Args:
        A (int): Number of successes in condition 1 and 2.
        B (int): Number of successes in condition 1 and failures in condition 2.
        C (int): Number of failures in condition 1 and successes in condition 2.
        D (int): Number of failures in condition 1 and 2.
        N (int): Total number of observations.

    Returns:
        diff (float): Mean difference in proportion.
        lower_conf (float): Lower 95% confidence interval.
        upper_conf (float): Upper 95% confidence interval.
    """
    # Adjusted proportions
    phi_12 = (B + 1) / (N + 2)
    phi_21 = (C + 1) / (N + 2)
    diff = phi_12 - phi_21

    # Standard error formula
    s_squared = ((phi_12 + phi_21) - (phi_12 - phi_21) ** 2) / (N + 2)
    lower_conf = diff - Z * (s_squared**0.5)
    upper_conf = diff + Z * (s_squared**0.5)

    return diff, lower_conf, upper_conf

