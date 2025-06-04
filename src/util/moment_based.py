import torch
import numpy as np

from torch import Tensor
from scipy.stats import moment


def calc_moment_based_stats(sample: np.ndarray) -> dict:
    """
    Calculate a panel of moment-based stats for an SPM sample with shape [N, M].
    - Ref: https://gwyddion.net/documentation/user-guide-en/statistical-analysis.html#stat-quantities 

    "Moment based quantities are expressed using integrals of the 
    height distribution function with some powers of height. 
    They include the familiar quantities"
    
    Returns
    ---
        1. Average value
        2. RMS roughnes (sq)
        3. RMS Mean roughness (Sa)
        4. Skew (Ssk)
        5. Excess kurtosis
    """

    if isinstance(sample, Tensor):
        sample = sample.detach().cpu().numpy()

    sample = sample.flatten()

    # calculate central moments
    mu_2 = moment(sample, moment=2)
    mu_3 = moment(sample, moment=3)
    mu_4 = moment(sample, moment=4)

    average_value = sample.mean()

    # σ = μ_2 ^ 1/2
    rms = mu_2 ** .5
    mean_rms = rms.mean()
    mean_roughness = abs(sample - sample.mean()).mean()

    # γ_1 = (μ_3) / (μ_2 ^ (3/2))
    skewness = mu_3 / (mu_2 ** (3/2))

    # γ_2 = (μ_4) / (μ_2 ^ 2) - 3
    kurtosis = (mu_4 / (mu_2 ** 2)) - 3

    return {
        "mean"          : average_value,
        "mean_rms"      : mean_rms,
        "mean_roughness": mean_roughness,
        "skewness"      : skewness,
        "kurtosis"      : kurtosis,
    }


if __name__ == "__main__":
    fp     = "/playpen/mufan/levi/tianlong-chen-lab/sparse-cafm/data/raw-data/3-12-25/A1 512_ Height_Backward_020.npy"
    sample = np.load(fp)
    
    from pprint import pprint
    pprint(calc_moment_based_stats(sample), indent=4)