import numpy as np
import scipy.stats as stats

# Generate sample data from an exponential distribution
np.random.seed(42)  # For reproducibility
sample_data = np.random.exponential(scale=2.0, size=100)  # Exponential with lambda=1/scale

# Estimate lambda from sample data (1/mean)
lambda_est = 1 / np.mean(sample_data)

# Define bin edges for the chi-square test (use equal-width bins)
num_bins = 10
bin_edges = np.linspace(0, np.max(sample_data), num_bins + 1)

# Compute observed frequencies
observed, _ = np.histogram(sample_data, bins=bin_edges)

# Compute expected frequencies using the exponential distribution CDF
total_data = len(sample_data)
expected = [
    total_data * (stats.expon.cdf(bin_edges[i + 1], scale=1/lambda_est) -
                  stats.expon.cdf(bin_edges[i], scale=1/lambda_est))
    for i in range(num_bins)
]

# Compute the chi-square statistic
chi_square_stat = sum((observed - expected) ** 2 / expected)

# Degrees of freedom (bins - 1 - estimated parameters)
dof = num_bins - 1 - 1

# P-value from chi-square distribution
p_value = stats.chi2.sf(chi_square_stat, dof)

# Display results
{
    "Chi-Square Statistic": chi_square_stat,
    "Degrees of Freedom": dof,
    "P-Value": p_value,
    "Reject Null Hypothesis?": p_value < 0.05
}
