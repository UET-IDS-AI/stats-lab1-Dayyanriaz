import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title("Normal(0,1) Distribution Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10, color='lightgreen', edgecolor='black')
    plt.title("Uniform(0,10) Distribution Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10, color='salmon', edgecolor='black')
    plt.title("Bernoulli(0.5) Distribution Histogram")
    plt.xlabel("Value (0 or 1)")
    plt.ylabel("Frequency")
    plt.show()
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    return np.mean(data)


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    n = len(data)
    mean = np.mean(data)
    return np.sum((data - mean) ** 2) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    data_sorted = np.sort(data)
    minimum = np.min(data_sorted)
    maximum = np.max(data_sorted)
    median = np.median(data_sorted)
    q1 = np.percentile(data_sorted, 25, interpolation='midpoint')
    q3 = np.percentile(data_sorted, 75, interpolation='midpoint')
    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    return cov


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    matrix = np.array([[var_x, cov_xy],
                       [cov_xy, var_y]])
    return matrix
