import numpy as np
from scipy.optimize import curve_fit


def polynomial_fitting(x, y, degree=2):
    """
    Fit the data (x, y) to a polynomial of a given degree.

    Args:
        x (array): The x data.
        y (array): The y data.
        degree (int): The degree of the polynomial (default is 3).

    Returns:
        coefficients: Fitted polynomial coefficients.
        fitted_curve: The y-values of the fitted curve.
    """
    coefficients = np.polyfit(x, y, degree)
    poly_model = np.poly1d(coefficients)
    fitted_curve = poly_model(x)

    return coefficients, fitted_curve


def gaussian_fitting(x, y):
    """
    Fit the data (x, y) to a Gaussian function.

    Args:
        x (array): The x data.
        y (array): The y data.

    Returns:
        popt: Fitted parameters (A, mu, sigma).
        fitted_curve: The y-values of the fitted Gaussian curve.
    """

    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    popt, _ = curve_fit(gaussian, x, y, p0=[1, 0, 1])  # Initial guesses
    fitted_curve = gaussian(x, *popt)

    return popt, fitted_curve


def fit_four_gaussians(x, y):
    """
    Fit the data (x, y) to a sum of four Gaussian functions.

    Args:
        x (array): The x data.
        y (array): The y data.

    Returns:
        popt: Fitted parameters [A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4].
        fitted_curve: The y-values of the fitted Gaussian curve.
    """

    def combined_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4):
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        return (gaussian(x, A1, mu1, sigma1) +
                gaussian(x, A2, mu2, sigma2) +
                gaussian(x, A3, mu3, sigma3) +
                gaussian(x, A4, mu4, sigma4))

    # Initial guesses for A, mu, and sigma for each Gaussian
    initial_guess = [-3, 510.860, 1,  # Gaussian 1
                     -3, 656.820, 1,  # Gaussian 2
                     -3, 4021.000, 1,  # Gaussian 3
                     -3, 5170.000, 1]  # Gaussian 4

    # Fit the combined Gaussian function
    popt, _ = curve_fit(combined_gaussian, x, y, p0=initial_guess, maxfev=100000)
    fitted_curve = combined_gaussian(x, *popt)

    return popt, fitted_curve


def resize_vector(vector, index1, index2):
    new_vector = []

    for i in range(len(vector)):
        if i >= index1 and i <= index2:
            new_vector.append(vector[i])
    return new_vector


def calculate_means(vector, chunk_size):
    means = []
    for i in range(0, len(vector), chunk_size):
        chunk = vector[i:i + chunk_size]
        means.append(np.mean(chunk))
    return means


def minimals(y):
    local_minima = []
    local_minima_index = []
    for i in range(1, len(y) - 1):  # Escludi il primo e l'ultimo punto
        if y[i] < y[i - 1] and y[i] < y[i + 1]:  # Se il punto è più basso dei vicini
            local_minima.append(y[i])
            local_minima_index.append(i)
    return local_minima, local_minima_index


def gaussian(x, mu, sigma, A):
    x = np.array(x)  # Ensure x is a numpy array
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sum_of_gaussians(mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4, x_axis):
    # Create an array of `num_points` evenly spaced values from -10 to 10 (you can change the range as needed)

    # Calculate the sum of the 4 Gaussians
    gaussian_sum = (
            gaussian(x_axis, mu1, sigma1, A1) +
            gaussian(x_axis, mu2, sigma2, A2) +
            gaussian(x_axis, mu3, sigma3, A3) +
            gaussian(x_axis, mu4, sigma4, A4)
    )

    return x_axis, gaussian_sum
