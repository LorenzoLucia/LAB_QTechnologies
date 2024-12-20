import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utilities import plotting


def fit_gaussian(x, y, initial_guesses=[1, 0, 1]):
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Initial guesses for A, mu, and sigma for each Gaussian
    initial_guesses = [1000, 3400, 8]

    popt, _ = curve_fit(gaussian, x, y, p0=initial_guesses, maxfev=500000)
    fitted_curve = gaussian(x, *popt)

    N = len(y)
    error = 0
    perc_error = 0

    for i in range(N):
        error += (y[i] - fitted_curve[i]) ** 2
        perc_error += np.abs(y[i] - fitted_curve[i]) / y[i]

    # Compute RMSE
    rmse = np.sqrt(error / N)

    # Compute percentage error
    perc_error = (perc_error / N) * 100

    return popt, fitted_curve, rmse, perc_error


def error_evaluation(original_curve, fitted_curve):
    error = 0
    perc_error = 0
    N = len(original_curve)

    for i in range(N):
        error += (original_curve[i] - fitted_curve[i]) ** 2

        perc_error += np.abs(original_curve[i] - fitted_curve[i]) / original_curve[i]

    error = np.sqrt(error / N)
    perc_error = (perc_error / N) * 100

    return error, perc_error


def lorentzian_fitting(x, y):
    def lorentzian(x, A, x0, gamma, C):
        return A / (1 + ((x - x0) / gamma) ** 2) + C

    params, covariance = curve_fit(lorentzian, x, y, p0=[max(y), np.argmax(y), 1, min(y)], maxfev=5000)
    fitted_curve = lorentzian(x, *params)

    return fitted_curve, params, covariance


# Read file1.txt and file2.txt into NumPy arrays
coincidenze_ON = np.loadtxt('coincidenzeG8.txt')  # Automatically handles numeric conversion
coincidenze_OFF = np.loadtxt('coincidenzeG8_OFF.txt')

print("Coincidenze ON:", coincidenze_ON, "with len:", len(coincidenze_ON))
print("Coincidenze OFF:", coincidenze_OFF, "with len:", len(coincidenze_OFF))

times = np.linspace(start=0, stop=len(coincidenze_OFF) - 1, num=len(coincidenze_OFF), endpoint=True, retstep=False,
                    dtype=None)

verbose = False

if verbose:
    plotting(x=times, y=coincidenze_ON, name='coincidences ON', x_axis='times', y_axis='coincidences', marker='none')
    plotting(x=times, y=coincidenze_OFF, name='coincidences OFF', x_axis='times', y_axis='coincidences', marker='none')

differences = []

for i in range(len(coincidenze_OFF)):
    differences.append(coincidenze_ON[i] - coincidenze_OFF[i])

if verbose:
    plotting(x=times, y=differences, name='difference', x_axis='times', y_axis='coincidences', marker='none')

differences_cutted = []
times_cutted = []

start = 3388
stop = 3420

i = 0
while i < len(coincidenze_OFF):
    while i > start and i < stop:
        differences_cutted.append(differences[i])
        times_cutted.append(times[i])
        i += 1
    i += 1

if verbose:
    plotting(x=times_cutted, y=differences_cutted, name='differences_cutted', x_axis='times', y_axis='coincidences',
             marker='none')

back_flash = []
times_back_flash = []

start_back_flash = 3401
end_back_flash = 3420

i = start_back_flash

while i < end_back_flash:
    back_flash.append(differences[i])
    times_back_flash.append(times[i])
    i += 1

if verbose:
    plotting(x=times_back_flash, y=back_flash, name='back flash', x_axis='times', y_axis='coincidences', marker='none')

points_back_flash, fitted_back_flash, rmse, perc_error = fit_gaussian(x=times_back_flash, y=back_flash)
verbose = True

if verbose:
    plt.plot(times_back_flash, back_flash, label='actual curve', color='blue')
    plt.plot(times_back_flash, fitted_back_flash, label='fitted curve', color='red')
    plt.xlabel('times')
    plt.ylabel('coincidences')
    plt.title('fitted curve')
    plt.legend()
    plt.grid(True)
    plt.show()

print("The rmse is:", rmse, "with percentage of success:", 100 - perc_error)
