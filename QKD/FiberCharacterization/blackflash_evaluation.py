import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

detector_efficiency = 0.20
channel_efficiency = 0.71


def reduce_rows(df, initial_index, final_index):
    df["coincidences_ON"] = df["coincidences_ON"][initial_index:final_index]
    df["coincidences_OFF"] = df["coincidences_OFF"][initial_index:final_index]
    df["Difference"] = df["Difference"][initial_index:final_index]
    df["Time"] = df["Time"][initial_index:final_index]


def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian_integral(amplitude, sigma):
    return amplitude * sigma * math.sqrt(2 * math.pi)


data_counts_on = pd.read_csv("ConteggiG8.txt", names=["Col1", "Col2"], delimiter="\t")
data_counts_off = pd.read_csv("ConteggiG8_OFF.txt", names=["Col1", "Col2"], delimiter="\t")
data_counts_on["Time"] = data_counts_on.index
data_counts_off["Time"] = data_counts_off.index
data_coinc = pd.DataFrame()

# Read file1.txt and file2.txt into NumPy arrays
data_coinc["coincidences_ON"] = np.loadtxt('coincidenzeG8.txt')
data_coinc["coincidences_OFF"] = np.loadtxt('coincidenzeG8_OFF.txt')
data_coinc["Difference"] = data_coinc["coincidences_ON"] - data_coinc["coincidences_OFF"]
data_coinc["Time"] = data_coinc.index

reduce_rows(data_coinc, 3350, 3450)

data_coinc.loc[3396:3400, "Difference"] = 0

guess_params = [
    3405,
    5,
    1600
]
optimal_params, covariance_matrix = curve_fit(gaussian,
                                              data_coinc["Time"].dropna().to_numpy(),
                                              data_coinc["Difference"].dropna().to_numpy(),
                                              p0=guess_params)

data_coinc["DifferenceFit"] = gaussian(data_coinc["Time"].to_numpy(),
                                       optimal_params[0],
                                       optimal_params[1],
                                       optimal_params[2])

n_photons = 0
for i in data_counts_on["Col1"]:
    n_photons += i

n_backflash_photons = gaussian_integral(optimal_params[2], optimal_params[1])

print(f"Total number of photons: {n_photons}")
print(f"Number of backflash photons: {math.ceil(n_backflash_photons)}")
print(
    f"Information gained by Eve: {100 * n_backflash_photons / (n_photons * detector_efficiency * channel_efficiency)} %")

data_coinc.plot(x="Time", y=["coincidences_ON", "coincidences_OFF"])
plt.title("Coincidences vs Time")
plt.xlabel("Time [bin=500 ps]")
plt.ylabel("Coincidences")
plt.grid(True)
plt.show()
plt.figure(1)

data_coinc.plot(x="Time", y=["Difference", "DifferenceFit"])
plt.title("Coincidences difference vs Time")
plt.xlabel("Time [bin=500 ps]")
plt.ylabel("Coincidences")
plt.grid(True)
plt.show()
plt.figure(1)

data_counts_on.plot(x="Time", y=["Col1", "Col2"])
plt.title("Photons per second (detector ON)")
plt.xlabel("Time")
plt.ylabel("Coincidences")
plt.grid(True)
plt.show()
plt.figure(1)

data_counts_off.plot(x="Time", y=["Col1", "Col2"])
plt.title("Photons per second (detector OFF)")
plt.xlabel("Time")
plt.ylabel("Coincidences")
plt.grid(True)
plt.show()
plt.figure(1)
