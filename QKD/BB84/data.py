import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def gaussian(x, *params):
    mu, sigma, amplitude = params
    return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


coincidences_file = "plot_coincidences_single_photons.txt"
bin_width = 100e-12
skip_rows = 800
fiber_ref_index = 1.5
c = 3e8

df = pd.read_csv(coincidences_file, names=("Time", "Amplitude"), skiprows=skip_rows, delimiter="\t")

optimal_params, covariance_matrix = curve_fit(
    f=gaussian,
    xdata=df["Time"].to_numpy(),
    ydata=df["Amplitude"].to_numpy(),
    p0=[900, 1, 30000])

df["AmplitudeFit"] = gaussian(df["Time"].to_numpy(), *optimal_params)

fit_center = optimal_params[0]

# This is not a precise calculation therefore I truncate at the second decimal unit
delta_length = round(c * fit_center * bin_width / fiber_ref_index, 2)

print(f"Difference in path length: {delta_length} m")

df.plot(x="Time", y=["Amplitude", "AmplitudeFit"])
plt.title("Coincidences single photons")
plt.xlabel("Time [ps]")
plt.ylabel("Coincidences")
plt.show()
