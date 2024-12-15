import math
from math import pi, log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import hbar, epsilon_0, c, k
from scipy.optimize import curve_fit

TRANSITIONS = [
    {"frequency": 335.116048807e12 - 656.820e6 - 4.021776399375e9,
     "c_f_squared": 7 / 12},
    {"frequency": 335.116048807e12 - 510.860e6 - 4.021776399375e9,
     "c_f_squared": 5 / 12},
    {"frequency": 335.116048807e12 - 656.820e6 + 5.170855370625e9,
     "c_f_squared": 7 / 36},
    {"frequency": 335.116048807e12 + 510.860e6 + 5.170855370625e9,
     "c_f_squared": 7 / 12},
]

CELL_LENGTH = 0.012  # 12 mm
# This coefficient has been calculated by averaging the two coefficients obtained for the
# two couples of neighbouring peaks
T_TO_F_COEFFICIENT = - 1.661612554743162e12
GAIN = 475000  # Taken from datasheet
RESPONSIVITY = 0.6
AGGREGATION_WINDOW = 1
MHz = 1e6
mm = 1e-3
GAMMA = 2 * pi * 4.56 * MHz
CS_NUCLEAR_SPIN = 7 / 2  # Cs nuclear spin
TEMPERATURE = 295  # 22 °C


# S_ge = line strength
def calculate_line_strength(transition_index: int, transition_frequency: float):
    c_f_squared = TRANSITIONS[transition_index]["c_f_squared"]
    return c_f_squared * 9 * pi * epsilon_0 * hbar * (c ** 3) * GAMMA / (transition_frequency ** 3)


def calculate_density_from_temperature(temperature):
    if temperature < 302:
        p = math.exp(12.6709 - log10(temperature) - 4150 / temperature)
    else:
        p = math.exp(13.178 - 1.35 * log10(temperature) - 4041 / temperature)

    return p / (k * temperature)


def calculate_density_from_gaussians(transition_index: int,
                                     transition_frequency: float,
                                     gaussian_amplitude: float,
                                     sigma_d: float):
    s_ge = calculate_line_strength(transition_index, transition_frequency)
    numerator = gaussian_amplitude * epsilon_0 * hbar * c * 2 * (2 * CS_NUCLEAR_SPIN + 1) * sigma_d
    denominator = transition_frequency * s_ge * math.sqrt(pi / 2)

    return numerator / denominator


def get_fitting_params(x_to_fit, y_to_fit, degree):
    # We only take the ranges outside the absorption peaks
    # between peaks range = 300000:450000
    # end range = 0:200000
    # initial range = 550000:600000
    x_fit = np.concatenate((
        x_to_fit.iloc[int(550000 / AGGREGATION_WINDOW):int(6000000 / AGGREGATION_WINDOW)],
        x_to_fit.iloc[int(300000 / AGGREGATION_WINDOW):int(450000 / AGGREGATION_WINDOW)],
        x_to_fit.iloc[:int(200000 / AGGREGATION_WINDOW)]))

    y_fit = np.concatenate((
        y_to_fit.iloc[int(550000 / AGGREGATION_WINDOW):int(600000 / AGGREGATION_WINDOW)],
        y_to_fit.iloc[int(300000 / AGGREGATION_WINDOW):int(450000 / AGGREGATION_WINDOW)],
        y_to_fit.iloc[:int(200000 / AGGREGATION_WINDOW)]))

    return np.polyfit(x_fit, y_fit, degree)


def gaussian_func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        # params[i] = w_ge
        center = params[i]
        # params[i + 1] = all the constants that multiply the exponential except for sigma_d
        amplitude_constants = params[i + 1]
        # params[i + 2] = sigma_d except for w_ge, therefore sqrt(K_b*T/(M*c^2))
        sigma_d = params[i + 2] * center
        y += (amplitude_constants / sigma_d) * np.exp(-(x - center) ** 2 / (2 * sigma_d ** 2))
    return y


# Specify the file name
filename = 'AAAA.csv'

# Read the data, skipping the first 11 rows and last 100000
df = pd.read_csv(filename, skiprows=11, skipfooter=100000, engine="python")
aggregation_index = []
for i in range(len(df.index)):
    aggregation_index.append(int(i / AGGREGATION_WINDOW))
df["AggregationIndex"] = aggregation_index

data = df.groupby("AggregationIndex", as_index=False).mean()

n_points = len(data.index)
print(n_points)

data["Frequency"] = data["Second"] * T_TO_F_COEFFICIENT * 1e-9
data["Frequency"] = data["Frequency"] - data["Frequency"].iloc[n_points - 1]

data["OutputPower"] = 1000000 * data["Volt.2"] / (RESPONSIVITY * GAIN)

data["Frequency"] = data["Frequency"] - data["Frequency"].iloc[n_points - 1]

a, b, c = get_fitting_params(data["Frequency"], data["OutputPower"], 2)

print(f"Fitting parameters: {a}, {b}, {c}")

data["Fit"] = a * data["Frequency"] ** 2 + b * data["Frequency"] + c

data.plot(x="Frequency", y=["OutputPower", "Fit"])  # Use 'Time' column as x-axis
plt.title(f"Output Power vs Frequency (aggregation window = {AGGREGATION_WINDOW})")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured Power [uW]")
plt.grid(True)
plt.show()
plt.figure(1)

guess_params = [
    2.5, 1, 0.01,
    3.5, 0.5, 0.01,
    11, 0.30, 0.01,
    12.5, 1, 0.01,
]

data["TransmissionCoefficient"] = data["OutputPower"] / data["Fit"]
data["AbsorptionCoefficient"] = np.log(data["TransmissionCoefficient"]) / CELL_LENGTH

optimal_params, covariance_matrix = curve_fit(
    f=gaussian_func,
    xdata=data["Frequency"].to_numpy(),
    ydata=data["AbsorptionCoefficient"].to_numpy(),
    p0=guess_params)

print(f"Optimal parameters: {optimal_params}")

data["GaussianFit"] = gaussian_func(data["Frequency"], *optimal_params)

data.plot(x="Frequency", y=["AbsorptionCoefficient", "GaussianFit"])  # Use 'Time' column as x-axis
plt.title(f"Absorption Coefficient vs Frequency (aggregation window = {AGGREGATION_WINDOW})")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Absorption Coefficient")
plt.grid(True)
plt.show()

transitions_parameters = []
for i in range(0, len(optimal_params), 3):
    transitions_parameters.append({
        "transition_frequency": TRANSITIONS[int(i / 3)]["frequency"],
        "sigma_d": optimal_params[i + 2] * optimal_params[i],
        "absorption_amplitude": optimal_params[i + 1] / (optimal_params[i + 2] * optimal_params[i])
    })

n_from_temp = calculate_density_from_temperature(TEMPERATURE)
print(f"Atomic density calculated from the cell temperature: {n_from_temp}")
n_from_fit = []
for i in range(len(transitions_parameters)):
    n_from_fit.append(calculate_density_from_gaussians(
        i,
        transitions_parameters[i]['transition_frequency'],
        - transitions_parameters[i]['absorption_amplitude'],
        transitions_parameters[i]['sigma_d']
    ))
    print(f"Atomic density calculated from the fitting for {i}-th transition: {n_from_fit[i]}")


# TODO: NELLA TERZA TRANSIZIONE LA SIGMA_D VIENE NEGATIVA ANZICHè LA AMPIEZZA PROCAJFPOAJADPAJTJAD