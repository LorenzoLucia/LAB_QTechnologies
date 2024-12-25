import math
from decimal import Decimal
from math import pi, log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import hbar, epsilon_0, k, c
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

# print(scipy.constants.physical_constants["hbar"])
CELL_LENGTH = 0.012  # 12 mm

T_TO_F_COEFFICIENT = - 1.661612554743162e12
print(f"T to F coefficient: {Decimal(T_TO_F_COEFFICIENT):.6E} Hz/s")
GAIN = 475000  # Taken from datasheet
RESPONSIVITY = 0.6
AGGREGATION_WINDOW = 1
GAMMA = 2 * pi * 4.56e6
CS_NUCLEAR_SPIN = 7 / 2  # Cs nuclear spin
TEMPERATURE = 295  # 22 °C


# S_ge = line strength
def calculate_line_strength(transition_index: int, transition_frequency: float):
    c_f_squared = TRANSITIONS[transition_index]["c_f_squared"]
    return c_f_squared * 9 * pi * epsilon_0 * hbar * (c ** 3) * GAMMA / (transition_frequency ** 3)


def calculate_density_from_temperature(temperature):
    if temperature < 302:
        p = 10 ** (12.6709 - log10(temperature) - 4150 / temperature)
    else:
        p = 10 ** (13.178 - 1.35 * log10(temperature) - 4041 / temperature)

    print(f"P: {p}")
    print(f"Using: k {k}, temperature {temperature}")

    return p / (k * temperature)


def calculate_density_from_gaussians(transition_index: int,
                                     transition_frequency: float,
                                     gaussian_amplitude: float,
                                     sigma_d: float):
    s_ge = calculate_line_strength(transition_index, 2 * pi * transition_frequency)
    numerator = gaussian_amplitude * epsilon_0 * hbar * c * 2 * (2 * CS_NUCLEAR_SPIN + 1) * sigma_d
    denominator = 2 * pi * transition_frequency * s_ge * math.sqrt(pi / 2)
    print(
        f"Using: epsilon_0 {epsilon_0}, c {c}, hbar {hbar}, transition fre {transition_frequency}, nuclear spin {CS_NUCLEAR_SPIN}, s_ge {s_ge}, amplitude {gaussian_amplitude}")

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
        amplitude_constant = - params[i + 1]
        # params[i + 2] = sigma_d except for w_ge, therefore sqrt(K_b*T/(M*c^2))
        sigma_d = params[i + 2]
        y += amplitude_constant * np.exp(- (2 * pi * x - 2 * pi * center) ** 2 / (2 * sigma_d ** 2))
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
print(f"Total number of points: {n_points}")

data["Frequency"] = data["Second"] * T_TO_F_COEFFICIENT
data["Frequency"] = data["Frequency"] - data["Frequency"].iloc[n_points - 1]

data["OutputPower"] = 1000000 * data["Volt.2"] / (RESPONSIVITY * GAIN)

data["Frequency"] = data["Frequency"] - data["Frequency"].iloc[n_points - 1]

a_2, a_1, a_0 = get_fitting_params(data["Frequency"], data["OutputPower"], 2)

data["Fit"] = a_2 * data["Frequency"] ** 2 + a_1 * data["Frequency"] + a_0

data.plot(x="Frequency", y=["OutputPower", "Fit"])  # Use 'Time' column as x-axis
plt.title(f"Output Power vs Frequency (aggregation window = {AGGREGATION_WINDOW})")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Measured Power [uW]")
plt.grid(True)
plt.show()
plt.figure(1)

guess_params = [
    0.25e10, 12, 0.02e10,  # First peak
    0.35e10, 8, 0.02e10,  # Second peak
    1.15e10, 3, 0.02e10,  # Third peak
    1.3e10, 11, 0.02e10,  # Fourth peak
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
# data.plot(x="Frequency", y=["AbsorptionCoefficient"])  # Use 'Time' column as x-axis
plt.title(f"Absorption Coefficient vs Frequency (aggregation window = {AGGREGATION_WINDOW})")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Absorption Coefficient")
plt.grid(True)
plt.show()

transitions_parameters = []
for i in range(0, len(optimal_params), 3):
    transitions_parameters.append({
        "theoretical_transition_frequency": TRANSITIONS[int(i / 3)]["frequency"],
        "center_frequency": optimal_params[i],
        "sigma_d": optimal_params[i + 2],
        "absorption_amplitude": optimal_params[i + 1]
    })

n_from_temp = calculate_density_from_temperature(TEMPERATURE)
print(f"Atomic density calculated from the cell temperature: {Decimal(n_from_temp):.4E}")
n_from_fit = []
for i in range(len(transitions_parameters)):
    n_from_fit.append(calculate_density_from_gaussians(
        i,
        abs(transitions_parameters[i]['theoretical_transition_frequency']),
        abs(transitions_parameters[i]['absorption_amplitude']),
        abs(transitions_parameters[i]['sigma_d'])
    ))
    print(f"Atomic density calculated from the fitting for {i}-th transition: {Decimal(n_from_fit[i]):.4E}")

print("------------------------------------------------------------")
print(f"Using delta t 'delta_t_1'")
val = abs(transitions_parameters[3]['center_frequency']) - abs(transitions_parameters[2]['center_frequency'])
print(
    f"Small difference 1: EXP {Decimal(val):.4E} - TRUE {Decimal(1167.680e6):.4E} - DIFF {Decimal(1167.680e6 - val):.4E}")

val = abs(transitions_parameters[1]['center_frequency']) - abs(transitions_parameters[0]['center_frequency'])
print(
    f"Small difference 2: EXP {Decimal(val):.4E} - TRUE {Decimal(1167.680e6):.4E} - DIFF {Decimal(1167.680e6 - val):.4E}")

val = abs(transitions_parameters[2]['center_frequency']) - abs(transitions_parameters[1]['center_frequency'])
print(
    f"Intermediate difference: EXP {Decimal(val):.4E} - TRUE {Decimal(9.192631770e9):.4E} - DIFF {Decimal(9.192631770e9 - val):.4E}")
