import math
from decimal import Decimal
from math import pi, log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import hbar, epsilon_0, k, c
from scipy.optimize import curve_fit

from AS.unc_estimator import calc_gaussians_density_unc, calc_temperature_density_unc, calc_temperature_unc

TRANSITIONS = [
    {"frequency": 335.116048807e12 - 656.820e6 - 4.021776399375e9,
     "c_f_squared": 7 / 12,
     "unc": math.sqrt(120e3 ** 2 + 44e3 ** 2)},
    {"frequency": 335.116048807e12 + 510.860e6 - 4.021776399375e9,
     "c_f_squared": 5 / 12,
     "unc": math.sqrt(120e3 ** 2 + 34e3 ** 2)},
    {"frequency": 335.116048807e12 - 656.820e6 + 5.170855370625e9,
     "c_f_squared": 7 / 36,
     "unc": math.sqrt(120e3 ** 2 + 44e3 ** 2)},
    {"frequency": 335.116048807e12 + 510.860e6 + 5.170855370625e9,
     "c_f_squared": 7 / 12,
     "unc": math.sqrt(120e3 ** 2 + 34e3 ** 2)},
]

# print(scipy.constants.physical_constants["hbar"])
CELL_LENGTH = 0.012  # 12 mm

T_TO_F_COEFFICIENT = - 1.408540410132690e+12
print(f"T to F coefficient: {Decimal(T_TO_F_COEFFICIENT):.6E} Hz/s")
GAIN = 475000  # Taken from datasheet
RESPONSIVITY = 0.6
AGGREGATION_WINDOW = 1
GAMMA = 2 * pi * 4.56e6
CS_NUCLEAR_SPIN = 7 / 2  # Cs nuclear spin

# The sensitivity of the oscilloscope data
UNC_VOLTAGE = 0.01e-3
UNC_TIME = 0.01e-6

CS_ATOMIC_MASS = 132.90545 * 1.67377e-27


def calculate_sigma(frequency, temp):
    return math.sqrt(k * temp / (CS_ATOMIC_MASS * c ** 2)) * 2 * pi * frequency


def calculate_line_strength(transition_index: int, transition_frequency: float):
    c_f_squared = TRANSITIONS[transition_index]["c_f_squared"]
    return c_f_squared * 9 * pi * epsilon_0 * hbar * (c ** 3) * GAMMA / (transition_frequency ** 3)


def calculate_density_from_temperature(temperature, unc_temperature):
    if temperature < 302:
        p = 10 ** (12.6709 - log10(temperature) - 4150 / temperature)
    else:
        p = 10 ** (13.178 - 1.35 * log10(temperature) - 4041 / temperature)
    #
    # print(f"P: {p}")
    # print(f"Using: k {k}, temperature {temperature}")
    unc = calc_temperature_density_unc(
        temperature,
        k,
        unc_temperature
    )

    return p / (k * temperature), unc


def calculate_density_from_gaussians(transition_index: int,
                                     transition_frequency: float,
                                     gaussian_amplitude: float,
                                     sigma_d: float,
                                     unc_amplitude: float,
                                     unc_sigma_d: float):
    s_ge = calculate_line_strength(transition_index, 2 * pi * transition_frequency)
    numerator = gaussian_amplitude * epsilon_0 * hbar * c * 2 * (2 * CS_NUCLEAR_SPIN + 1) * sigma_d
    denominator = 2 * pi * transition_frequency * s_ge * math.sqrt(pi / 2)
    # print(
    #     f"Using: epsilon_0 {epsilon_0}, c {c}, hbar {hbar}, transition fre {transition_frequency}, nuclear spin {CS_NUCLEAR_SPIN}, s_ge {s_ge}, amplitude {gaussian_amplitude}")

    unc = calc_gaussians_density_unc(
        gaussian_amplitude,
        transition_frequency,
        c,
        CS_NUCLEAR_SPIN,
        sigma_d,
        TRANSITIONS[transition_index]["c_f_squared"],
        GAMMA,
        unc_amplitude,
        TRANSITIONS[transition_index]["unc"],
        unc_sigma_d
    )
    return numerator / denominator, unc


def get_fitting_params(x_to_fit, y_to_fit, degree, indexes):
    # We only take the ranges outside the absorption peaks
    # between peaks range = 300000:450000
    # end range = 0:200000
    # initial range = 550000:600000
    x_fit = np.concatenate((
        x_to_fit.iloc[int(indexes["forth"] / AGGREGATION_WINDOW):],
        x_to_fit.iloc[int(indexes["second"] / AGGREGATION_WINDOW):int(indexes["third"] / AGGREGATION_WINDOW)],
        x_to_fit.iloc[:int(indexes["first"] / AGGREGATION_WINDOW)]))

    y_fit = np.concatenate((
        y_to_fit.iloc[int(indexes["forth"] / AGGREGATION_WINDOW):],
        y_to_fit.iloc[int(indexes["second"] / AGGREGATION_WINDOW):int(indexes["third"] / AGGREGATION_WINDOW)],
        y_to_fit.iloc[:int(indexes["first"] / AGGREGATION_WINDOW)]))

    return np.polyfit(x_fit, y_fit, degree, full=False, cov=True)


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


a_temp_formula = 0.001129241
b_temp_formula = 0.0002341077
c_temp_formula = 0.000000087755
unc_thermistor = 1
file_name = 'temperatures/40.csv'
thermistor_value = {'temperatures/40.csv': 5400,
                    'temperatures/38.csv': 5800,
                    'temperatures/35.csv': 6600,
                    'temperatures/33.csv': 7000,
                    'temperatures/31.csv': 7800,
                    'temperatures/29.csv': 8600,
                    'temperatures/27.csv': 9200,
                    'temperatures/25.csv': 10000,
                    'AAAA.csv': 12335}[file_name]

temperature_indexes = {
    293: {
        "skip_rows": 12,
        "skip_footer": 100000,
        "first": 200000,
        "second": 300000,
        "third": 450000,
        "forth": 550000,
    },
    298: {
        "skip_rows": 20012,
        "skip_footer": 55000,
        "first": 180000,
        "second": 270000,
        "third": 460000,
        "forth": 540000,
    },
    300: {
        "skip_rows": 20012,
        "skip_footer": 55000,
        "first": 180000,
        "second": 265000,
        "third": 460000,
        "forth": 540000,
    },
    302: {
        "skip_rows": 10012,
        "skip_footer": 65000,
        "first": 190000,
        "second": 260000,
        "third": 470000,
        "forth": 540000,
    },
    304: {
        "skip_rows": 15012,
        "skip_footer": 55000,
        "first": 200000,
        "second": 275000,
        "third": 480000,
        "forth": 545000,
    },
    306: {
        "skip_rows": 15012,
        "skip_footer": 55000,
        "first": 200000,
        "second": 275000,
        "third": 480000,
        "forth": 545000,
    },
    308: {
        "skip_rows": 15012,
        "skip_footer": 55000,
        "first": 200000,
        "second": 270000,
        "third": 480000,
        "forth": 545000,
    },
    311: {
        "skip_rows": 15012,
        "skip_footer": 55000,
        "first": 200000,
        "second": 270000,
        "third": 480000,
        "forth": 545000,
    },
    313: {
        "skip_rows": 15012,
        "skip_footer": 55000,
        "first": 200000,
        "second": 270000,
        "third": 480000,
        "forth": 545000,
    },
}

temperature = 1 / (a_temp_formula + b_temp_formula * math.log(thermistor_value) + c_temp_formula * (
        math.log(thermistor_value) ** 3))
unc_temperature = calc_temperature_unc(
    thermistor_value,
    a_temp_formula,
    b_temp_formula,
    c_temp_formula,
    unc_thermistor
)

print(f"Working at temperature: {temperature} +- {unc_temperature} K")

df = pd.read_csv(file_name,
                 names=["Second", "Volt", "Volt.1", "Volt.2"],
                 skiprows=temperature_indexes[round(temperature)]["skip_rows"],
                 skipfooter=temperature_indexes[round(temperature)]["skip_footer"],
                 engine="python")

# Sin

aggregation_index = []
for i in range(len(df.index)):
    aggregation_index.append(int(i / AGGREGATION_WINDOW))
df["AggregationIndex"] = aggregation_index

data = df.groupby("AggregationIndex", as_index=False).mean()

n_points = len(data.index)
# print(f"Total number of points: {n_points}")

# Going from seconds to Hz
data["Frequency"] = data["Second"] * T_TO_F_COEFFICIENT
data["Frequency"] = data["Frequency"] - data["Frequency"].iloc[n_points - 1]
# ASSUMPTION: t_to_f_coefficient does not have any uncertainty
unc_frequency = UNC_TIME * T_TO_F_COEFFICIENT

# Calculating output power from the measured voltage
data["OutputPower"] = data["Volt.2"] / (RESPONSIVITY * GAIN)
unc_output_power = UNC_VOLTAGE / (RESPONSIVITY * GAIN)

(a_2, a_1, a_0), cov_matrix = get_fitting_params(
    data["Frequency"],
    data["OutputPower"],
    2,
    temperature_indexes[round(temperature)])

unc_a_2, unc_a_1, unc_a_0 = np.sqrt(np.diag(cov_matrix))
print(f"Linear fit paramaters deviations: {unc_a_2, unc_a_1, unc_a_0}, output power uNC: {unc_output_power}")

data["Fit"] = a_2 * data["Frequency"] ** 2 + a_1 * data["Frequency"] + a_0

data.plot(x="Frequency", y=["OutputPower", "Fit"])  # Use 'Time' column as x-axis
plt.title(f"Output Power vs Frequency (aggr win = {AGGREGATION_WINDOW}) with T = {round(temperature)} K")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Measured Power [W]")
plt.grid(True)
plt.show()
plt.figure(1)

guess_params = {
    293: [
        0.2e10, 12, 0.02e10,  # First peak
        0.32e10, 8, 0.02e10,  # Second peak
        0.98e10, 3, 0.02e10,  # Third peak
        1.1e10, 11, 0.02e10,  # Fourth peak
    ],
    298: [
        0.29e10, 17.5, 0.02e10,  # First peak
        0.37e10, 11, 0.02e10,  # Second peak
        1.05e10, 6, 0.02e10,  # Third peak
        1.15e10, 16, 0.02e10,  # Fourth peak
    ],
    300: [
        0.26e10, 12, 0.02e10,  # First peak
        0.35e10, 8, 0.02e10,  # Second peak
        1.1e10, 3, 0.02e10,  # Third peak
        1.2e10, 11, 0.02e10,  # Fourth peak
    ],
    302: [
        0.3e10, 12, 0.02e10,  # First peak
        0.4e10, 8, 0.02e10,  # Second peak
        1.08e10, 6, 0.02e10,  # Third peak
        1.22e10, 11, 0.02e10,  # Fourth peak
    ],
    304: [
        0.26e10, 12, 0.02e10,  # First peak
        0.35e10, 8, 0.02e10,  # Second peak
        1.1e10, 3, 0.02e10,  # Third peak
        1.2e10, 11, 0.02e10,  # Fourth peak
    ],
    306: [
        0.26e10, 12, 0.02e10,  # First peak
        0.35e10, 8, 0.02e10,  # Second peak
        1.05e10, 8, 0.02e10,  # Third peak
        1.2e10, 11, 0.02e10,  # Fourth peak
    ],
    308: [
        0.26e10, 12, 0.02e10,  # First peak
        0.35e10, 8, 0.02e10,  # Second peak
        1.05e10, 8, 0.02e10,  # Third peak
        1.2e10, 11, 0.02e10,  # Fourth peak
    ],
    311: [
        0.26e10, 12, 0.02e10,  # First peak
        0.35e10, 8, 0.02e10,  # Second peak
        1.05e10, 8, 0.02e10,  # Third peak
        1.2e10, 11, 0.02e10,  # Fourth peak
    ],
    313: [
        0.26e10, 12, 0.02e10,  # First peak
        0.35e10, 8, 0.02e10,  # Second peak
        1.05e10, 8, 0.02e10,  # Third peak
        1.2e10, 11, 0.02e10,  # Fourth peak
    ]}

data["TransmissionCoefficient"] = data["OutputPower"] / data["Fit"]
data["AbsorptionCoefficient"] = np.log(data["TransmissionCoefficient"]) / CELL_LENGTH

optimal_params, covariance_matrix, _, msg, _ = curve_fit(
    f=gaussian_func,
    xdata=data["Frequency"].to_numpy(),
    ydata=data["AbsorptionCoefficient"].to_numpy(),
    p0=guess_params[round(temperature)],
    full_output=True)

gauss_fit_unc = np.sqrt(np.diag(covariance_matrix))

# print(f"Gaussian fit uncertainties: {gauss_fit_unc}")
print(f"Optimal parameters found: {optimal_params}, with msg: {msg}")

data["GaussianFit"] = gaussian_func(data["Frequency"], *optimal_params)

data.plot(x="Frequency", y=["AbsorptionCoefficient", "GaussianFit"])  # Use 'Time' column as x-axis
# data.plot(x="Frequency", y=["AbsorptionCoefficient"])  # Use 'Time' column as x-axis
plt.title(
    f"Absorption Coefficient vs Frequency (aggr win = {AGGREGATION_WINDOW}) with T = {round(temperature)} K")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Absorption Coefficient [1/m]")
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

# Uncertainty on the temperature is of 1 degree (we should calculate it from the termistor formula)
n_from_temp, n_from_temp_unc = calculate_density_from_temperature(temperature, unc_temperature)
print(
    f"Atomic density calculated from the cell temperature: {Decimal(n_from_temp):.8E} +- {Decimal(n_from_temp_unc):.4E} m^(-3)")
n_from_fit = []
for i in range(len(transitions_parameters)):
    n_from_fit.append(
        calculate_density_from_gaussians(
            i,
            abs(transitions_parameters[i]['theoretical_transition_frequency']),
            abs(transitions_parameters[i]['absorption_amplitude']),
            abs(transitions_parameters[i]['sigma_d']),
            gauss_fit_unc[i * 3 + 1],
            gauss_fit_unc[i * 3 + 2],
        )
    )
    print(
        f"Atomic density calculated from the fitting for {i}-th transition: {Decimal(n_from_fit[i][0]):.6E} +- {Decimal(n_from_fit[i][1]):.4E} m^(-3)")

print("------------------------------------------------------------")
val = abs(transitions_parameters[3]['center_frequency']) - abs(transitions_parameters[2]['center_frequency'])
print(
    f"Small difference 1: EXP {Decimal(val):.4E} Hz - TRUE {Decimal(1167.680e6):.4E} Hz - DIFF {Decimal(1167.680e6 - val):.4E} Hz")

val = abs(transitions_parameters[1]['center_frequency']) - abs(transitions_parameters[0]['center_frequency'])
print(
    f"Small difference 2: EXP {Decimal(val):.4E} Hz - TRUE {Decimal(1167.680e6):.4E} Hz - DIFF {Decimal(1167.680e6 - val):.4E} Hz")

val = abs(transitions_parameters[2]['center_frequency']) - abs(transitions_parameters[0]['center_frequency'])
print(
    f"Intermediate difference: EXP {Decimal(val):.4E} Hz - TRUE {Decimal(9.192631770e9):.4E} Hz - DIFF {Decimal(9.192631770e9 - val):.4E} Hz")

theoretical_sigma_1 = calculate_sigma(TRANSITIONS[0]["frequency"], temperature)
experimental_sigma_1 = transitions_parameters[0]['sigma_d']

theoretical_sigma_2 = calculate_sigma(TRANSITIONS[1]["frequency"], temperature)
experimental_sigma_2 = transitions_parameters[1]['sigma_d']

theoretical_sigma_3 = calculate_sigma(TRANSITIONS[2]["frequency"], temperature)
experimental_sigma_3 = transitions_parameters[2]['sigma_d']

theoretical_sigma_4 = calculate_sigma(TRANSITIONS[3]["frequency"], temperature)
experimental_sigma_4 = transitions_parameters[3]['sigma_d']

print("------------------------------------------------")
print(
    f"1st: Experimental sigma: {Decimal(experimental_sigma_1):.6E} Hz, Theoretical sigma: {Decimal(theoretical_sigma_1):.6E} Hz")
print(
    f"2st: Experimental sigma: {Decimal(experimental_sigma_2):.6E} Hz, Theoretical sigma: {Decimal(theoretical_sigma_2):.6E} Hz")
print(
    f"3st: Experimental sigma: {Decimal(experimental_sigma_3):.6E} Hz, Theoretical sigma: {Decimal(theoretical_sigma_3):.6E} Hz")
print(
    f"4st: Experimental sigma: {Decimal(experimental_sigma_4):.6E} Hz, Theoretical sigma: {Decimal(theoretical_sigma_4):.6E} Hz")
