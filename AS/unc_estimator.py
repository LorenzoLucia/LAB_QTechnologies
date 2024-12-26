import math
from math import pi


def calc_temperature_unc(
        resistance,
        a,
        b,
        c,
        unc_resistance
):
    derivative_denominator = a + b * math.log(resistance) + c * (math.log(resistance) ** 3)
    derivative_numerator = - (b + c * math.log(resistance) ** 2) / resistance
    return abs(derivative_numerator / derivative_denominator * unc_resistance)


def calc_gaussians_density_unc(
        amplitude,
        transition_frequency,
        c,
        I,
        sigma_d,
        c_f_squared,
        gamma,
        unc_amplitude,
        unc_transition_frequency,
        unc_sigma_d
):
    denominator = c_f_squared * 9 * pi * (c ** 2) * gamma * math.sqrt(pi / 2)
    derivative_amplitude = 2 * (2 * I + 1) * sigma_d * (2 * pi * transition_frequency) ** 2 / denominator
    derivative_frequency = amplitude * 4 * (2 * I + 1) * 2 * pi * transition_frequency * sigma_d / denominator
    derivative_sigma_d = amplitude * 2 * (2 * I + 1) * (2 * pi * transition_frequency) ** 2 / denominator

    return math.sqrt((unc_amplitude * derivative_amplitude) ** 2 + (unc_sigma_d * derivative_sigma_d) ** 2 + (
            unc_transition_frequency * derivative_frequency) ** 2)


def calc_temperature_density_unc(
        temperature,
        k_b,
        unc_temperature
):
    derivative_temperature: float
    if temperature > 302:
        derivative_temperature = 10 ** 13.178 * (math.log(10) * 10 ** (-4041 / temperature) * 4041 * temperature ** (
            -2) * k_b * (temperature ** 2) - 10 ** (-4041 / temperature) * 2.35 * k_b * temperature ** 1.35) / (
                                         k_b * temperature ** 2.35) ** 2

    else:
        derivative_temperature = 10 ** 12.6709 * (math.log(10) * 10 ** (-4150 / temperature) * 4150 * temperature ** (
            -2) * k_b * (temperature ** 2) - 10 ** (-4150 / temperature) * 2 * k_b * temperature) / (
                                         k_b * temperature ** 2) ** 2

    return abs(derivative_temperature * unc_temperature)
