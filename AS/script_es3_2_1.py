import math
from math import pi, sqrt, log10

from scipy.constants import hbar, epsilon_0, c, k

# constants
MHz = 1e6
mm = 1e-3
GAMMA = 2 * pi * 4.56 * MHz
CS_NUCLEAR_SPIN = 7 / 2  # Cs nuclear spin


# S_ge = line strength
def calculate_line_strength(transition_index: int, transition_frequency: float):
    c_f_squared = c_f_square(transition_index)
    return c_f_squared * 9 * pi * epsilon_0 * hbar * (c ** 3) * GAMMA / (transition_frequency ** 3)


# density w.r.t. Temperature
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


def c_f_square(transition_index: int):
    # c_F^2 from table 1 page 7 
    # need a way to encode the transition
    match transition_index:
        # just a trial, I need the transition to be encoded from previous scripts
        case 0:
            return 7 / 12
        case 1:
            return 5 / 12
        case 2:
            return 7 / 36
        case 3:
            return 7 / 12


def main(omega_ge, sigma_D, k):
    if ((len(omega_ge) == len(sigma_D)) and (len(omega_ge) == len(k)) and (len(k) == len(sigma_D))):
        dim = len(omega_ge)
        Sge_list = []
        n_list = []

        for i in range(dim):
            Sge_list.append(c_f_square(i) * 9 * pi * epsilon_0 * hbar * (c ** 3) * GAMMA / (omega_ge[i] ** 3))
            n_list.append(
                epsilon_0 * k[i] * hbar * c * 2 * (2 * CS_NUCLEAR_SPIN + 1) * sqrt(2 / pi) * sigma_D[i] / (
                        omega_ge[i] * Sge_list[i]))
        return n_list
    else:
        return ([-99])


print(main([1], [1], [1]))
print(calculate_density_from_temperature([273]))
