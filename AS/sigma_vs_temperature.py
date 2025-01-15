import math

from matplotlib import pyplot as plt
from scipy.constants import k, c


def calculate_sigma(frequency, temp):
    return math.sqrt(k * temp / (CS_ATOMIC_MASS * c ** 2)) * 2 * math.pi * frequency


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
temperatures = [
    293, 298, 300, 302, 304, 306, 308, 311, 313
]

CS_ATOMIC_MASS = 132.90545 * 1.67377e-27

first_sigma = [
    7.896461E+8, 7.756097E+8, 7.826198E+8, 7.858089E+8, 7.911081E+8, 7.951446E+8, 7.963088E+8, 7.961368E+8, 8.002325E+8
]

second_sigma = [
    8.183374E+8, 7.900867E+8, 8.015217E+8, 8.032456E+8, 8.054053E+8, 8.078698E+8, 8.088087E+8, 8.137869E+8, 8.138014E+8
]
third_sigma = [
    8.749238E+8, 8.963164E+8, 8.902701E+8, 9.035456E+8, 8.933777E+8, 8.993368E+8, 8.998008E+8, 8.981366E+8, 9.021845E+8
]

fourth_sigma = [
    9.117376E+8, 9.057094E+8, 9.131298E+8, 9.173737E+8, 9.110012E+8, 9.176501E+8, 9.162565E+8, 9.137365E+8, 9.211050E+8
]

first_theo_sigma = [calculate_sigma(TRANSITIONS[0]["frequency"], i) for i in temperatures]
second_theo_sigma = [calculate_sigma(TRANSITIONS[1]["frequency"], i) for i in temperatures]
third_theo_sigma = [calculate_sigma(TRANSITIONS[2]["frequency"], i) for i in temperatures]
forth_theo_sigma = [calculate_sigma(TRANSITIONS[3]["frequency"], i) for i in temperatures]
plt.plot(temperatures, first_sigma, label='Fg = 4 to Fe = 3', marker='x', color='green')
plt.plot(temperatures, second_sigma, label='Fg = 4 to Fe = 4', marker='x', color='red')
plt.plot(temperatures, third_sigma, label='Fg = 3 to Fe = 3', marker='x', color='purple')
plt.plot(temperatures, fourth_sigma, label='Fg = 3 to Fe = 4', marker='x', color='orange')
plt.plot(temperatures, first_theo_sigma, label="Theoretical value", linestyle='-', color='blue')
plt.title(f"Doppler linewidth vs Temperature")
plt.xlabel("Temperature [K]")
plt.ylabel("Sigma_D [s^-1]")
plt.legend()
plt.grid(True)

plt.show()
plt.figure(1)
#
# plt.plot(temperatures, np.log10(first_sigma), label='Fg = 4 to Fe = 3', linestyle='-', color='green')
# plt.plot(temperatures, np.log10(second_sigma), label='Fg = 4 to Fe = 4', linestyle='-', color='red')
# plt.plot(temperatures, np.log10(third_sigma), label='Fg = 3 to Fe = 3', linestyle='-', color='purple')
# plt.plot(temperatures, np.log10(fourth_sigma), label='Fg = 3 to Fe = 4', linestyle='-', color='orange')
# plt.title(f"Log of Atomic density vs Temperature")
# plt.xlabel("Temperature [K]")
# plt.ylabel("Log Atomic density")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.figure(1)
