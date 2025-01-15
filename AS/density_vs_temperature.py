import numpy as np
from matplotlib import pyplot as plt

temperatures = [
    293, 298, 300, 302, 304, 306, 308, 311, 313
]

temp_densities = [
    2.8369329e16, 4.6001982e16, 5.5704864e16, 6.5012798e16, 8.06787076e16, 10.2252089e16, 11.6282556e16, 15.411933e16,
    18.800319e16
]

first_densities = [
    1.35284e16, 1.81682e16, 1.91235e16, 1.96184e16, 2.16399e16, 2.47847e16, 2.47847e16, 2.94606e16, 3.48097e16
]

second_densities = [
    1.25257e16, 1.71819e16, 1.82098e16, 1.86278e16, 2.02695e16, 2.34337e16, 2.46121e16, 2.79635e16, 3.25738e16
]
third_densities = [
    1.51991e16, 2.14489e16, 2.20505e16, 2.29086e16, 2.48545e16, 2.84528e16, 2.9952e16, 3.3713e16, 3.9484e16
]

fourth_densities = [
    1.40532e16, 1.96418e16, 2.04847e16, 2.09679e16, 2.28822e16, 2.64246e16, 2.78219e16, 3.13934e16, 3.69875e16
]
print(np.log10(temp_densities))
print(np.log10(first_densities))
plt.plot(temperatures, temp_densities, label='Theoretical value', linestyle='-', color='blue')
plt.plot(temperatures, first_densities, label='Fg = 4 to Fe = 3', linestyle='-', marker="x", color='green')
plt.plot(temperatures, second_densities, label='Fg = 4 to Fe = 4', linestyle='-', marker="x", color='red')
plt.plot(temperatures, third_densities, label='Fg = 3 to Fe = 3', linestyle='-', marker="x", color='purple')
plt.plot(temperatures, fourth_densities, label='Fg = 3 to Fe = 4', linestyle='-', marker="x", color='orange')
plt.title(f"Atomic density vs Temperature")
plt.xlabel("Temperature [K]")
plt.ylabel("Atomic density [cm^-3]")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(1)

plt.plot(temperatures, np.log10(temp_densities), label='Theoretical value', linestyle='-', color='blue')
plt.plot(temperatures, np.log10(first_densities), label='Fg = 4 to Fe = 3', linestyle='-', marker="x", color='green')
plt.plot(temperatures, np.log10(second_densities), label='Fg = 4 to Fe = 4', linestyle='-', marker="x", color='red')
plt.plot(temperatures, np.log10(third_densities), label='Fg = 3 to Fe = 3', linestyle='-', marker="x", color='purple')
plt.plot(temperatures, np.log10(fourth_densities), label='Fg = 3 to Fe = 4', linestyle='-', marker="x", color='orange')
plt.title(f"Log of Atomic density vs Temperature")
plt.xlabel("Temperature [K]")
plt.ylabel("Log Atomic density")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(1)
