import matplotlib.pyplot as plt
import pandas as pd
import scipy

from utilities import find_maxima as f_max


def plotting(x, y, name, x_axis, y_axis, marker):
    # Create a plot
    plt.plot(x, y, marker=marker, label=name)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(name)
    plt.legend()
    plt.show()


class Fiber:
    def __init__(self, length=None, core_diameter=None, cladding_diameter=None, refractive_index=None, attenuation=None,
                 velocity=None):
        self.length = length
        self.core_diameter = core_diameter
        self.cladding_diameter = cladding_diameter
        self.refractive_index = refractive_index
        self.attenuation = attenuation
        self.velocity = velocity

    def __repr__(self):
        return (f"Fiber(length={self.length}, core_diameter={self.core_diameter}, "
                f"cladding_diameter={self.cladding_diameter}, "
                f"refractive_index={self.refractive_index}, attenuation={self.attenuation})")


file_path = 'OTDR8.xlsx'

data = pd.read_excel(file_path, engine='openpyxl')

bins_vector = data.iloc[:, 0].tolist()
coincidences_vector = data.iloc[:, 1].tolist()
verbose = True

if verbose:
    plotting(x=bins_vector, y=coincidences_vector, name='Coincidences plot', x_axis='Time [bin=100 ps]',
             y_axis='Coincidences',
             marker='none')

maxima_indices, maxima_values = f_max(vector=coincidences_vector, thres=500)
print(len(maxima_indices))

if verbose:
    for i in range(len(maxima_indices)):
        print("Found maxima at index ", maxima_indices[i], "with value", maxima_values[i])

fiber_refr_index = 1.5
fibers = []
for i in range(len(maxima_indices)):
    velocity = scipy.constants.c / fiber_refr_index
    fiber = Fiber(velocity=velocity)
    fibers.append(fiber)

bin_width = 100e-12

for i in range(len(fibers)):
    fibers[i].length = fibers[i].velocity * (bins_vector[maxima_indices[i]] * bin_width) / 2

    print("The fiber", i + 1, "has a length of ", fibers[i].length, "m")
