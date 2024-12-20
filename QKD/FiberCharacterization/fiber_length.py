import pandas as pd
import matplotlib.pyplot as plt
from utilities import find_maxima as f_max
from utilities import find_value

def plotting(x, y, name, x_axis, y_axis, marker):
    # Create a plot
    plt.plot(x, y, marker= marker, label= name)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(name)
    plt.legend()
    plt.show()

class Fiber:
    def __init__(self, length=None, core_diameter=None, cladding_diameter=None, refractive_index=None, attenuation=None, velocity = None):
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
verbose = False

if verbose:
    plotting(x = bins_vector, y = coincidences_vector,name = 'complete plot', x_axis= 'bins_vector', y_axis= 'coincidences', marker= 'none')





#constants 
c = 299792458 #m/s

refr_indices = [1.5, 1.5, 1.5]
fibers = []
for i in range(len(refr_indices)):
    velocity = c / refr_indices[i]
    fiber = Fiber(refractive_index=refr_indices[i], velocity=velocity)
    fibers.append(fiber)

maxima_indices = []
maxima_values = []
maxima_indices, maxima_values = f_max(vector = coincidences_vector, thres= 500)
print(len(maxima_indices))
verbose = False
if verbose:
    for i in range(len(fibers)):
        print("\nFound maxuma at index ", maxima_indices[i], "with value", maxima_values[i])

norm_of_t = 1e-12

fibers[0].length = fibers[0].velocity * bins_vector[maxima_indices[0]]
print("The fiber 1 has a length of ", fibers[0].length * norm_of_t, "m")

for i in range(1, len(fibers)):
    tot_path = sum(fibers[j].length for j in range(i))
    fibers[i].length = fibers[i].velocity * bins_vector[maxima_indices[i]] - tot_path

    print("The fiber", i+1, "has a length of ", fibers[i].length * norm_of_t, "m")