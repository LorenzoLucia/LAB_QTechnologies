import pandas as pd
import matplotlib.pyplot as plt
from utilities import resize_vector, calculate_means, minimals
from utilities import polynomial_fitting as poly_fit
from utilities import fit_four_gaussians as fit_4_gaus
from utilities import sum_of_gaussians as sum_of_gaus
import numpy as np

file_name = 'AAAA.csv'  # Use raw string (r'') for Windows paths

# Read the CSV file, skipping the first 12 rows
data = pd.read_csv(file_name, skiprows=12, names=['time', 'ramp', 'AC', 'DC'])

# Extract the relevant columns into separate vectors

gain_db = 70
gain = 10 ** (gain_db/10)
responsivity = 0.58
voltage_to_intensity = 1 / (gain*responsivity)


time = data['time']
AC = data['AC']
DC = data['DC']
ramp = data['ramp']


time1 = 0
time2 = 0.022

index1 = index2 = 0
for i in range(len(time)):
    if time[i] == time1:
        index1 = i
    if time[i] == time2:
        index2 = i
time = resize_vector(vector = time, index1 = index1, index2 = index2)
AC = resize_vector(vector = AC, index1 = index1, index2 = index2)
DC = resize_vector(vector = DC, index1 = index1, index2 = index2)
ramp = resize_vector(vector = ramp, index1 = index1, index2 = index2)


AC = [voltage_to_intensity * AC_value for AC_value in AC]
DC = [voltage_to_intensity * DC_value for DC_value in DC]


verbose = False #flag for print the graphs

offset_current = 0.150 #150 mA as offset current 
currents = [offset_current + ramp_current for ramp_current in ramp]

chunk = 30

time_means = calculate_means(vector = time, chunk_size = chunk)
AC_means = calculate_means(vector = AC, chunk_size = chunk)
DC_means = calculate_means(vector = DC, chunk_size = chunk)
current_means = calculate_means(vector = currents, chunk_size = chunk)

if verbose:
    #comparison between means or not means 
    plt.plot(time,AC, label="AC current", color='cyan')
    plt.plot(time_means,AC_means, label="AC current with means", color='blue')  # Fitted line
    plt.plot(time,DC, label="DC current", color='magenta')
    plt.plot(time_means,DC_means, color='red', label='DC current with means')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("DC")
    plt.title("AC/DC with or without means")
    plt.grid(True)

    # Show the plot
    plt.show()



used_time = time_means
used_current = current_means
used_AC = AC_means
used_DC = DC_means

minim, minim_index = minimals(y = DC)


if minim == []:
    print("\nMinimals not found")
else:
    diff = abs(used_current[minim_index[1]]- used_current[minim_index[0]])
    alpha = diff * 1179600

print()
current_to_freq = alpha
used_freq = [current_to_freq * current_value for current_value in used_current]

if verbose:
    plt.plot(used_freq,used_AC, label="AC", color='blue')  # Fitted line
    plt.plot(used_freq,used_DC, color='red', label='DC ')
    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("DC/AC")
    plt.title("AC and DC intensities vs freq")
    plt.grid(True)

    # Show the plot
    plt.show()

coeff, fitted_line = poly_fit(x = used_freq, y = used_DC, degree = 1)
a, b = coeff[0], coeff[1]
coeff, fitted_par = poly_fit(x = used_freq, y = used_DC, degree = 2)


if verbose:
    plt.plot(used_freq,used_DC, label="DC intensity", color='blue') 
    plt.plot(used_freq,fitted_line, color='red', label='fitted_DC_line ')
    plt.plot(used_freq,fitted_par, color='magenta', label='fitted_DC_pr ')
    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("DC/AC")
    plt.title("fittings")
    plt.grid(True)

    # Show the plot
    plt.show()

T = []
for i in range(len(used_DC)):
    T.append(used_DC[i]/fitted_line[i])

ln_T = np.log(T)


if verbose:
    plt.plot(used_freq,used_DC, label="DC intensity", color='blue') 
    plt.plot(used_freq,T, color='red', label='Trasmissivity ')
    plt.plot(used_freq,ln_T, color='cyan', label='ln(T)')
    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("Trasmissivity")
    plt.title("Trasmissivity vs DC current")
    plt.grid(True)

    # Show the plot
    plt.show()

coeff, fitted_curve = fit_4_gaus(x = used_freq, y = ln_T)

verbose = True
if verbose:
    plt.plot(used_freq,used_DC, label="DC intensity", color='blue') 
    #plt.plot(used_freq,fitted_curve, color='red', label='fitted gaussians ')
    #plt.plot(used_freq,used_DC, label="DC intensity", color='blue') 
    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("fitting")
    plt.title("fitting of the gaussians")
    plt.grid(True)

    # Show the plot
    plt.show()

A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4 = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], coeff[6], coeff[7], coeff[8], coeff[9], coeff[10], coeff[11]
print("\n\n\n\n The coefficients mu of the gaussians are:", mu1, mu2, mu3, mu4)
print("\n\n\n\n The coefficients A of the gaussians are:", A1, A2, A3, A4)
print("\n\n\n\n The coefficients sigma of the gaussians are:", sigma1, sigma2, sigma3, sigma4)


x_axis = np.linspace(300,6000,10000)

x_axis, gaussians = sum_of_gaus(mu1 = mu1, sigma1 = 10, A1 = 2, mu2= mu2, sigma2 = 10 , A2 = 1.5, mu3= mu3, sigma3 = 10, A3 = 1.2, mu4 = mu4, sigma4 = 10, A4 =0.9, x_axis = x_axis)
verbose = True

if verbose:
    plt.plot(x_axis,gaussians, label="gaussians", color='blue') 
    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("fitting")
    plt.title("fitting of the gaussians")
    plt.grid(True)

    # Show the plot
    plt.show()