import numpy as np
from utilities import plotting
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def evaluate_error_fitting(fitted_curve, ydata):
    residuals = ydata - fitted_curve
    return np.sqrt(np.mean(residuals**2)) 

def opt_cos_square_fit(x, y, max_error = 1e-5,max_iter=100, initial_guesses = [1,1,1,0]):
    best_popt = initial_guesses
    best_error = np.inf
    
    i = 0
    while i < max_iter and best_error > max_error:
        popt, fitted_curve = cos_square_fitting(x = x, y = y, initial_guesses = best_popt)
        
        error = evaluate_error_fitting(fitted_curve, y)
        
        if error < best_error:
            best_error = error
            best_popt = popt
            best_fitted_curve = fitted_curve
        
        best_popt = best_popt + np.random.normal(0, 0.1, len(best_popt))
        
        #print(f"Iteration {i+1}: Error = {error}")
        i += 1
    
    return best_popt, best_fitted_curve, best_error


def evaluate_error(initial_curve, fitted_curve):
    #copy in utilities

    initial_curve = np.array(initial_curve)
    fitted_curve = np.array(fitted_curve)
    
    if len(initial_curve) != len(fitted_curve):
        raise ValueError("Initial curve and fitted curve must have the same length.")
    
    residuals = initial_curve - fitted_curve
    mae = np.mean(np.abs(residuals))  # Mean Absolute Error
    mse = np.mean(residuals ** 2)     # Mean Squared Error
    rmse = np.sqrt(mse)               # Root Mean Squared Error
    max_error = np.max(np.abs(residuals))  # Maximum Absolute Error
    r_squared = 1 - (np.sum(residuals ** 2) / np.sum((initial_curve - np.mean(initial_curve)) ** 2))  # R²


    #return mae, mse, rmse, max_error, r_squared

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Max Error": max_error,
        "R^2": r_squared
    }

def cos_square(x, A, B, C, D):
    x = np.asarray(x) 
    return A * (np.cos(B * x + C) ** 2) + D


def cos_square_fitting(x,y, initial_guesses = [1, 1, 1, 0]):
    popt, _ = curve_fit(cos_square, x, y, p0=initial_guesses)
    fitted_curve = cos_square(x, *popt)

    return popt, fitted_curve

def find_maximum(vector):
    maximum = vector[0]
    max_index = 0
    for i in range(1, len(vector)-1):
        if vector[i] > maximum:
            max_index = i
        
    return maximum, max_index




angles_lamina_1 = [50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114]
powers_lamina_1 = [0.483,0.523, 0.544,0.55,0.532,  0.479, 0.427, 0.361, 0.284, 0.218, 0.143, 0.082, 0.033, 0.008, 0.0006, 0.014, 0.05];

angles_lamina_2 = [70, 74, 78, 82, 86, 90,94,  98, 102, 106, 110]
powers_lamina_2 = [0.296, 0.36, 0.414, 0.463, 0.498,0.505,  0.505, 0.472, 0.432, 0.372, 0.302]

points_fitting1 , fitted_curve1, _ = opt_cos_square_fit(x = angles_lamina_1, y = powers_lamina_1)
points_fitting2 , fitted_curve2, _ = opt_cos_square_fit(x = angles_lamina_2, y = powers_lamina_2)

errors1 = evaluate_error(initial_curve = powers_lamina_1, fitted_curve = fitted_curve1)
errors2 = evaluate_error(initial_curve = powers_lamina_2, fitted_curve = fitted_curve2)

verbose = True

if verbose:
    #lamina 1
    plt.plot(angles_lamina_1, powers_lamina_1, label="original curve", color="red")
    plt.plot(angles_lamina_1, fitted_curve1, label="fitted curve", color="blue")
    plt.xlabel("angles (degrees)")
    plt.ylabel("fitted curve")
    plt.title("fitting of the first lamina")
    plt.legend()
    plt.show()

    #lamina 2
    plt.plot(angles_lamina_2, powers_lamina_2, label="original curve", color="red")
    plt.plot(angles_lamina_2, fitted_curve2, label="fitted curve", color="blue")
    plt.xlabel("angles (degrees)")
    plt.ylabel("fitted curve")
    plt.title("fitting of the second lamina")
    plt.legend()
    plt.show()


print("\n\n")
print(f"The found parameters for the fisrt lamina are: [A: {points_fitting1[0]},B : {points_fitting1[1]}, C: {points_fitting1[2]},D: {points_fitting1[3]},")
print(f"The found errors are: [Mean absolute error: {errors1['MAE']}, "
      f"Mean squared error: {errors1['MSE']}, "
      f"Root mean squared error: {errors1['RMSE']}, "
      f"Max error: {errors1['Max Error']}, "
      f"R^2: {errors1['R^2']}]")

print(f"\nThe found parameters are: [A: {points_fitting2[0]},B : {points_fitting2[1]}, C: {points_fitting2[2]},D: {points_fitting2[3]},")
print(f"The found errors are: [Mean absolute error: {errors2['MAE']}, "
      f"Mean squared error: {errors2['MSE']}, "
      f"Root mean squared error: {errors2['RMSE']}, "
      f"Max error: {errors2['Max Error']}, "
      f"R^2: {errors2['R^2']}]")