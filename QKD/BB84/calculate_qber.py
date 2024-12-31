import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BasisVector(Enum):
    V = "V"
    H = "H"
    D = "D"
    A = "Anti"


files = [
    {
        "name": "coincidences_A_Anti_B_Anti.txt",
        "basis": (BasisVector.A, BasisVector.A)
    }, {
        "name": "coincidences_A_Anti_B_H.txt",
        "basis": (BasisVector.A, BasisVector.H)
    }, {
        "name": "coincidences_A_Anti_B_V.txt",
        "basis": (BasisVector.A, BasisVector.V)
    }, {
        "name": "coincidences_A_D_B_Anti.txt",
        "basis": (BasisVector.D, BasisVector.A)
    }, {
        "name": "coincidences_A_D_B_D.txt",
        "basis": (BasisVector.D, BasisVector.D)
    }, {
        "name": "coincidences_A_D_B_V.txt",
        "basis": (BasisVector.D, BasisVector.V)
    }, {
        "name": "coincidences_A_H_B_D.txt",
        "basis": (BasisVector.H, BasisVector.D)
    }, {
        "name": "coincidences_A_H_B_H.txt",
        "basis": (BasisVector.H, BasisVector.H)
    }, {
        "name": "coincidences_A_V_B_D.txt",
        "basis": (BasisVector.V, BasisVector.D)
    }, {
        "name": "coincidences_A_V_B_H.txt",
        "basis": (BasisVector.V, BasisVector.H)
    }
]

'''
def read_file(file_name):
    with open(file_name, "r") as file:
        temp_df = pd.read_csv(file, names=["CountsA", "CountsB", "Coincidences"], sep="\t")
        return temp_df["Coincidences"]
'''
def read_file(file_name):
    # Also Counts A and B are given
    with open(file_name, "r") as file:
        temp_df = pd.read_csv(file, names=["CountsA", "CountsB", "Coincidences"], sep="\t")
        return temp_df


def get_2_significant_figures(x, uncertanties):
    floored_log_10 = int(math.log10(uncertanties[x.name]))
    return round(x, 1 - floored_log_10)


def unc_propagation(unc_wrong_photons, unc_correct_photons):
    val = unc_correct_photons / (unc_correct_photons + unc_wrong_photons) ** 2
    return round(val, 1 - int(math.log10(val)))


counts_A_list = []
counts_B_list = []
coincidences_list = []

mean_series = {}
unc_series = {}


for file_info in files:
    file_name = file_info["name"]
    basis_pair = file_info["basis"]

    # Read the coincidences data from the file
    data = read_file(file_name)
    counts_A = data["CountsA"]
    counts_B = data["CountsB"]
    coincidences= data["Coincidences"]
    #coincidences = read_file(file_name)

    # Compute mean and uncertainty for the current basis pair
    mean_series[basis_pair] = coincidences.mean()
    unc_series[basis_pair] = coincidences.std() / np.sqrt(len(coincidences))
    #saved as Bob + alice
    coincidences_list.append([coincidences, basis_pair[0].value , basis_pair[1].value])
    counts_A_list.append([counts_A, basis_pair[0].value , basis_pair[1].value])
    counts_B_list.append([counts_B, basis_pair[0].value , basis_pair[1].value])


# Convert to Pandas Series for easier manipulation
mean_series = pd.Series(mean_series).apply(lambda x: round(x, 2))
unc_series = pd.Series(unc_series).apply(lambda x: round(x, 2))

# Generalized QBER calculation for all basis pairs
qber_results = []

for error_basis in mean_series.index:
    # Correct basis is assumed to be (H, H)
    if error_basis != (BasisVector.H, BasisVector.H) and error_basis != (BasisVector.A, BasisVector.A) and error_basis != (BasisVector.V, BasisVector.V) and \
    error_basis != (BasisVector.D, BasisVector.D):
    
        correct_basis = (BasisVector.H, BasisVector.H)

        # Calculate QBER for the current pair
        qber = round(mean_series[error_basis] * 100 / (
            mean_series[correct_basis]), 3)

        # Calculate uncertainty propagation
        unc_qber = unc_propagation(unc_series[error_basis], unc_series[correct_basis])

        # Store results as a dictionary
        qber_results.append({
            "Error Basis": error_basis,
            "Correct Basis": correct_basis,
            "QBER (%)": qber,
            "Uncertainty (%)": unc_qber
        })

# Convert results into a DataFrame for better readability
qber_df = pd.DataFrame(qber_results)



# Display the results
print(qber_df)
verbose = True

print("coincidendes", len(coincidences_list), "A", len(counts_A_list))
if verbose:
    for coincidence, bob_basis, alice_basis in coincidences_list:
        time = np.linspace(0, len(coincidences)-1, len(coincidences))
        if len(coincidence) != len(time):
            coincidence_new = []
            for i in range(len(time)):
                coincidence_new.append(coincidence[i])
            coincidence = coincidence_new

        plt.plot(time, coincidence, label=f'Coincidence ({bob_basis}, {alice_basis})', marker='o', markersize=3)

    # Add labels and title
    plt.xlabel('counts')
    plt.ylabel('Coincidences')
    plt.title('Coincidences')
    plt.legend(fontsize=7)
    plt.show()

time = np.linspace(0, len(coincidences)-1, len(coincidences))

for count_A, bob_basis, alice_basis in counts_A_list:
    if len(count_A) != len(time):
            count_A_new = []
            for i in range(len(time)):
                count_A_new.append(count_A[i])
            count_A = count_A_new

    plt.plot(time, counts_A, label=f'Basis ({bob_basis}, {alice_basis})', marker='o', markersize=3)

plt.xlabel('counts')
plt.ylabel('clicks')
plt.title('Counts A')
plt.legend(fontsize=7)
plt.show()

if verbose:
    for count_B, bob_basis, alice_basis in counts_B_list:
        time = np.linspace(0, len(coincidences)-1, len(coincidences))

        if len(count_B) > len(time):
            count_B_new = []
            for i in range(len(time)):
                count_B_new.append(count_B[i])
            count_B = count_B_new


        plt.plot(time, count_B, marker='o',label=f'Basis ({bob_basis}, {alice_basis})', markersize=3)

    plt.xlabel('counts')
    plt.ylabel('clicks')
    plt.title('Counts B')
    plt.legend(fontsize=7)
    plt.show()
'''
df = pd.DataFrame()


for i in files:
    first_vector: BasisVector
    second_vector: BasisVector
    df[i["basis"]] = read_file(i["name"])

unc_series = (df.std() / np.sqrt(len(df))).apply((lambda x: round(x, 1 - int(math.log10(x)))))
print(unc_series)

mean_series = pd.DataFrame(df.mean()).apply(get_2_significant_figures, axis=1, args=(unc_series,))[0]
print(mean_series)

# We now have the correct number of significant figures for both the uncertainties and the mean values


    #The QBER can be evaluated by doing the fraction Errors/Total. In our case we can suppose that if we have the H,H configuration (in mean) 
    #all the photons are passing, so the mean will be the correct normalization. If we measure for V,H then, in principle, we have 0 error, but we 
    #have some non 0 factor that will be mean(V,H); in this case we have to average by summing also this contribution.

qber_VH = round(mean_series[(BasisVector.V, BasisVector.H)] * 100 / (
        mean_series[(BasisVector.H, BasisVector.H)] + mean_series[(BasisVector.V, BasisVector.H)]), 3)
unc_qber_VH = unc_propagation(unc_series[(BasisVector.V, BasisVector.H)], unc_series[(BasisVector.H, BasisVector.H)])

qber_HV = round(mean_series[(BasisVector.H, BasisVector.V)] * 100 / (
        mean_series[(BasisVector.H, BasisVector.H)] + mean_series[(BasisVector.H, BasisVector.V)]), 3)
unc_qber_HV = unc_propagation(unc_series[(BasisVector.H, BasisVector.V)], unc_series[(BasisVector.H, BasisVector.H)])

# TODO: fix units
print(f"QBER calculated for V-H: {qber_VH} +- {unc_qber_VH} %")

'''