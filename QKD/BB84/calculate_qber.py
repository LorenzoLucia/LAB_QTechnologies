import math
from enum import Enum

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


def read_file(file_name):
    with open(file_name, "r") as file:
        temp_df = pd.read_csv(file, names=["CountsA", "CountsB", "Coincidences"], sep="\t")
        return temp_df["Coincidences"]


def get_2_significant_figures(x, uncertanties):
    floored_log_10 = int(math.log10(uncertanties[x.name]))
    return round(x, 1 - floored_log_10)


def unc_propagation(unc_wrong_photons, unc_correct_photons):
    val = unc_correct_photons / (unc_correct_photons + unc_wrong_photons) ** 2
    return round(val, 1 - int(math.log10(val)))


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

qber = round(mean_series[(BasisVector.V, BasisVector.H)] * 100 / (
        mean_series[(BasisVector.H, BasisVector.H)] + mean_series[(BasisVector.V, BasisVector.H)]), 3)
unc_qber = unc_propagation(unc_series[(BasisVector.V, BasisVector.H)], unc_series[(BasisVector.H, BasisVector.H)])

# TODO: fix units
print(f"QBER calculated with H-H and V-H: {qber} +- {unc_qber} %")
