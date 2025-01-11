import matplotlib.pyplot as plt
import pandas as pd

CELL_LENGTH = 0.012  # 12 mm
# This coefficient has been calculated by averaging the two coefficients obtained for the
# two couples of neighbouring peaks
T_TO_F_COEFFICIENT = - 1.408540410132690e+12

GAIN = 475000  # Taken from datasheet
RESPONSIVITY = 0.6
AGGREGATION_WINDOW = 10
MHz = 1e6

# Specify the file name
filename = 'I.csv'

df = pd.read_csv(filename, names=["Second", "Volt", "Volt.1", "Volt.2"], skiprows=12, skipfooter=0, engine="python")

print(f"Right peak: smallest time 0.011815 - biggest time 0.011824")
print(f"Left peak: smallest time 0.010982 - biggest time 0.010998")

# print(f"Time at maximum: {df.iloc[np.where(df['Volt.1'] == max(df['Volt.1']))]['Second']}")
aggregation_index = []
for i in range(len(df.index)):
    aggregation_index.append(int(i / AGGREGATION_WINDOW))
df["AggregationIndex"] = aggregation_index


data = df

n_points = len(data.index)
print(n_points)

data["Frequency"] = data["Second"].dropna() * T_TO_F_COEFFICIENT
data["Frequency"] = data["Frequency"] - data["Frequency"].iloc[n_points - 1]

data["OutputPower"] = data["Volt.1"] / (RESPONSIVITY * GAIN)


data.plot(x="Frequency", y="OutputPower")  # Use 'Time' column as x-axis
plt.title(f"Output Power vs Frequency (aggregation window = {AGGREGATION_WINDOW})")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Output Power [W]")
plt.grid(True)
plt.show()
plt.figure(1)
