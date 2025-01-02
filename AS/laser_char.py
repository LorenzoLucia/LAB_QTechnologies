import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Specify the file name
filename = 'laser_char.csv'

df = pd.read_csv(filename)

slope, intercept = np.polyfit(df["Current"][3:].to_numpy(), df["Power"][3:].to_numpy(), 1)  # 1 indicates linear fit
df["PowerFit"] = 0
df.loc[3:, "PowerFit"] = slope * df["Current"][3:] + intercept

df.plot(x="Current", y=["Power", "PowerFit"], style=["x"])
plt.title(f"Output Power vs Driving Current")
plt.xlabel("Current [mA]")
plt.ylabel("Measured Power [mW]")
plt.grid(True)
plt.show()
plt.figure(1)
