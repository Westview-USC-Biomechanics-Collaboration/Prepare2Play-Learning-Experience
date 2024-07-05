import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd

df = pd.read_csv('data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv')

# Select subset of data for plotting
timex_subset = df.iloc[18:10000, 0].astype(float).tolist()
forcex_subset = df.iloc[18:10000, 1].astype(float).tolist()
forcey_subset = df.iloc[18:10000, 2].astype(float).tolist()

plt.plot(timex_subset, forcex_subset)
plt.ylabel("ForceX")
plt.xlabel("Time")
plt.xlim([0, 3])
plt.ylim([-35, 35])


plt.show()
