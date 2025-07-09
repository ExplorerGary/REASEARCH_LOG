import pandas as pd

# read a csv:
path = r"D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\DATA\\ggd_fit_results.csv"

import matplotlib.pyplot as plt

df = pd.read_csv(path)

# Print basic statistics of the "beta" column
print(df["beta"].describe())

plt.hist(df["beta"], bins=50)
plt.xlabel("beta")
plt.ylabel("Frequency")
plt.title("Histogram of beta")
plt.show()