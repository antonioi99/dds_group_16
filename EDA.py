import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_csv("/Users/markrudolf/VSCode/Seoul_Bike_Demand/SeoulBikeData.csv", encoding="latin1")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

print(df.head())
print(df.info())
print(df.describe())
