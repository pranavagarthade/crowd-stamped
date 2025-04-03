import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv("people_count_results.csv")

print(df.head())

plt.hist(df["People_Count"], bins=20, color='blue', edgecolor='black')
plt.xlabel("Number Of People in Image")
plt.ylabel("Frequency")
plt.title("Crowd Density Distribution")
plt.show()