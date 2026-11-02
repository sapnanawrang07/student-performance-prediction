import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset load
df = pd.read_csv("data/student-mat.csv", sep=";")

# Dataset ki first 5 rows
print("First 5 Student Records:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Statistical analysis
print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Student final marks distribution

plt.figure(figsize=(8, 5))

plt.hist(df["G3"], bins=10, edgecolor="black")

plt.title("Student Final Marks Distribution")
plt.xlabel("Final Marks")
plt.ylabel("Number of Students")

plt.savefig("marks_distribution.png")

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(x="studytime", y="G3", data=df)

plt.title("Study Time vs Final Marks")
plt.xlabel("Study Time")
plt.ylabel("Final Marks")

plt.savefig("studytime_analysis.png")

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(df["absences"], df["G3"])

plt.title("Student Absences vs Final Marks")
plt.xlabel("Number of Absences")
plt.ylabel("Final Marks")

plt.savefig("attendance_analysis.png")

plt.show()