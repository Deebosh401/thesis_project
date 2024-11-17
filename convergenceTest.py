import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Provided data (add the full data as a dictionary or load from file)
data = {
    "Instance name": [
        "eil51", "eil76", "eil101", "fl417", "gil262", "gr96", "gr120", "gr137", "gr202", "gr229",
        "gr431", "gr666", "kroA100", "kroA150", "kroA200", "kroB100", "kroB150", "kroB200",
        "kroC100", "kroD100", "kroE100", "lin105", "lin318", "linhp318", "p654", "pa561", "pcb442",
        "pr76", "pr107", "pr124", "pr136", "pr144", "pr152", "pr226", "pr264", "pr299", "pr439",
        "rat99", "rat195", "rat575", "rat783", "rd100", "rd400", "st70", "ts225", "tsp225",
        "u159", "u574", "u724", "ulysses16", "ulysses22", "a280", "ali535", "att48", "att532",
        "fl1400", "fl1577", "nrw1379", "pcb1173", "pr1002", "rl1304", "rl1323", "rl1889",
        "u1060", "u1432", "u1817", "vm1084", "vm1748", "bayg29", "berlin52", "bier127", "pr2392",
        "u2152", "u2319", "ch130", "ch150", "fl3795", "pcb3038", "d198", "d493", "d657", "d1291",
        "d1655", "d2103", "dantzig42", "dsj1000", "fnl4461", "rl5915", "rl5934", "pla7397",
        "rl11849", "usa13509", "brd14051", "d15112", "d18512"],

    "Adaptive": [
        430.83, 555.75, 659.46, 12309.99, 2500.64, 529.43, 1679.97, 730.69, 489.79, 1676.19, 1995.9, 3330.31,
        21622.42, 27631.21, 30270.89, 22795.46, 27023.54, 31093.83, 21095.31, 22422.74, 22837.64, 14996.55, 44476.05,
        44642.59, 35900.75, 15860.89, 52246.3, 110889.24, 44613.53, 60261.29, 102592.08, 58424.09, 74789.26, 82809.79,
        52510.31, 51032.82, 114858.32, 1280.33, 2419.68, 7162.72, 9610.41, 8163.13, 16079.44, 707.84, 129669.52, 4056.49
        , 45056.74, 39339.52, 44671.96, 71.36, 72.77, 2726.26, 2138.12, 33791.98, 91022.49, 23924.09, 27055.41, 63350,
        64232.5, 284086.88, 288497.32, 306104.73, 366044.51, 249826.78, 173040.4, 66504.25, 262866.32, 386624.59, 9020.8
        , 7827.3, 121362.75, 440521.88, 74639.53, 257005.37, 6343.18, 6722.63, 36926.29, 165098.45, 15863.98, 36350.94,
        52216.33, 57671.69, 72298.51, 87794.59, 685.18, 20454248.87, 218052.64, 667414.34, 659541.05, 27237229.05,
        1098511.2, 24136524.62, 561609.63, 1891539.35, 773749.66],
    "Optimal Solution": [
        426, 538, 629, 11861, 2378, 552.09,1649,698.53, 401.6, 1346.02, 1714.14, 2943.58, 21282, 26524, 29368,
        22141, 26130, 29437, 20749, 21294, 22068, 14379, 42029, 41345, 34643, 15196, 50778, 108159, 44303, 59030, 96772,
        57537, 73682, 80369, 49135, 48191, 107217, 1211, 2323, 6773, 8806, 7910, 15281, 675, 126643, 3916, 42080, 36905,
        41910, 68.59, 70.13, 2579, 2023.39, 31884, 83058, 20127, 22249, 56638, 56892, 259045, 252948, 270199, 316536,
        224094, 152970, 57201, 239297, 336556, 8855, 7542, 118282, 378032, 64253, 234256, 6110, 6528, 28772, 137694,
        15780, 35002, 48912, 50801, 62128, 80450, 683, 18660188, 182566, 565530, 556045, 23260728, 923288, 19982859,
        469385, 1573084, 645238]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Extract the number of cities from instance names
df['Cities'] = df['Instance name'].str.extract('(\d+)$').astype(int)

# Define clusters
def classify_cluster(cities):
    if cities <= 50:
        return '50'
    elif cities <= 100:
        return '100'
    elif cities <= 500:
        return '500'
    elif cities <= 1000:
        return '1000'
    elif cities <= 5000:
        return '5000'
    else:
        return '5000+'

df['Cluster'] = df['Cities'].apply(classify_cluster)

# Calculate deviation from optimal
df['Deviation (%)'] = ((df['Adaptive'] - df['Optimal Solution']) / df['Optimal Solution']) * 100

# Group data by clusters
clustered_data = df.groupby('Cluster').agg({
    'Deviation (%)': 'mean'
}).reset_index()

# Mean iterations per cluster
mean_iterations = {
    '50': 173.293,
    '100': 415.551,
    '500': 9742.412,
    '1000': 160457.214,
    '5000': 571632.631,
    '5000+': 985714.472
}
clustered_data['Mean Iterations'] = clustered_data['Cluster'].map(mean_iterations)

# Plot: Average Deviation from Optimal
plt.figure(figsize=(10, 6))
plt.bar(clustered_data['Cluster'], clustered_data['Deviation (%)'], color='skyblue')
plt.xlabel('Problem Size Cluster (Number of Cities)')
plt.ylabel('Average Deviation from Optimal (%)')
plt.title('Average Deviation from Optimal for Different Problem Sizes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("average_deviation_from_optimal.png", dpi=300)
plt.show()

# Plot: Mean Iterations per Cluster
plt.figure(figsize=(10, 6))
plt.plot(clustered_data['Cluster'], clustered_data['Mean Iterations'], marker='o', color='coral', linestyle='-')
plt.xlabel('Problem Size Cluster (Number of Cities)')
plt.ylabel('Mean Iterations')
plt.title('Mean Iterations per Problem Size Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("mean_iterations_per_cluster.png", dpi=300)
plt.show()
