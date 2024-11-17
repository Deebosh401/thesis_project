import matplotlib.pyplot as plt
import pandas as pd
import re

# Data for instances and convergence
instances = [
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
    "rl11849", "usa13509", "brd14051", "d15112", "d18512"
]

convergence = [
    98.86619718, 96.70074349, 95.15739269, 96.21456875, 94.84272498, 98.20415697, 98.12189206, 95.39604598,
    78.04033865, 75.4706468, 83.56260282, 86.86191644, 98.40043229, 95.82562962, 96.92559929, 97.04412628,
    96.58040566, 94.37160716, 98.33095571, 94.69925801, 96.51241617, 95.70519508, 94.17771063, 92.02421091,
    96.36939641, 95.62457226, 97.1083934, 97.47571631, 99.29907681, 97.91412841, 93.98578101, 98.45822688,
    98.49724492, 96.96302057, 93.13053831, 94.10300679, 92.87303319, 94.27497936, 95.83814034, 94.24597667,
    90.86520554, 96.79987358, 94.77494928, 95.13481481, 97.61019559, 96.41241062, 92.9259981, 93.40327869,
    93.40978287, 95.96151042, 96.23556253, 94.2900349, 94.32981284, 94.01587003, 90.41092971, 81.1343469,
    78.39718639, 88.14929906, 87.09748295, 90.33300006, 85.94599681, 86.71137569, 84.35927983, 88.51697056,
    86.87951886, 83.73586126, 90.15059946, 85.12325141, 98.12761152, 96.21718377, 97.39541942, 83.46968511,
    83.83494934, 90.28867137, 96.18363339, 97.01853554, 71.65893925, 80.09757143, 99.46780735, 96.14610594,
    93.24433677, 86.47528592, 83.62974826, 90.87061529, 99.68081991, 90.38562275, 80.56229528, 81.98427316,
    81.38710896, 82.9046578, 81.02182634, 79.21385714, 80.35202872, 79.75598569, 80.08306082
]

# Create DataFrame for instances and convergence
df_convergence = pd.DataFrame({
    'Instance': instances,
    'Convergence': convergence
})

# Function to extract the number of cities from instance name
def extract_cluster(instance):
    match = re.search(r'(\d+)', instance)
    if match:
        num_cities = int(match.group(1))
        if num_cities <= 50:
            return '50'
        elif num_cities <= 100:
            return '100'
        elif num_cities <= 500:
            return '500'
        elif num_cities <= 1000:
            return '1000'
        elif num_cities <= 5000:
            return '5000'
        else:
            return '5000+'
    return None

# Apply the function to create a new column 'Cluster'
df_convergence['Cluster'] = df_convergence['Instance'].apply(extract_cluster)

# Group by 'Cluster' and calculate mean convergence for each cluster
df_cluster_convergence = df_convergence.groupby('Cluster')['Convergence'].mean().reset_index()

# Sorting the clusters in ascending order
cluster_order = ['50', '100', '500', '1000', '5000', '5000+']
df_cluster_convergence['Cluster'] = pd.Categorical(df_cluster_convergence['Cluster'], categories=cluster_order, ordered=True)
df_cluster_convergence = df_cluster_convergence.sort_values('Cluster')

# Plotting the convergence curve for each cluster with higher DPI
plt.figure(figsize=(10, 6), dpi=250)
plt.plot(df_cluster_convergence['Cluster'], df_cluster_convergence['Convergence'], marker='o', color='tab:blue', linestyle='-', markersize=8)

# Displaying the exact convergence values on the graph
for i, row in df_cluster_convergence.iterrows():
    plt.text(row['Cluster'], row['Convergence'], f'{row["Convergence"]:.2f}%', ha='center', va='bottom', fontsize=10)

# Titles and labels
plt.title('Average Convergence to Optimal Solution', fontsize=16)
plt.xlabel('Number of Cities', fontsize=14)
plt.ylabel('Convergence Rate(%)', fontsize=14)

# Set the y-axis range from 80% to 100%
plt.ylim(80, 100)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
