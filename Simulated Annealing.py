import random
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import gc
from google.colab import drive
drive.mount('/content/drive/')


def optimal_mutation_probability(n):
    if n <= 100:
        return 0.3 + 0.0005 * n
    elif n <= 500:
        return 0.35 + 0.00025 * (n - 100)
    elif n <= 1000:
        return 0.45 + 0.0002 * (n - 500)
    elif n <= 5000:
        return 0.55 + 0.0000125 * (n - 1000)
    else:
        return 0.6


def vectorToDistMatrix(coords):
    return cdist(coords, coords, metric='euclidean')


# Generate a nearest neighbor solution
def nearestNeighbourSolution(dist_matrix):
    n = len(dist_matrix)
    start_node = np.random.randint(n)
    route = [start_node]
    visited = {start_node}

    while len(route) < n:
        last_node = route[-1]
        nearest_node = min((node for node in range(n) if node not in visited),
                           key=lambda node: dist_matrix[last_node][node])
        route.append(nearest_node)
        visited.add(nearest_node)

    return route


# Generate nodes for TSP
class NodeGenerator:
    def __init__(self, width, height, nodes_number):
        self.width = width
        self.height = height
        self.nodesNumber = nodes_number

    def generate(self, filename=None):
        if filename:
            points = []
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            _, x, y = map(float, parts)
                            points.append((x, y))
                        except ValueError:
                            print(f"Warning: Skipping line with invalid data: {line}")
                    else:
                        print(f"Warning: Skipping line with unexpected number of parts: {line}")
            return np.array(points)
        else:
            xs = np.random.randint(self.width, size=self.nodesNumber)
            ys = np.random.randint(self.height, size=self.nodesNumber)
            return np.column_stack((xs, ys))


class AdaptiveCoolingSchedule:
    def __init__(self, T0, sample_size):
        self.initial_T0 = T0
        self.alpha = 0.9995
        self.c = 0.9995
        self.mutation_prob = optimal_mutation_probability(sample_size)
        self.schedules = ["Constant", "LMC", "QMC", "Exponential"]
        self.current_schedule = "Constant"
        self.sample_size = sample_size
        self.last_improvement_iteration = 0
        self.no_improvement_counter = 0

    def update(self, T0, t, schedule, weight_change):
        if T0 < 0.01:
            return 0.00001 * 15000

        if weight_change > 0:
            self.last_improvement_iteration = t
            if schedule == "Constant":
                new_T0 = T0 * 0.995
            elif schedule == "LMC":
                new_T0 = T0 / (1 + self.alpha * t)
            elif schedule == "QMC":
                new_T0 = T0 / (1 + self.alpha * t ** 2)
            elif schedule == "Exponential":
                new_T0 = T0 / (t ** 2)
            elif schedule == "Adaptive":
                new_T0 = self.adaptive_cooling(T0, t, weight_change)
        else:
            new_T0 = T0

        if t <= 490000 and t % 100000 == 0:
            new_T0 *= 1.13
        elif 500000 <= t <= 740000 and t % 50000 == 0:
            new_T0 *= 1.075
        elif t >= 750000 and t % 20000 == 0:
            new_T0 *= 1.15

        return new_T0

    def adaptive_cooling(self, T0, t,weight_change):
        self.mutation_prob = optimal_mutation_probability(self.sample_size)
        if random.random() < self.mutation_prob:
            self.current_schedule = self.current_schedule = random.choice(self.schedules[:-1])

        T_constant = T0 * 0.97
        T_lmc = T0 / (1 + (0.02 * 0.02) * t)
        T_qmc = T0 / (1 + (0.00002 * 0.02) * i ** t)
        T_exponential = T0 / (0.00005 * t ** 2)


        schedule_temps = {
            "Constant": T_constant,
            "LMC": T_lmc,
            "QMC": T_qmc,
            "Exponential": T_exponential
        }

        min_drop_schedule = min(schedule_temps, key=lambda k: schedule_temps[k])
        new_T0 = schedule_temps[min_drop_schedule]

        if weight_change<=0:
            self.no_improvement_counter+=1
        else: self.no_improvement_counter = 0

        if self.no_improvement_counter>100000:
            qmc_or_lmc = random.choice(["QMC", "LMC"])
            new_T0 = schedule_temps[qmc_or_lmc]

        return new_T0



# Simulated Annealing class
class SimulatedAnnealing:
    def __init__(self, coords, temp, stopping_iter, schedule_name):
        self.coords = coords
        self.temp = temp
        self.stopping_iter = stopping_iter
        self.schedule_name = schedule_name

        # Initialization to ensure a fresh start each run
        self.dist_matrix = vectorToDistMatrix(coords)
        self.sample_size = len(coords)
        self.curr_solution = nearestNeighbourSolution(self.dist_matrix)
        self.best_solution = self.curr_solution
        self.solution_history = [self.curr_solution]
        self.curr_weight = self.weight(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight
        self.weight_list = [self.curr_weight]

        # Pass the sample_size to the cooling schedule
        self.cooling_schedule = AdaptiveCoolingSchedule(temp, self.sample_size)

    def weight(self, sol):
        return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])

    def acceptance_probability(self, candidate_weight):
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        candidate_weight = self.weight(candidate)
        weight_change = candidate_weight - self.curr_weight
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate
        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

        return weight_change

    def saveFinalResult(self, points, filename):
        ''' Save the final result as a plot '''
        fig, ax = plt.subplots()
        x = [points[i][0] for i in self.best_solution + [self.best_solution[0]]]
        y = [points[i][1] for i in self.best_solution + [self.best_solution[0]]]
        ax.plot(x, y, 'b-o')
        ax.set_title(f'TSP Solution - {self.schedule_name}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_xlim(min(x) - 10, max(x) + 10)
        ax.set_ylim(min(y) - 10, max(y) + 10)
        plt.savefig(filename, dpi=300)  # Set DPI for high resolution
        plt.close()

    def plotLearning(self, weight_lists, initial_weight, final_weight, label):
        max_length = max(len(weight_list) for weight_list in weight_lists)
        summed_weights = np.zeros(max_length)

        for weight_list in weight_lists:
            weights_padded = np.pad(weight_list, (0, max_length - len(weight_list)), constant_values=np.inf)
            summed_weights += weights_padded

        average_weights = summed_weights / len(weight_lists)

        plt.plot(average_weights, label=f'{label} (Init: {average_weights[0]:.2f}, Final: {average_weights[-1]:.2f})')

    def anneal(self):
        iteration = 1
        while iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
            weight_change = self.accept(candidate)
            self.temp = self.cooling_schedule.update(self.temp, iteration, self.schedule_name, weight_change)
            self.weight_list.append(self.curr_weight)
            iteration += 1

        self.solution_history.append(self.curr_solution)
        self.weight_list.append(self.curr_weight)
        print('Minimum weight obtained:', self.min_weight)
        print('Best solution:', self.best_solution)

    def getWeightList(self):
        return self.weight_list

    def saveResults(self, base_filename, output_path):
        final_result_filename = os.path.join(output_path, f"{self.schedule_name}_final_solution.png")
        self.saveFinalResult(self.coords, final_result_filename)
        return self.getWeightList(), output_path



def run_single_simulation(schedule, nodes, temp, stopping_iter):
    np.random.seed()
    random.seed()
    sa = SimulatedAnnealing(nodes, temp, stopping_iter, schedule)
    sa.anneal()
    return sa.getWeightList(), sa.initial_weight, sa.min_weight


def run_simulation_for_schedule(schedule, nodes, temp, stopping_iter, num_runs, output_path):
    initial_weights = []
    final_weights = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(run_single_simulation, schedule, nodes, temp, stopping_iter) for _ in
                   range(num_runs)]
        weight_lists = []
        for future in as_completed(futures):
            weight_list, initial_weight, final_weight = future.result()
            weight_lists.append(weight_list)
            initial_weights.append(initial_weight)
            final_weights.append(final_weight)

    # Save results for one representative run
    sa = SimulatedAnnealing(nodes, temp, stopping_iter, schedule)
    sa.anneal()
    sa.saveResults(f"{schedule}_results", output_path)

    return schedule, weight_lists, initial_weights, final_weights


def process_dataset(file_path, folder_path, schedules, temp, stopping_iter, num_runs, size_width, size_height):
    print(f"Processing file: {file_path}")

    generator = NodeGenerator(size_width, size_height, 0)
    nodes = generator.generate(filename=file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # Create a single folder for the dataset based on its name in the graphs folder
    output_path = os.path.join('/content/drive/My Drive/Graphs/third_adaptation', base_filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_weight_lists = {schedule: [] for schedule in schedules}
    all_initial_weights = {schedule: [] for schedule in schedules}
    all_final_weights = {schedule: [] for schedule in schedules}

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [
            executor.submit(run_simulation_for_schedule, schedule, nodes, temp, stopping_iter, num_runs, output_path)
            for schedule in schedules]
        for future in as_completed(futures):
            schedule, weight_lists, initial_weights, final_weights = future.result()
            all_weight_lists[schedule].extend(weight_lists)
            all_initial_weights[schedule].extend(initial_weights)
            all_final_weights[schedule].extend(final_weights)

    plt.figure()
    for schedule in schedules:
        weight_lists = all_weight_lists[schedule]
        initial_weight = np.mean(all_initial_weights[schedule])
        final_weight = np.mean(all_final_weights[schedule])
        sa = SimulatedAnnealing(nodes, temp, stopping_iter, schedule)
        sa.plotLearning(weight_lists,
                        initial_weight,
                        final_weight,
                        schedule)

    plt.ylabel('Tour Length')
    plt.xlabel('Iteration')
    plt.title(f'Average Performance Across Runs - {base_filename}')
    plt.legend()

    # Save the average performance graph in the dataset folder
    combined_performance_filename = os.path.join(output_path, f"{base_filename}_all_schedules_average_performance.png")
    plt.savefig(combined_performance_filename, dpi=650)  # Set DPI for high resolution
    plt.close()
    print(f"Average performance graph saved: {combined_performance_filename}")

    del generator
    del nodes
    del all_weight_lists
    del all_initial_weights
    del all_final_weights
    del sa
    gc.collect()


def main():
    temp = 15000
    stopping_iter = 1000000
    size_width = 2000
    size_height = 2000
    folder_path = '/content/drive/My Drive/first'
    num_runs = 4
    global no_improvement_count

    schedules = ["Constant", "LMC", "QMC", "Exponential", "Adaptive"]

    # List all dataset files
    dataset_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                     filename.endswith('.tsp')]

    # Define batch size
    batch_size = 1000  # Adjust this based on your memory limits and available RAM

    for i in range(0, len(dataset_files), batch_size):
        batch_files = dataset_files[i:i + batch_size]

        # Use multiprocessing to process each dataset file in the current batch
        with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
            futures = [pool.submit(process_dataset, file_path, folder_path, schedules, temp, stopping_iter, num_runs,
                                   size_width, size_height) for file_path in batch_files]
            for future in as_completed(futures):
                try:
                    future.result()  # Ensure any exceptions are raised
                except Exception as e:
                    print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()