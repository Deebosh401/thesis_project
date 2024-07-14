import random
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import gc

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


def Euclidean_distance(coordinates):
    return cdist(coordinates, coordinates, metric='euclidean')

def nearestNeighbour(dist_matrix):
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

class CoordinatesExtraction:
    def __init__(self, width, height, nodes_number):
        self.width = width
        self.height = height
        self.nodesNumber = nodes_number

    def generateMatrix(self, filename=None):
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

    def update(self, T0, t, schedule, length_change):
        if T0 < 0.01:
            return 0.00001 * 15000
        if length_change > 0:
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
                new_T0 = self.adaptive_cooling(T0, t,length_change)
        else:
            new_T0 = T0

        if t - self.last_improvement_iteration > 500:
            new_T0 = min(self.initial_T0, new_T0 * 1.1)
            self.last_improvement_iteration = t  # Reset improvement iteration

        return new_T0

    def adaptive_cooling(self, T0, t,length_change):
        self.mutation_prob = optimal_mutation_probability(self.sample_size)
        if random.random() < self.mutation_prob:
            self.current_schedule = self.current_schedule = random.choice(self.schedules[:-1])

        T_constant = T0 * 0.995
        T_lmc = T0 / (1 + self.alpha * t)
        T_qmc = T0 / (1 + self.alpha * t ** 2)
        T_exponential = T0 / (t ** 2)


        schedule_temps = {
            "Constant": T_constant,
            "LMC": T_lmc,
            "QMC": T_qmc,
            "Exponential": T_exponential
        }

        min_drop_schedule = min(schedule_temps, key=lambda k: schedule_temps[k])
        new_T0 = schedule_temps[min_drop_schedule]

        if length_change<=0:
            self.no_improvement_counter+=1
        else: self.no_improvement_counter = 0

        if self.no_improvement_counter>100000:
            qmc_or_lmc = random.choice(["QMC", "LMC"])
            new_T0 = schedule_temps[qmc_or_lmc]

        return new_T0

class SimulatedAnnealing:
    def __init__(self, coordinates, temp, stopping_iter, schedule_name):
        self.coordinates = coordinates
        self.temp = temp
        self.stopping_iter = stopping_iter
        self.schedule_name = schedule_name
        self.dist_matrix = Euclidean_distance(coordinates)
        self.sample_size = len(coordinates)
        self.curr_solution = nearestNeighbour(self.dist_matrix)
        self.best_solution = self.curr_solution
        self.solution_history = [self.curr_solution]
        self.curr_length = self.curr_length(self.curr_solution)
        self.initial_length = self.curr_length
        self.min_length = self.curr_length
        self.length_list = [self.curr_length]
        self.cooling_schedule = AdaptiveCoolingSchedule(temp, self.sample_size)

    def length(self, sol):
        return np.sum(self.dist_matrix[sol[i], sol[(i + 1) % self.n]] for i in range(self.n))

    def acceptance_probability(self, candidate_length):
        return math.exp(-abs(candidate_length - self.curr_length) / self.temp)

    def acceptCandidate(self, candidate):
        candidate_length = self.length(candidate)
        length_change = candidate_length- self.curr_length
        if candidate_length < self.curr_length:
            self.curr_length = candidate_length
            self.curr_solution = candidate
            if candidate_length < self.min_length:
                self.min_length = candidate_length
                self.best_solution = candidate
        else:
            if random.random() < self.acceptance_probability(candidate_length):
                self.curr_length = candidate_length
                self.curr_solution = candidate

        return length_change

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

    def plot(self, length_lists, initial_length, final_length, label):
        max_length = max(len(length_list) for length_list in length_lists)

        summed_lengths = np.zeros(max_length)

        for length_list in length_lists:
            lengths_padded = np.pad(length_list, (0, max_length - len(length_list)), constant_values=np.inf)
            summed_lengths += lengths_padded

        average_lengths = summed_lengths / len(length_lists)

        plt.plot(average_lengths, label=f'{label} (Init: {initial_length:.2f}, Final: {final_length:.2f})')

    def anneal(self):
        iteration = 1
        while iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
            length_change = self.acceptCandidate(candidate)
            self.temp = self.cooling_schedule.update(self.temp, iteration, self.schedule_name, length_change)
            self.length_list.append(self.curr_length)
            iteration += 1

        self.solution_history.append(self.curr_solution)
        self.length_list.append(self.curr_length)
        print('Minimum tour length obtained:', self.min_length)
        print('Best solution:', self.best_solution)

    def getWeightList(self):
        return self.length_list

    def saveResults(self, base_filename, output_path):
        final_result_filename = os.path.join(output_path, f"{self.schedule_name}_final_solution.png")
        self.saveFinalResult(self.coordinates, final_result_filename)
        return self.getWeightList(), output_path

def run_single_simulation(schedule, nodes, temp, stopping_iter):
    np.random.seed()
    random.seed()
    processAnealing = SimulatedAnnealing(nodes, temp, stopping_iter, schedule)
    processAnealing.anneal()
    return processAnealing.getWeightList(), processAnealing.initial_length, processAnealing.min_length


def run_simulation_for_schedule(schedule, nodes, temp, stopping_iter, num_runs, output_path):
    initial_lengths = []
    final_lengths = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(run_single_simulation, schedule, nodes, temp, stopping_iter) for _ in
                   range(num_runs)]
        length_lists = []
        for future in as_completed(futures):
            length_list, initial_length, final_length = future.result()
            length_lists.append(length_list)
            initial_lengths.append(initial_length)
            final_lengths.append(final_length)

    processAnealing= SimulatedAnnealing(nodes, temp, stopping_iter, schedule)
    processAnealing.anneal()
    processAnealing.saveResults(f"{schedule}_results", output_path)

    return schedule, length_lists, initial_lengths, final_lengths


def process_dataset(file_path, folder_path, schedules, temp, stopping_iter, num_runs, size_width, size_height):
    print(f"Processing file: {file_path}")

    generator =CoordinatesExtraction(size_width, size_height, 0)
    nodes = generator.generateMatrix(filename=file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    output_path = os.path.join('/Users/maxim/Desktop/Test', base_filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_length_lists = {schedule: [] for schedule in schedules}
    all_initial_lengths = {schedule: [] for schedule in schedules}
    all_final_lengths = {schedule: [] for schedule in schedules}

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [
            executor.submit(run_simulation_for_schedule, schedule, nodes, temp, stopping_iter, num_runs, output_path)
            for schedule in schedules]
        for future in as_completed(futures):
            schedule, length_lists, initial_lengths, final_lengths = future.result()
            all_length_lists[schedule].extend(length_lists)
            all_initial_lengths[schedule].extend(initial_lengths)
            all_final_lengths[schedule].extend(final_lengths)

    plt.figure()
    for schedule in schedules:
        length_lists = all_length_lists[schedule]
        initial_length = np.mean(all_initial_lengths[schedule])
        final_length = np.mean(all_final_lengths[schedule])
        processAnealing = SimulatedAnnealing(nodes, temp, stopping_iter, schedule)
        processAnealing.plot(length_lists,
                        initial_length,
                        final_length,
                        schedule)

    plt.ylabel('Tour Length')
    plt.xlabel('Iteration')
    plt.title(f'Average Performance Across Runs - {base_filename}')
    plt.legend()

    combined_performance_filename = os.path.join(output_path, f"{base_filename}_all_schedules_average_performance.png")
    plt.savefig(combined_performance_filename, dpi=650)
    plt.close()
    print(f"Average performance graph saved: {combined_performance_filename}")

    del generator
    del nodes
    del all_length_lists
    del all_initial_lengths
    del all_final_lengths
    del processAnealing
    gc.collect()


def main():
    temp = 15000
    stopping_iter = 1000000
    size_width = 2000
    size_height = 2000
    folder_path = ('/Users/maxim/Desktop/Test')
    num_runs = 4
    global no_improvement_count

    schedules = ["Constant", "LMC", "QMC", "Exponential", "Adaptive"]
    dataset_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                     filename.endswith('.tsp')]

    batch_size = 1000

    for i in range(0, len(dataset_files), batch_size):
        batch_files = dataset_files[i:i + batch_size]

        with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
            futures = [pool.submit(process_dataset, file_path, folder_path, schedules, temp, stopping_iter, num_runs,
                                   size_width, size_height) for file_path in batch_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()