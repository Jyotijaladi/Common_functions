import heapq
from tkinter import *
from tkinter import messagebox
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from gtts import gTTS
from playsound import playsound 

import os

def is_prime():
    number = int(input("Enter a number: "))

    if number <= 1:
        return False
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return False
    return True

def factorial():
    number = int(input("Enter a number: "))

    if number == 0:
        return 1
    result = 1
    for i in range(1, number + 1):
        result *= i
    return result

#simple calculator
def Calculator():
    print("choose operation:")
    temp=0
    print("""
    1. add
    2. Sub
    3. multiply
    4. divide""")
    opr=[(lambda a,b:a+b),(lambda a,b:a-b),(lambda a,b:a*b),(lambda a,b:a/b)]
    opt=int(input())
    
    a=int(input("Enter the value of a:"))
    b=int(input("Enter the value of b:"))
    res=opr[opt-1](a,b)
    print("Answer = ",res)

    

def bfs():
    graph = {}
    num_nodes = int(input("Enter the number of nodes: "))
    for i in range(num_nodes):
        node = input("Enter the node: ")
        neighbors = input(f"Enter the neighbors of node {node} (space-separated): ").split()
        graph[node] = neighbors

    start_node = input("Enter the start node: ")

    visited = set()
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node])
    return visited

def dfs():
    graph = {}
    num_nodes = int(input("Enter the number of nodes: "))
    for i in range(num_nodes):
        node = input("Enter the node: ")
        neighbors = input(f"Enter the neighbors of node {node} (space-separated): ").split()
        graph[node] = neighbors

    start_node = input("Enter the start node: ")

    visited = set()
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node])
    return visited

def water_jug_problem():
    jug1_capacity = int(input("Enter the capacity of jug 1: "))
    jug2_capacity = int(input("Enter the capacity of jug 2: "))
    target = int(input("Enter the target amount: "))

    jug1 = 0
    jug2 = 0
    steps = []
    while jug1 != target and jug2 != target:
        if jug1 == 0:
            jug1 = jug1_capacity
            steps.append((jug1, jug2))
        elif jug2 == jug2_capacity:
            jug2 = 0
            steps.append((jug1, jug2))
        else:
            amount = min(jug1, jug2_capacity - jug2)
            jug1 -= amount
            jug2 += amount
            steps.append((jug1, jug2))
    return steps

class TicTacToe:
    def __init__(self):
        self.Player1 = 'X'
        self.stop_game = False

        self.b = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

        self.states = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

        self.root = Tk()
        self.root.title("Tic Tac Toe")
        self.root.resizable(0, 0)

        for i in range(3):
            for j in range(3):
                self.b[i][j] = Button(
                    self.root,
                    height=4, width=8,
                    font=("Helvetica", "20"),
                    command=lambda r=i, c=j: self.clicked(r, c)
                )
                self.b[i][j].grid(row=i, column=j)

        self.root.mainloop()

    def clicked(self, r, c):
        if self.Player1 == "X" and self.states[r][c] == 0 and self.stop_game == False:
            self.b[r][c].configure(text="X")
            self.states[r][c] = 'X'
            self.Player1 = 'O'
        elif self.Player1 == 'O' and self.states[r][c] == 0 and self.stop_game == False:
            self.b[r][c].configure(text='O')
            self.states[r][c] = 'O'
            self.Player1 = 'X'

        self.check_if_win()

    def check_if_win(self):
        for i in range(3):
            if self.states[i][0] == self.states[i][1] == self.states[i][2] != 0:
                self.stop_game = True
                winner = messagebox.showinfo("Winner", self.states[i][0] + " Won")
                break
            elif self.states[0][i] == self.states[1][i] == self.states[2][i] != 0:
                self.stop_game = True
                winner = messagebox.showinfo("Winner", self.states[0][i] + " Won!")
                break
            elif self.states[0][0] == self.states[1][1] == self.states[2][2] != 0:
                self.stop_game = True
                winner = messagebox.showinfo("Winner", self.states[0][0] + " Won!")
                break
            elif self.states[0][2] == self.states[1][1] == self.states[2][0] != 0:
                self.stop_game = True
                winner = messagebox.showinfo("Winner", self.states[0][2] + " Won!")
                break
            elif all(self.states[row][col] != 0 for row in range(3) for col in range(3)):
                self.stop_game = True
                winner = messagebox.showinfo("Tie", "It's a tie")
                break

def start_tic_tac_toe():
    game=TicTacToe()



class Graph:
    def __init__(self):
        self.graph = {}

    def create_graph_from_input(self):
        num_nodes = int(input("Enter the number of nodes: "))
        for i in range(num_nodes):
            node = input("Enter the node: ")
            neighbors = input(f"Enter the neighbors of node {node} (space-separated): ").split()
            edges = []
            for neighbor in neighbors:
                edge_cost = float(input(f"Enter the cost of the edge between {node} and {neighbor}: "))
                edges.append((neighbor, edge_cost))
            self.graph[node] = edges
        return self.graph

def uniform_cost_search():
    graph = Graph().create_graph_from_input()
    start = input("Enter the start node: ")
    goal = input("Enter the goal node: ")
    visited = set()
    queue = [(0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            return True
        if node not in visited:
            visited.add(node)
            neighbors = graph[node]
            for neighbor, edge_cost in neighbors:
                heapq.heappush(queue, (cost + edge_cost, neighbor))
    return False

def iterative_deepening_search():
    graph = Graph().create_graph_from_input()
    start = input("Enter the start node: ")
    goal = input("Enter the goal node: ")
    max_depth = int(input("Enter the maximum depth: "))
    
    def depth_limited_search(graph, node, goal, depth):
        if node == goal:
            return True
        if depth == 0:
            return False
        for neighbor in graph[node]:
            if depth_limited_search(graph, neighbor, goal, depth - 1):
                return True
        return False

    for depth in range(max_depth):
        if depth_limited_search(graph, start, goal, depth):
            return True
    return False


#genetic algorithm
def genetic_algorithm():
    def mutation(individual, mutation_rate):
        # Perform mutation on an individual
        # Return the mutated individual
        mutated_individual = []
        for gene in individual:
            if random.random() < mutation_rate:  # Randomly mutate genes based on mutation rate
                mutated_gene = 1 - gene  # Flip the gene (0 to 1 or 1 to 0)
            else:
                mutated_gene = gene
            mutated_individual.append(mutated_gene)
        return mutated_individual

    def crossover(parent1, parent2):
        # Perform crossover between two parents to create offspring
        # Return the offspring
        crossover_point = random.randint(1, len(parent1) - 1)  # Randomly choose a crossover point
        offspring = parent1[:crossover_point] + parent2[crossover_point:]  # Combine parent genes
        return offspring

    def fitness_function(individual):
        # Calculate the fitness value of an individual
        # Return a higher value for fitter individuals
        fitness = sum(individual)  # Fitness is the sum of the genes in the individual
        return fitness

    def selection(population):
        # Perform selection to choose parents for reproduction
        # Return the selected parents
        selected_parents = random.choices(population, k=2)  # Randomly select 2 parents
        return selected_parents
    
    def create_population(population_size, chromosome_length):
            # Create a random population of individuals
            population = []
            for _ in range(population_size):
                individual = [random.randint(0, 1) for _ in range(chromosome_length)]
                population.append(individual)
            return population
    
    population_size = int(input("Enter the population size: "))
    chromosome_length = int(input("Enter the chromosome length: "))
    generations = int(input("Enter the number of generations: "))
    mutation_rate = float(input("Enter the mutation rate: "))


    population = create_population(population_size, chromosome_length)
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(individual)

    for _ in range(generations):
        # Evaluate the fitness of each individual
        fitness_values = [fitness_function(individual) for individual in population]

        # Perform selection
        parents = selection(population)

        # Create the next generation through crossover and mutation
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.choices(parents, k=2)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)
            offspring.append(child)

        # Replace the current population with the offspring
        population = offspring

    # Find the best individual in the final population
    best_individual = max(population, key=fitness_function)
    return best_individual


# 17. Hill Climbing
def hill_climbing():
    def get_neighbors(state):
        # Generate neighboring states by flipping a single bit in the current state
        neighbors = []
        for i in range(len(state)):
            neighbor = state.copy()
            neighbor[i] = 1 - neighbor[i]  # Flip the bit
            neighbors.append(neighbor)
        return neighbors

    initial_state = list(map(int, input("Enter the initial state (space-separated 0s and 1s): ").split()))

    def evaluate(state):
        # Evaluate the current state and return a score
        # Higher score indicates a better state
        # Replace this with your own evaluation function
        return sum(state)

    current_state = initial_state

    while True:
        neighbors = get_neighbors(current_state)
        best_neighbor = None
        best_score = evaluate(current_state)

        for neighbor in neighbors:
            neighbor_score = evaluate(neighbor)
            if neighbor_score > best_score:
                best_neighbor = neighbor
                best_score = neighbor_score

        if best_neighbor is None:
            return current_state

        current_state = best_neighbor



# 18. Neural Network

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer = np.random.randn(num_hidden, num_inputs + 1)
        self.output_layer = np.random.randn(num_outputs, num_hidden + 1)

    def forward_propagation(self, inputs):
        inputs = np.append(inputs, 1)  # Add bias term
        hidden_activations = self.hidden_layer @ inputs
        hidden_outputs = self.sigmoid(hidden_activations)

        hidden_outputs = np.append(hidden_outputs, 1)  # Add bias term
        output_activations = self.output_layer @ hidden_outputs
        output_outputs = self.sigmoid(output_activations)

        return output_outputs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                x = inputs[i]
                target = targets[i]

                # Forward propagation
                inputs_with_bias = np.append(x, 1)  # Add bias term
                hidden_activations = self.hidden_layer @ inputs_with_bias
                hidden_outputs = self.sigmoid(hidden_activations)

                hidden_outputs_with_bias = np.append(hidden_outputs, 1)  # Add bias term
                output_activations = self.output_layer @ hidden_outputs_with_bias
                output_outputs = self.sigmoid(output_activations)

                # Backpropagation
                output_errors = target - output_outputs
                output_delta = output_errors * output_outputs * (1 - output_outputs)

                hidden_errors = self.output_layer.T @ output_delta
                hidden_delta = hidden_errors * hidden_outputs * (1 - hidden_outputs)

                # Update weights
                self.output_layer += learning_rate * np.outer(output_delta, hidden_outputs_with_bias)
                self.hidden_layer += learning_rate * np.outer(hidden_delta, inputs_with_bias)
def implement_neural_network():
    num_inputs = int(input("Enter the number of inputs: "))
    num_hidden = int(input("Enter the number of hidden units: "))
    num_outputs = int(input("Enter the number of outputs: "))

    neural_network = NeuralNetwork(num_inputs, num_hidden, num_outputs)

    inputs = []
    targets = []
    num_samples = int(input("Enter the number of training samples: "))
    for _ in range(num_samples):
        input_data = list(map(float, input("Enter the input data (space-separated): ").split()))
        target_data = list(map(float, input("Enter the target data (space-separated): ").split()))
        inputs.append(input_data)
        targets.append(target_data)

    learning_rate = float(input("Enter the learning rate: "))
    epochs = int(input("Enter the number of epochs: "))

    neural_network.train(inputs, targets, learning_rate, epochs)

    while True:
        test_input = list(map(float, input("Enter the test input data (space-separated): ").split()))
        output = neural_network.forward_propagation(test_input)
        print("Output:", output)

        choice = input("Do you want to continue testing? (y/n): ")
        if choice.lower() != 'y':
            break

# 19. Traveling Salesperson Problem

def tsp():
    num_cities = int(input("Enter the number of cities: "))
    start = int(input("Enter the starting city: "))
    
    # Input the graph distances
    graph = []
    for i in range(num_cities):
        row = list(map(float, input(f"Enter the distances from city {i} to other cities (space-separated): ").split()))
        graph.append(row)

    visited = [False] * num_cities
    visited[start] = True
    path = [start]
    total_distance = 0

    while len(path) < num_cities:
        current_city = path[-1]
        min_distance = float('inf')
        next_city = None

        for neighbor in range(num_cities):
            if not visited[neighbor] and graph[current_city][neighbor] < min_distance:
                min_distance = graph[current_city][neighbor]
                next_city = neighbor

        if next_city is None:
            return None

        path.append(next_city)
        visited[next_city] = True
        total_distance += min_distance

    path.append(start)
    total_distance += graph[path[-2]][path[-1]]

    return path, total_distance



# 20. Text-to-Speech Conversion


def text_to_speech():
    text=input("Enter the text to convert it to speech")
    try:
        tts = gTTS(text=text, lang='en')
        filename="sound.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        #print("Exception ",str(e))
        os.remove(filename)



# 21. Classification with Multiple Classifiers
def train_classifiers():
    csv_file = input("Enter the path to the CSV file: ")
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'SGD': SGDClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'Gaussian Naive Bayes': GaussianNB()
    }

    for name, classifier in classifiers.items():
        classifier.fit(X, y)
        predictions = classifier.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"{name}: Accuracy = {accuracy:.2f}")


