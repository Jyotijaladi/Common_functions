
def factorial(number):
    if number < 0:
        return None
    elif number == 0 or number == 1:
        return 1
    else:
        result = 1
        for i in range(2, number + 1):
            result *= i
        return result


def is_prime(number):
    if number <= 1:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True

def calculator():
    print("Simple Calculator Program")
    print("Enter 'x' to exit")

    while True:
        num1 = float(input("Enter the first number: "))
        if num1 == 'x':
            break

        operator = input("Enter the operator (+, -, *, /): ")
        if operator == 'x':
            break

        num2 = float(input("Enter the second number: "))
        if num2 == 'x':
            break

        result = None

        if operator == '+':
            result = num1 + num2
        elif operator == '-':
            result = num1 - num2
        elif operator == '*':
            result = num1 * num2
        elif operator == '/':
            if num2 != 0:
                result = num1 / num2
            else:
                print("Error: Division by zero!")
                continue
        else:
            print("Invalid operator!")
            continue

        print(f"Result: {result}\n")



def simple_chatbot():
    print("Simple Chatbot")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye!")
            break

        
        chatbot_response = "Chatbot: I'm just a simple chatbot. How can I help you?"

        print(chatbot_response)


from collections import deque

def breadth_first_search(graph, start_node):
    visited = set()  # Set to track visited nodes
    queue = deque()  # Queue for BFS traversal

    visited.add(start_node)
    queue.append(start_node)

    while queue:
        node = queue.popleft()
        print(node)  # Process the node (print it in this example)

        # Explore neighbors of the current node
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def depth_first_search(graph, start_node):
    visited = set()  # Set to track visited nodes

    def dfs_helper(node):
        visited.add(node)
        print(node)  # Process the node (print it in this example)

        # Explore neighbors of the current node
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_helper(neighbor)

    dfs_helper(start_node)

from collections import deque

def water_jug_problem(capacity1, capacity2, target):
    visited = set()  # Set to track visited states
    queue = deque()  # Queue for BFS traversal

    # Initial state: both jugs are empty
    initial_state = (0, 0)
    visited.add(initial_state)
    queue.append(initial_state)

    while queue:
        current_state = queue.popleft()
        jug1, jug2 = current_state

        # Check if the target amount is achieved
        if jug1 == target or jug2 == target:
            return current_state

        # Generate all possible next states
        next_states = []

        # Fill jug1 to its capacity
        next_states.append((capacity1, jug2))
        # Fill jug2 to its capacity
        next_states.append((jug1, capacity2))
        # Empty jug1
        next_states.append((0, jug2))
        # Empty jug2
        next_states.append((jug1, 0))
        # Pour jug1 into jug2 until jug2 is full or jug1 is empty
        amount_to_pour = min(jug1, capacity2 - jug2)
        next_states.append((jug1 - amount_to_pour, jug2 + amount_to_pour))
        # Pour jug2 into jug1 until jug1 is full or jug2 is empty
        amount_to_pour = min(jug2, capacity1 - jug1)
        next_states.append((jug1 + amount_to_pour, jug2 - amount_to_pour))

        # Add unvisited next states to the queue and mark them as visited
        for state in next_states:
            if state not in visited:
                visited.add(state)
                queue.append(state)

    return None  # Target amount cannot be achieved


def play_tic_tac_toe():
    # Initialize the game board
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'

    # Function to print the game board
    def print_board():
        for row in board:
            print('|'.join(row))
            print('-' * 5)

    # Function to check if a player has won
    def check_win(player):
        # Check rows
        for row in board:
            if row.count(player) == 3:
                return True

        # Check columns
        for col in range(3):
            if board[0][col] == player and board[1][col] == player and board[2][col] == player:
                return True

        # Check diagonals
        if board[0][0] == player and board[1][1] == player and board[2][2] == player:
            return True
        if board[0][2] == player and board[1][1] == player and board[2][0] == player:
            return True

        return False

    # Function to check if the game is a draw
    def check_draw():
        for row in board:
            if ' ' in row:
                return False
        return True

    # Main game loop
    while True:
        print_board()

        # Get the current player's move
        while True:
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the column (0-2): "))

            if board[row][col] == ' ':
                board[row][col] = current_player
                break
            else:
                print("Invalid move. Try again.")

        # Check if the current player has won
        if check_win(current_player):
            print_board()
            print(f"Player {current_player} wins!")
            break

        # Check if the game is a draw
        if check_draw():
            print_board()
            print("It's a draw!")
            break

        # Switch to the other player
        current_player = 'O' if current_player == 'X' else 'X'

        import heapq


class Node:
    def __init__(self, state, cost, parent=None):
        self.state = state
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost


def uniform_cost_search(start_state, goal_state, actions, transition_model):
    frontier = []
    explored = set()

    start_node = Node(start_state, 0)
    heapq.heappush(frontier, start_node)

    while frontier:
        current_node = heapq.heappop(frontier)
        current_state = current_node.state

        if current_state == goal_state:
            return construct_path(current_node)

        explored.add(current_state)

        for action in actions:
            next_state = transition_model(current_state, action)
            if next_state is not None and next_state not in explored:
                cost = current_node.cost + 1  # Assuming all actions have a uniform cost of 1
                next_node = Node(next_state, cost, current_node)
                heapq.heappush(frontier, next_node)

    return None


def construct_path(node):
    path = []
    current_node = node
    while current_node is not None:
        path.append(current_node.state)
        current_node = current_node.parent
    path.reverse()
    return path


# Example usage
def actions(state):
    # Define the possible actions from a given state
    # For example, return a list of valid moves in a puzzle or game

    # Sample actions for a grid-based problem
    row, col = state
    moves = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    valid_moves = [(r, c) for r, c in moves if 0 <= r < 3 and 0 <= c < 3]  # Example grid size: 3x3

    return valid_moves


def transition_model(state, action):
    # Define the transition model that returns the next state given a state and action
    # For example, update the puzzle configuration or game state based on the action

    return action


def main():
    start_state = (0, 0)  # Define the start state
    goal_state = (2, 2)  # Define the goal state

    path = uniform_cost_search(start_state, goal_state, actions, transition_model)

    if path is None:
        print("Goal state is not reachable.")
    else:
        print("Uniform Cost Search Path:")
        for state in path:
            print(state)


if __name__ == '__main__':
    main()


def iterative_deepening_search(graph, start, goal, max_depth):
    # Depth-limited DFS function
    def depth_limited_search(node, depth):
        if depth == 0 and node == goal:
            return True
        if depth > 0:
            for neighbor in graph[node]:
                if depth_limited_search(neighbor, depth - 1):
                    return True
        return False

    # Main IDS loop
    for depth in range(max_depth + 1):
        if depth_limited_search(start, depth):
            return depth

    return None  # No path found within the depth limit

def min_max(game_state, depth, maximizing_player):
    # Base case: return the evaluation of the game state if depth is 0 or the game is over
    if depth == 0 or game_over(game_state):
        return evaluate(game_state)

    if maximizing_player:
        max_eval = float('-inf')

        # Generate all possible moves
        possible_moves = generate_moves(game_state)

        for move in possible_moves:
            # Apply the move to the game state
            new_game_state = apply_move(game_state, move)

            # Recursive call to min_max with decreased depth and switched player
            eval = min_max(new_game_state, depth - 1, False)

            # Update the maximum evaluation
            max_eval = max(max_eval, eval)

        return max_eval
    else:
        min_eval = float('inf')

        # Generate all possible moves
        possible_moves = generate_moves(game_state)

        for move in possible_moves:
            # Apply the move to the game state
            new_game_state = apply_move(game_state, move)

            # Recursive call to min_max with decreased depth and switched player
            eval = min_max(new_game_state, depth - 1, True)

            # Update the minimum evaluation
            min_eval = min(min_eval, eval)

        return min_eval

        def alpha_beta(game_state, depth, alpha, beta, maximizing_player):
    # Base case: return the evaluation of the game state if depth is 0 or the game is over
    if depth == 0 or game_over(game_state):
        return evaluate(game_state)

    if maximizing_player:
        max_eval = float('-inf')

        # Generate all possible moves
        possible_moves = generate_moves(game_state)

        for move in possible_moves:
            # Apply the move to the game state
            new_game_state = apply_move(game_state, move)

            # Recursive call to alpha_beta with decreased depth and switched player
            eval = alpha_beta(new_game_state, depth - 1, alpha, beta, False)

            # Update the maximum evaluation and alpha value
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)

            # Perform alpha-beta pruning
            if beta <= alpha:
                break

        return max_eval
    else:
        min_eval = float('inf')

        # Generate all possible moves
        possible_moves = generate_moves(game_state)

        for move in possible_moves:
            # Apply the move to the game state
            new_game_state = apply_move(game_state, move)

            # Recursive call to alpha_beta with decreased depth and switched player
            eval = alpha_beta(new_game_state, depth - 1, alpha, beta, True)

            # Update the minimum evaluation and beta value
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            # Perform alpha-beta pruning
            if beta <= alpha:
                break

        return min_eval

# Example functions to be implemented according to the specific game:

def game_over(game_state):
    # Check if the game is over and return True or False
    pass

def evaluate(game_state):
    # Evaluate the game state and return a score
    pass

def generate_moves(game_state):
    # Generate all possible moves from the current game state and return a list of moves
    pass

def apply_move(game_state, move):
    # Apply the given move to the game state and return the updated game state
    pass

import heapq

def heuristic(node, goal):
    # Calculate the heuristic value between the current node and the goal node
    # Return the estimated cost from the current node to the goal node
    pass

def a_star(graph, start, goal):
    open_list = [(0, start)]  # Priority queue for A* traversal
    came_from = {}  # Dictionary to store the parent node for each visited node
    g_score = {node: float('inf') for node in graph}  # Dictionary to store the cost from start to each node
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}  # Dictionary to store the total estimated cost from start to each node
    f_score[start] = heuristic(start, goal)

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct the path from the goal node to the start node
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor, edge_cost in graph[current]:
            tentative_g_score = g_score[current] + edge_cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

import heapq

def heuristic(node, goal):
    # Calculate the heuristic value between the current node and the goal node
    # Return the estimated cost from the current node to the goal node
    pass

def a_star(graph, start, goal, max_cost):
    open_list = [(0, start)]  # Priority queue for A* traversal
    came_from = {}  # Dictionary to store the parent node for each visited node
    g_score = {node: float('inf') for node in graph}  # Dictionary to store the cost from start to each node
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}  # Dictionary to store the total estimated cost from start to each node
    f_score[start] = heuristic(start, goal)
    best_path = None

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct the path from the goal node to the start node
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            # Update the best path if it's the first solution or has lower cost than the current best
            if best_path is None or g_score[goal] < g_score[best_path[-1]]:
                best_path = path

        if g_score[current] <= max_cost:
            for neighbor, edge_cost in graph[current]:
                tentative_g_score = g_score[current] + edge_cost
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return best_path

import random

def create_population(population_size, chromosome_length):
    # Create a random population of individuals
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(individual)
    return population

def fitness_function(individual):
    # Calculate the fitness value of an individual
    # Return a higher value for fitter individuals
    pass

def selection(population):
    # Perform selection to choose parents for reproduction
    # Return the selected parents
    pass

def crossover(parent1, parent2):
    # Perform crossover between two parents to create offspring
    # Return the offspring
    pass

def mutation(individual, mutation_rate):
    # Perform mutation on an individual
    # Return the mutated individual
    pass

def genetic_algorithm(population_size, chromosome_length, generations, mutation_rate):
    population = create_population(population_size, chromosome_length)

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
def hill_climbing(problem):
    current_state = problem.initial_state()

    while True:
        neighbors = problem.generate_neighbors(current_state)
        best_neighbor = None

        for neighbor in neighbors:
            if best_neighbor is None or problem.heuristic(neighbor) > problem.heuristic(best_neighbor):
                best_neighbor = neighbor

        if problem.heuristic(best_neighbor) <= problem.heuristic(current_state):
            return current_state

        current_state = best_neighbor

        import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_neural_network(train_data, train_labels, input_size, hidden_size, output_size, num_epochs, learning_rate):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return model

import itertools

def tsp(cities, start_city):
    num_cities = len(cities)
    all_cities = set(range(num_cities))
    best_path = None
    best_distance = float('inf')

    for perm in itertools.permutations(all_cities - {start_city}):
        current_path = [start_city] + list(perm) + [start_city]
        current_distance = calculate_distance(cities, current_path)

        if current_distance < best_distance:
            best_distance = current_distance
            best_path = current_path

    return best_path, best_distance

def calculate_distance(cities, path):
    distance = 0
    num_cities = len(cities)

    for i in range(num_cities - 1):
        start_city = path[i]
        end_city = path[i + 1]
        distance += cities[start_city][end_city]

    return distance

from gtts import gTTS
import playsound

def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    playsound.playsound(filename)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

def run_classifiers(csv_file, target_column):
    # Load data from CSV file
    data = pd.read_csv(csv_file)

    # Split data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    rf_accuracy = rf_classifier.score(X_test, y_test)

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    dt_accuracy = dt_classifier.score(X_test, y_test)

    # K-Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    knn_accuracy = knn_classifier.score(X_test, y_test)

    # AdaBoost Classifier
    ada_classifier = AdaBoostClassifier()
    ada_classifier.fit(X_train, y_train)
    ada_accuracy = ada_classifier.score(X_test, y_test)

    # SGD Classifier
    sgd_classifier = SGDClassifier()
    sgd_classifier.fit(X_train, y_train)
    sgd_accuracy = sgd_classifier.score(X_test, y_test)

    # Extra Trees Classifier
    et_classifier = ExtraTreesClassifier()
    et_classifier.fit(X_train, y_train)
    et_accuracy = et_classifier.score(X_test, y_test)

    # Gaussian Naive Bayes Classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    nb_accuracy = nb_classifier.score(X_test, y_test)

    # Return accuracy scores
    return {
        'Random Forest': rf_accuracy,
        'Decision Tree': dt_accuracy,
        'K-Nearest Neighbors': knn_accuracy,
        'AdaBoost': ada_accuracy,
        'SGD': sgd_accuracy,
        'Extra Trees': et_accuracy,
        'Gaussian Naive Bayes': nb_accuracy
    }
