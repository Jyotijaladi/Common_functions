import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering

def data_exploration():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Print column types
        print("Column Types:")
        print(data.dtypes)

        # Print column names
        print("Column Names:")
        print(data.columns)

        # Print head
        print("Head:")
        print(data.head())

        # Print tail
        print("Tail:")
        print(data.tail())

        # Print mean
        print("Mean:")
        print(data.mean())

        # Print standard deviation
        print("Standard Deviation:")
        print(data.std())

    except Exception as e:
        print("An error occurred:", str(e))


def data_visualization():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Pairplot
        sns.pairplot(data)
        plt.show()

        # Distribution plot
        for column in data.columns:
            if data[column].dtype == 'float64':
                sns.displot(data[column])
                plt.title(column)
                plt.show()

        # Box plot
        sns.boxplot(data=data)
        plt.show()

        # Scatter plot
        for column1 in data.columns:
            if data[column1].dtype == 'float64':
                for column2 in data.columns:
                    if data[column2].dtype == 'float64' and column1 != column2:
                        sns.scatterplot(data=data, x=column1, y=column2)
                        plt.xlabel(column1)
                        plt.ylabel(column2)
                        plt.show()

    except Exception as e:
        print("An error occurred:", str(e))

def data_preprocessing():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Handling missing values
        data.fillna(0, inplace=True)

        # Print pre-processed data
        print("Pre-processed Data:")
        print(data)

    except Exception as e:
        print("An error occurred:", str(e))


def normalization():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform min-max normalization
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)

        # Print normalized data
        print("Normalized Data:")
        print(normalized_data)

    except Exception as e:
        print("An error occurred:", str(e))

def standardization():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform standardization
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)

        # Print standardized data
        print("Standardized Data:")
        print(standardized_data)

    except Exception as e:
        print("An error occurred:", str(e))


def data_reduction():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)

        # Print reduced data
        print("Reduced Data:")
        print(reduced_data)

    except Exception as e:
        print("An error occurred:", str(e))



def binary_logistic_regression():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    target_column = input("Enter the name of the target column: ")

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform binary logistic regression
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        model = LogisticRegression()
        model.fit(X, y)

        # Perform predictions
        # ...
        return model
    except Exception as e:
        print("An error occurred:", str(e))


def decision_tree_classification():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    target_column = input("Enter the name of the target column: ")

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform decision tree classification
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        model = DecisionTreeClassifier()
        model.fit(X, y)

        # Perform predictions
        # ...
        return model
    except Exception as e:
        print("An error occurred:", str(e))

def naive_bayes_classification():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    target_column = input("Enter the name of the target column: ")

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform Naive Bayes classification
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        model = GaussianNB()
        model.fit(X, y)

        # Perform predictions
        # ...
        return model
    except Exception as e:
        print("An error occurred:", str(e))

def knn_classification():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    target_column = input("Enter the name of the target column: ")

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform KNN classification
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        model = KNeighborsClassifier()
        model.fit(X, y)

        # Perform predictions
        # ...
        return model
    except Exception as e:
        print("An error occurred:", str(e))




def frequent_item_set_mining():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    min_support = float(input("Enter the minimum support: "))
    min_threshold = float(input("Enter the minimum threshold: "))

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform frequent item set mining
        frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_threshold)

        # Print frequent itemsets and association rules
        print("Frequent Itemsets:")
        print(frequent_itemsets)
        print("Association Rules:")
        print(rules)

    except Exception as e:
        print("An error occurred:", str(e))




def linear_regression():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    target_column = input("Enter the name of the target column: ")

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform linear regression
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        model = LinearRegression()
        model.fit(X, y)

        # Perform predictions
        # ...

    except Exception as e:
        print("An error occurred:", str(e))





def kmeans_clustering():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    num_clusters = int(input("Enter the number of clusters: "))

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(data)

        # Print cluster assignments
        print("Cluster Assignments:")
        print(clusters)

        # Visualize clusters
        plt.scatter(data['x'], data['y'], c=clusters)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('K-means Clustering')
        plt.show()

    except Exception as e:
        print("An error occurred:", str(e))


def hierarchical_clustering():
    dataset_path = input("Enter the dataset path: ")

    if not os.path.exists(dataset_path):
        print("File not found. Please provide a valid dataset path.")
        return

    num_clusters = int(input("Enter the number of clusters: "))

    try:
        # Load data
        data = pd.read_csv(dataset_path)

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        clusters = clustering.fit_predict(data)

        # Print cluster assignments
        print("Cluster Assignments:")
        print(clusters)

        # Visualize clusters
        plt.scatter(data['x'], data['y'], c=clusters)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Hierarchical Clustering')
        plt.show()

    except Exception as e:
        print("An error occurred:", str(e))


import networkx as nx
import os

def social_network_analysis():
    graph_file = input("Enter the file location of the graph: ")

    if not os.path.exists(graph_file):
        print("File not found. Please provide a valid graph file location.")
        return

    try:
        # Load the social network graph
        graph = nx.read_edgelist(graph_file)

        # Print basic network properties
        print("Number of nodes:", graph.number_of_nodes())
        print("Number of edges:", graph.number_of_edges())
        print("Average clustering coefficient:", nx.average_clustering(graph))
        print("Average shortest path length:", nx.average_shortest_path_length(graph))

        # Perform community detection
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        print("Communities:", communities)

        # Visualize the graph
        nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title('Social Network Graph')
        plt.show()

    except Exception as e:
        print("An error occurred:", str(e))


