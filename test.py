from decimal import Decimal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

num=Decimal("0.1")+Decimal("0.3")
print(type(num),float(num))
def social_network_analysis():
    dataset_path = input("Enter the dataset path: ")
    
    try:
        # Load the dataset
        dataset = pd.read_csv(dataset_path)
        
        # Extract features and labels
        X = dataset.iloc[:,2:4]
        y = dataset.iloc[:, 4]
        
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        
        # Standardize the features
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        # Fit a Gaussian Naive Bayes classifier
        clf1 = GaussianNB()
        clf1.fit(X_train, y_train)
        
        # Plot the decision boundary
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, clf1.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        color=ListedColormap(('red', 'green'))(i), label=j)
        
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        plt.title('Naive Bayes Classifier')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
    
    except Exception as e:
        print("An error occurred:", str(e))
social_network_analysis()