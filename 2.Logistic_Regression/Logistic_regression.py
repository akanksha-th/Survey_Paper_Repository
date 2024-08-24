import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    
    def __init__(self, lr=0.001, n_iters=4500):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            pred = np.dot(x_train, self.weights) + self.bias
            y_pred = sigmoid(pred)
            
            dw = (1/n_samples) * np.dot(x_train.T, (y_pred - y_train))
            db = (1/n_samples) * np.sum(y_pred - y_train)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, x_test):
        pred = np.dot(x_test, self.weights) + self.bias
        y_pred = sigmoid(pred)  # contains probability
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    
#--------------------------------------------------------------------------------------------------------------------#

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = datasets.load_breast_cancer()
X, y = df.data, df.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

'''print(X.shape, y.shape)
fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', s=30)
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title('Breast Cancer Dataset')
plt.show()'''

clf = LogisticRegression(lr = 0.005)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)

def accuracy(prediction, y_test):
    return np.sum(prediction==y_test)/len(y_test)

acc = accuracy(prediction, y_test)
print(f"Accuracy of this model is: {acc}")