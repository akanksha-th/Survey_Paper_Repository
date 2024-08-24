import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(self.weights, x_train.T) + self.bias
            
            dw = (1/n_samples) * (-2) * np.dot((y_train - y_pred), x_train)
            db = (1/n_samples) * (-2) * np.sum(y_train - y_pred)
            
            self.weights = self.weights - (self.lr*dw)
            self.bias = self.bias - (self.lr*db)
            
    def predict(self, x_test):
        y_pred = np.dot(x_test, self.weights) + self.bias
        return y_pred
    
    
#-------------------------------------------------------------------------------------------------------------------------#

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=130, n_features=1, random_state=3, noise=20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], y, color='b', marker='o', s=30)
plt.savefig('1.Linear_Regression/data_distribution.png')
plt.show()

model = LinearRegression(lr=0.005)
model.fit(x_train, y_train)
prediction = model.predict(x_test)

def mse(y_test, prediction):
    return np.mean((y_test - prediction)**2)
final_loss = mse(y_test, prediction)

y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='k', linewidth=2, label='predicted_line')
plt.legend()
plt.text(0.05, 0.90, f'Final Loss: {final_loss:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('1.Linear_Regression/regression_plot.png')
plt.show()