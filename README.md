# Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the given dataset.
2. Fitting the dataset into the training set and test set.
3. Applying the feature scaling method.
4. Fitting the logistic regression into the training set.
5. Prediction of the test and result
6. Making the confusion matrix
7. Visualizing the training set results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANJAI A
RegisterNumber: 212220040142 
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data = np.loadtxt("ex2data1.txt",delimiter=',')
X = data[:,[0,1]]
y = data[:,2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFunction(theta,X,Y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T,h-y) / X.shape[0]
  return J,grad
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J, grad  = costFunction(theta,X_train,y)
print(J)
print(grad)
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
J, grad  = costFunction(theta,X_train,y)
print(J)
print(grad)
def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min ,x_max = X[:,0].min()-1,X[:,0].max()+1
  y_min ,y_max = X[:,1].min()-1,X[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                      np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
plotDecisionBoundary(res.x,X,y)
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
![image](https://user-images.githubusercontent.com/95969295/204102563-24b196d9-bd9a-4edb-8c6a-17ee02ce0d62.png)

![image](https://user-images.githubusercontent.com/95969295/204102575-9bde1e87-cf8d-4082-a296-774a1cd071cd.png)

![image](https://user-images.githubusercontent.com/95969295/204102588-8bdea145-b91a-4f40-a7dc-c7d7fecc3af0.png)

![image](https://user-images.githubusercontent.com/95969295/204102602-cb7de79b-290c-4f2b-a384-5afcdc239b5a.png)

![image](https://user-images.githubusercontent.com/95969295/204102626-089d7c8d-45e0-4114-b27b-514f513a1816.png)

![image](https://user-images.githubusercontent.com/95969295/204102644-8f535f3d-cc5b-4aec-b69e-8165434860f3.png)

![image](https://user-images.githubusercontent.com/95969295/204102656-6fb7e94d-407d-408f-bba6-9627129aa3c5.png)

![image](https://user-images.githubusercontent.com/95969295/204102689-978f4f73-bd5e-42a4-ba60-39945715e6d5.png)

![image](https://user-images.githubusercontent.com/95969295/204102693-e2c4d9a1-2936-442e-9429-5e9ab83becc8.png)

![image](https://user-images.githubusercontent.com/95969295/204102698-d00bc66f-c7c1-4d49-a246-45508620c944.png)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

