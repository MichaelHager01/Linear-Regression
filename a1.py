import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



'''
display_results_table

A specialized display utility for MEEN 423 A1. 
Creates a table for visualizing the training and test loss statistics
for three different fits.

Takes three arguments, each of which is a two-element tuple. Assumes the 
first element of the tuple is for training MSE and the second element is for 
test MSE. 
'''

def display_results_table(linear, quadratic, cubic):
    print('\n\t\t\tTrain MSE\tTest MSE')
    print('Linear    \t%6.4f\t\t%6.4f' % linear)
    print('Quadratic \t%6.4f\t\t%6.4f' % quadratic)
    print('Cubic     \t%6.4f\t\t%6.4f' % cubic)
  


# Read data from csv
raw_data = pd.read_csv("a1_data.csv")


# Separate into features (inputs) and response (output)
features = raw_data[['x1','x2']].to_numpy()
response = raw_data['y'].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(features, response, test_size=0.2, random_state=2022)


# Perform the linear regression
reg = PolynomialFeatures(degree=1, include_bias=False)
reg.fit(X_train)
reg_features = reg.transform(X_train)
reg_reg = LinearRegression()
reg_reg.fit(reg_features,Y_train)


# Visualize the sample data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], Y_train)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')


# Visualize the fitted response
x_min = X_train.min(0)
x_max = X_train.max(0)
N = 10
X0 = np.linspace(x_min[0], x_max[0], N)
X1 = np.linspace(x_min[1], x_max[1], N)
X0, X1 = np.meshgrid(X0, X1)
PL = np.zeros((N,N), dtype='float')
for i in range(N):
    for j in range(N):
        # Notice we are transforming the data before calling predict
        x = reg.transform(np.array( [[X0[i,j], X1[i,j]]]))
        PL[i,j] = reg_reg.predict(x)

ax.plot_wireframe(X0,X1,PL,color='orange')


# Perform quadratic regression
quad = PolynomialFeatures(degree=2, include_bias=False)
quad.fit(X_train)
quad_features = quad.transform(X_train)
quad_reg = LinearRegression()
quad_reg.fit(quad_features,Y_train)


# Visualize the sample data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], Y_train)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')


# Visualize the fitted response
x_min = X_train.min(0)
x_max = X_train.max(0)
N = 10
X0 = np.linspace(x_min[0], x_max[0], N)
X1 = np.linspace(x_min[1], x_max[1], N)
X0, X1 = np.meshgrid(X0, X1)
PL = np.zeros((N,N), dtype='float')
for i in range(N):
    for j in range(N):
        # Notice we are transforming the data before calling predict
        x = quad.transform(np.array( [[X0[i,j], X1[i,j]]]))
        PL[i,j] = quad_reg.predict(x)

ax.plot_wireframe(X0,X1,PL,color='gray')


# Perform cubic regression
cubic = PolynomialFeatures(degree=3, include_bias=False)
cubic.fit(X_train)
cubic_features = cubic.transform(X_train)
cubic_reg = LinearRegression()
cubic_reg.fit(cubic_features,Y_train)


# Visualize the sample data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], Y_train)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')


# Visualize the fitted response
x_min = X_train.min(0)
x_max = X_train.max(0)
N = 10
X0 = np.linspace(x_min[0], x_max[0], N)
X1 = np.linspace(x_min[1], x_max[1], N)
X0, X1 = np.meshgrid(X0, X1)
PL = np.zeros((N,N), dtype='float')
for i in range(N):
    for j in range(N):
        # Notice we are transforming the data before calling predict
        x = cubic.transform(np.array( [[X0[i,j], X1[i,j]]]))
        PL[i,j] = cubic_reg.predict(x)

ax.plot_wireframe(X0,X1,PL,color='red')


# Calculate MSE for each model
y_train_pred_linear = reg_reg.predict(reg.transform(X_train))
y_test_pred_linear = reg_reg.predict(reg.transform(X_test))
y_train_pred_quad = quad_reg.predict(quad.transform(X_train))
y_test_pred_quad = quad_reg.predict(quad.transform(X_test))
y_train_pred_cubic = cubic_reg.predict(cubic.transform(X_train))
y_test_pred_cubic = cubic_reg.predict(cubic.transform(X_test))

mse_train_linear = mean_squared_error(Y_train, y_train_pred_linear)
mse_test_linear = mean_squared_error(Y_test, y_test_pred_linear)
mse_train_quad = mean_squared_error(Y_train, y_train_pred_quad)
mse_test_quad = mean_squared_error(Y_test, y_test_pred_quad)
mse_train_cubic = mean_squared_error(Y_train, y_train_pred_cubic)
mse_test_cubic = mean_squared_error(Y_test, y_test_pred_cubic)

display_results_table((mse_train_linear, mse_test_linear), (mse_train_quad, mse_test_quad), (mse_train_cubic, mse_test_cubic))


