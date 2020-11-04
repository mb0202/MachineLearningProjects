import numpy as np
import matplotlib.pyplot as plt
import random

def rsquared(X,Y,m,b):
    ''' R squared loss function '''
    result = 0
    for (x, y) in zip(X,Y):
        result += (y-(m*x+b))**2

    return result/(2*X.size)


# Create initial data 
X = np.arange(0, 7, 1)
Y = []
random_m = random.random()*random.randint(-10, 10)
for i in X:
    Y.append(random_m * i + 4*random.random())

plt.scatter(X, Y)
plt.show()


# Plot the initial guess for the line of best fit and calculate initial loss
m = random.random()
b = random.random()
x_hat = np.arange(0, 10, 1)
y_hat = m*x_hat + b
print("Loss:",rsquared(X,Y, m, b))

plt.scatter(X, Y)
plt.plot(x_hat, y_hat, color='r')
plt.show()


# Gradient descent
epochs = 25000
learning_rate = 0.0001

for epoch in range(epochs+1):

    # Gradient calculation using derivative of R squared 
    gradient_m = 0
    for (x, y) in zip(X, Y):
        gradient_m += (-x/X.size) * (y-m*x-b)

    gradient_b = 0
    for (x,y) in zip(X, Y):
        gradient_b += (-1/X.size)*( y-m*x-b)\

    # Update parameters
    m_new = m - learning_rate * gradient_m
    b_new = b - learning_rate * gradient_b
    m = m_new
    b = b_new

    if epoch % 100 == 0:
        print("Loss at epoch {}: {}".format(epoch+1, rsquared(X, Y, m, b)))


# Show final line and loss
x_hat = np.arange(0, 10, 1)
y_hat = m*x_hat + b

print("Loss:",rsquared(X,Y, m, b))

plt.scatter(X, Y)
plt.plot(x_hat, y_hat, color='r')
plt.show()
