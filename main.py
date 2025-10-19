import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('data.csv')

def mse(m,b,points):
    total_err=0
    for i in range(len(points)):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores
        total_err+= (y-(m*x + b))**2
    return total_err/float(len(points))

def gradient_descent(m0,b0,points,learning_rate):
    m_gradient=0
    b_gradient=0

    n=len(points)
    
    for i in range(n):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores

        m_gradient+= -(2/n)*x*(y-(m0*x+b0))
        b_gradient+= -(2/n)*(y-(m0*x+b0))

    m=m0-m_gradient*learning_rate
    b=b0-b_gradient*learning_rate
    return m,b

def train(points, learning_rate, epochs):
    m = 0
    b = 0
    errors = []

    for i in range(epochs):
        m, b = gradient_descent(m, b, points, learning_rate)
        error = mse(m, b, points)
        errors.append(error)
        if i % 100 == 0:
            print(f"Epoch {i}: m={m:.4f}, b={b:.4f}, mse={error:.4f}")

    return m, b, errors
learning_rate = 0.01
epochs = 1000

m, b, errors = train(data, learning_rate, epochs)
print(f"Final parameters: m = {m:.4f}, b = {b:.4f}")
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Loss vs Epochs")
plt.savefig("loss_vs_epochs.png")
plt.close()


plt.scatter(data.Hours, data.Scores, color='blue', label='Actual Data')
plt.plot(data.Hours, m * data.Hours + b, color='red', label='Best Fit Line')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Linear Regression Fit")
plt.legend()
plt.savefig("best_fit_line.png")
plt.close()