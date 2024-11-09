7import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import add_dummy_feature

def batch_gradient_descent_adaptive(X, y, eta=0.1, initial_epochs=100, threshold=1e-6, max_epochs=10000):
    X_b = add_dummy_feature(X)
    m = len(X_b)
    
    np.random.seed(42)
    theta = np.random.randn(X_b.shape[1], 1)
    
    cost_history = []
    theta_path = []
    global_minimum = None
    local_minima = []
    total_epochs = 0
    epochs = initial_epochs
    
    oscillation_count = 0
    max_oscillations = 3
    prev_cost = float('inf')
    cost_increase_threshold = 0.1
    
    while total_epochs < max_epochs:
        converged = False
        for epoch in range(epochs):
            gradients = 2 / m * X_b.T @ (X_b @ theta - y)
            theta = theta - eta * gradients
            theta_path.append(theta.copy())
            
            cost = (1 / m) * np.sum((X_b @ theta - y) ** 2)
            cost_history.append(cost)
            
            if abs(prev_cost - cost) < threshold:
                converged = True
                break
            
            if cost > prev_cost * (1 + cost_increase_threshold):
                oscillation_count += 1
                if oscillation_count >= max_oscillations:
                    converged = True
                    break
            else:
                oscillation_count = 0

            prev_cost = cost
        
        total_epochs += (epoch + 1)
        
        if global_minimum is None or cost < global_minimum['cost']:
            global_minimum = {'epoch': total_epochs, 'cost': cost, 'theta': theta.tolist()}
        else:
            local_minima.append({'epoch': total_epochs, 'cost': cost, 'theta': theta.tolist()})
        
        if converged:
            break
        
        prev_cost = float('inf')
    
    return theta, cost_history, theta_path, global_minimum, local_minima

def plot_gradient_descent_adaptive(X, y, eta):
    theta, cost_history, theta_path, global_minimum, local_minima = batch_gradient_descent_adaptive(X, y, eta)
    
    X_b = add_dummy_feature(X)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(X, y, "b.")
    X_new = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_new_b = add_dummy_feature(X_new)
    n_shown = min(20, len(theta_path))
    for i, theta_i in enumerate(theta_path[:n_shown]):
        y_predict = X_new_b @ theta_i
        color = mpl.colors.rgb2hex(plt.cm.OrRd(i / n_shown + 0.15))
        plt.plot(X_new, y_predict, linestyle="solid", color=color)
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([X.min(), X.max(), y.min(), y.max()])
    plt.grid()
    plt.title(fr"$\eta = {eta}$")
    
    plt.subplot(132)
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.grid()
    
    plt.subplot(133)
    plt.plot(X, y, "b.")
    y_predict = X_new_b @ theta
    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([X.min(), X.max(), y.min(), y.max()])
    plt.grid()
    plt.title("Final Model")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return theta, global_minimum, local_minima

# Usage example
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

theta, global_minimum, local_minima = plot_gradient_descent_adaptive(X, y, eta=0.1)