import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def function_z(x, y, _theta):
    return _theta[0] + _theta[1] * x + _theta[2] * y


def draw_plot(_data, _theta, _data_labels, iter, learn_rate, save=False):
    # scatter plot and label assignment
    ax = plt.axes(projection='3d')
    ax.scatter3D(_data[:, 0], _data[:, 1], _data[:, 2], s=25, marker="x")
    ax.set_xlabel(_data_labels[0])
    ax.set_ylabel(_data_labels[1])
    ax.set_zlabel(_data_labels[2])

    # axis formatting
    max_x = np.amax(_data[:, 0])
    max_y = np.amax(_data[:, 1])
    max_z = np.amax(_data[:, 2])

    ax.set_xlim(0, max_x + 2)
    ax.set_ylim(0, max_y + 15)
    ax.set_zlim(0, max_z + 10)

    # range for surface plot
    x_values = np.arange(0, max_x + 2, 0.1)
    y_values = np.arange(0, max_y + 15, 0.1)

    # surface plot
    X, Y = np.meshgrid(x_values, y_values)
    Z = function_z(X, Y, _theta)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='grey')

    labels = []
    labels.append(f"{round(_theta[0], 2)} + {round(_theta[1], 2)}x + {round(_theta[2], 2)}y")
    labels.append(f"Iteration:{iter}")
    labels.append(f"Learn Rate:{learn_rate}")
    labels.append(f"Cost:{round(calc_cost(_data, _theta), 2)}")

    infostr = '\n'.join(labels)
    ax.text(9, 40, max_z + 20, infostr)
    ax.view_init(20, -135)

    if save:
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig = plt.gcf()
        fig.set_size_inches((10, 10), forward=False)
        fig.savefig(f'(L)F3,Iter={iter} a={learn_rate} Cost={round(calc_cost(_data, _theta), 2)}.png', dpi=1000)
    plt.show()


def calc_cost(_data, _theta):
    cost = 0
    k = len(_data)
    for i in range(k):
        x = _data[i, 0]
        y = _data[i, 1]
        z = _data[i, 2]
        z_pred = function_z(x, y, _theta)
        cost += (z - z_pred) ** 2

    cost *= 1 / (2 * k)
    return cost


def calc_step(_data, _theta, learn_rate):
    n = len(_data)
    x = _data[:, 0]
    y = _data[:, 1]
    z = _data[:, 2]

    z_pred = function_z(x, y, _theta)

    der_theta0 = -(1 / n) * sum(z - z_pred)
    der_theta1 = -(1 / n) * sum((z - z_pred) * x)
    der_theta2 = -(1 / n) * sum((z - z_pred) * y)

    _theta[0] -= learn_rate * der_theta0
    _theta[1] -= learn_rate * der_theta1
    _theta[2] -= learn_rate * der_theta2

    return _theta


# run gradient descent for given iter and plot the resulting theta
def gradient_descent(_data, _theta, data_labels, max_iter, learn_rate):

    #build progress bar
    loop = tqdm(range(max_iter), desc="Training", leave=True)

    for i in loop:
        cost = calc_cost(_data, _theta)
        #print(f"Iteration:{i} Theta_0:{_theta[0]} Theta_1:{_theta[1]}  Theta_2:{_theta[2]} Cost:{cost}")
        _theta = calc_step(_data, _theta, learn_rate)
    draw_plot(_data, _theta, data_labels, max_iter, learn_rate)
    return _theta

def main():
    data = np.loadtxt("data.csv", delimiter=",")  # load data from csv to array 'data'
    data_labels = ["Area", "Age", "Rent Price"]

    theta = [-.5, 1, 1.5]
    max_iterations = 10000
    learning_rate = 0.00035

    theta = gradient_descent(data, theta, data_labels, max_iterations, learning_rate)
    print(theta)

if __name__ == '__main__':
    main()

#learning_rate = 0.000361