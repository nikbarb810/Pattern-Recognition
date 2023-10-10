import numpy as np
import matplotlib.pyplot as plt


def draw_plot(data_x, data_y, x_label, _theta0, _theta1, iteration, learn_rate, isSaved =False):
    # scatter plot
    plt.scatter(data_x, data_y, s=25, marker="x")
    plt.xlabel(x_label)
    plt.ylabel("Rent Price")

    # format x and y axis
    max_x = np.amax(data_x)
    max_y = np.amax(data_y)
    plt.axis([0, max_x + 2, 0, max_y + 20])

    # line plot
    x = np.arange(0, int(max_x) + 10, 0.1)
    y = _theta0 + _theta1 * x
    plt.plot(x, y, c='red')

    # create label for information
    line_label = f"{round(_theta0, 5)} + {round(_theta1, 5)}x"
    labels = []
    labels.append(line_label)
    labels.append(f"Iteration:{iteration}")
    labels.append(f"Learn Rate:{learn_rate}")

    cost = calc_cost(data_x, data_y, _theta0, _theta1)
    labels.append(f"Cost:{round(cost, 5)}")

    infostr = '\n'.join(labels)
    plt.text(0.5, max_y + 10, infostr)

    if isSaved:
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig = plt.gcf()
        fig.set_size_inches((10, 10), forward=False)
        fig.savefig(
            f'F2,Iter={iteration} a={learn_rate} Cost={round(calc_cost(data_x, data_y, _theta0, _theta1), 2)}.png',
            dpi=1000)

    plt.show()


def calc_cost(data_x, data_y, _theta0, _theta1):
    cost = 0
    k = len(data_x)
    for i in range(k):
        x = data_x[i]
        y = data_y[i]
        y_pred = _theta0 + _theta1 * x
        cost += (y - y_pred) ** 2

    cost *= 1 / (2 * k)
    return cost


def calc_step(data_x, data_y, _theta0, _theta1, learn_rate):
    n = len(data_x)
    y_pred = _theta0 + _theta1 * data_x

    der_theta0 = -(1 / n) * sum(data_y - y_pred)
    der_theta1 = -(1 / n) * sum((data_y - y_pred) * data_x)

    _theta0 -= learn_rate * der_theta0
    _theta1 -= learn_rate * der_theta1

    return [_theta0, _theta1]


def grad_descent(data_x, x_label, data_y, _theta0, _theta1, max_iterations, learn_rate):
    for i in range(max_iterations):
        cost = calc_cost(data_x, data_y, _theta0, _theta1)
        print(f"Iteration:{i} Theta_0:{_theta0} Theta_1:{_theta1} Cost:{cost}")
        _theta0, _theta1 = calc_step(data_x, data_y, _theta0, _theta1, learn_rate)

    draw_plot(data_x, data_y, x_label, _theta0, _theta1, max_iterations, learn_rate)


def main():
    data = np.loadtxt("data.csv", delimiter=",")  # load data from csv to array 'data'

    data_area = data[:, 0]
    data_age = data[:, 1]
    data_price = data[:, 2]

    max_iteration = 100
    learning_rate = 0.047
    theta_0 = -0.5
    theta_1 = 1.5

    grad_descent(data_area, "Area", data_price, theta_0, theta_1, max_iteration, learning_rate)

    theta_0 = -1.5
    theta_1 = 1
    learning_rate = 0.00035
    max_iteration = 1

    grad_descent(data_age, "Age", data_price, theta_0, theta_1, max_iteration, learning_rate)


main()
