import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


class Scheduler:
    def __init__(self, lr=0.1, factor=0.99, patience=1000):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.counter = 0

    def get_next_lr(self):
        self.counter += 1
        if self.counter >= self.patience:
            self.lr *= self.factor
            self.counter = 0

        return self.lr


def calc_step(data_x, data_y, theta, learn_rate):
    n = len(data_x)

    y_pred = np.dot(data_x, theta)

    # derivatives = -(1 / n) * np.dot((data_y - y_pred), data_x)

    derivatives = np.zeros(len(theta))
    derivatives[0] = -(1 / n) * np.sum(data_y - y_pred)
    derivatives[1] = -(1 / n) * np.sum((data_y - y_pred) * data_x[:, 1])
    derivatives[2] = -(1 / n) * np.sum((data_y - y_pred) * data_x[:, 2])

    # calculate the update
    upd = learn_rate * derivatives

    theta -= upd

    return theta


# df: dataframe
# a: initial weight vector
# lr: learning rate
# n_iter: number of iterations
# display_progress: whether to display progress bar or not
def calc_cost(data_x, data_y, a):
    n = len(data_x)

    y_pred = np.dot(data_x, a)

    cost = (1 / n) * np.sum(np.square(data_y - y_pred))

    return cost


def train_grad_desc(df, a, lr, n_iter, variable_lr=False, display_progress=True, his_saving_interval=1000):
    acc_history = []

    # build the progress bar
    if display_progress:
        loop = tqdm(
            range(n_iter),
            desc='Training',
            leave=True
        )
    else:
        loop = range(n_iter)

    # initialize the scheduler
    if variable_lr:
        scheduler = Scheduler(lr=lr)

    # split the dataframe into x and y
    data_x = df.iloc[:, :-1]
    data_y = df.iloc[:, -1]

    data_y = data_y.to_numpy()

    data_x = np.c_[np.ones(len(data_x)), data_x]

    for i in loop:

        # calculate the mean squared error
        mse = calc_cost(data_x, data_y, a)

        # update the weight vector
        a = calc_step(data_x, data_y, a, lr)

        # update the learning rate
        if variable_lr:
            lr = scheduler.get_next_lr()

        # save model every his_saving_interval iterations
        if i % his_saving_interval == 0:
            # calculate total errors
            acc_history.append((mse, copy.deepcopy(a)))

        # print(f"Iteration:{i}   mse:{mse}   a:{a}")

    return a, acc_history


# def plotting function
def plot(df, min_a):
    # create a 3d scatter plot where x1,x2 are the features and y is the result
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, -1], c='r', marker='o')

    ax.set_xlabel('Area')
    ax.set_ylabel('Age')
    ax.set_zlabel('Price')

    # find the min,max values of x1,x2
    x1_min, x1_max = df.iloc[:, 0].min(), df.iloc[:, 0].max()
    x2_min, x2_max = df.iloc[:, 1].min(), df.iloc[:, 1].max()

    # create a meshgrid of x1,x2
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

    # calculate the predicted y values
    y_pred = min_a[0] + min_a[1] * x1 + min_a[2] * x2

    # plot the predicted y values
    ax.plot_surface(x1, x2, y_pred, alpha=0.5)

    plt.show()


# create a function that will call gradient descent algorithm
# and save best results
def grad_desc(df, a, lr, n_iter, his_interval_size=1000, variable_lr=False, show_info=True):
    a, acc_history = train_grad_desc(df, a, lr, n_iter, his_saving_interval=his_interval_size, variable_lr=variable_lr)

    # find the model with the smallest mse
    min_index = np.argmin([x[0] for x in acc_history])

    # get the a vector of the model with the smallest mse
    min_a = acc_history[min_index][1]

    if show_info:
        print(f"Min MSE: {acc_history[min_index][0]}")
        print(f"Min a: {min_a}")

    return acc_history, min_index


def prepare_animation(ax_left, ax_right, left_surface, mse_line, meshgrid, acc_history, camera_step):
    def animate(frame_number):
        if frame_number < len(acc_history):
            # get the a vector of the current frame
            a = acc_history[frame_number][1]

            # update/create surface
            nonlocal left_surface

            # clear prev surface
            if left_surface is not None:
                left_surface.remove()

            # calculate new surf
            y_pred = a[0] + a[1] * meshgrid[0] + a[2] * meshgrid[1]

            # plot new surf
            left_surface = ax_left.plot_surface(meshgrid[0], meshgrid[1], y_pred, alpha=0.8, color='b')

            # update/create mse line
            nonlocal mse_line

            # clear prev mse line
            if mse_line is not None:
                mse_line.pop(0).remove()

            # get the mse values
            mse_values = [entry[0] for entry in acc_history[:frame_number + 1]]

            # plot the mse
            mse_line = ax_right.plot(range(len(mse_values)), mse_values, color='r')

        else:
            # update camera step(azim)
            ax_left.view_init(elev=15, azim=-45 + ((frame_number - len(acc_history)) * camera_step), roll=0)

        return plot

    return animate


def animate_grad_desc(df, acc_history, save=False, regr_dur_s=5, rotation_dur_s=5, saving_fps=22):
    # create a 1x2 3d figure
    fig = plt.figure(figsize=(12, 6), layout="constrained")

    gs = GridSpec(7, 15, figure=fig)
    ax0 = fig.add_subplot(gs[1:6, 1:8], projection='3d')
    ax1 = fig.add_subplot(gs[2:5, 9:14])

    # plot the data
    ax0.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, -1], c='r', marker='x')

    ax0.view_init(elev=15, azim=-45)

    # titles
    ax0.set_title('Linear Regression', y=0.95)
    ax1.set_title('MSE')

    # set ax1 x limits
    ax1.set_xlim(0, len(acc_history))

    # find min and max values of the dataset
    x_min, x_max = ax0.get_xlim()
    y_min, y_max = ax0.get_ylim()
    z_min, z_max = ax0.get_zlim()

    # keep the axis constant
    ax0.set_xlim3d([x_min, x_max])
    ax0.set_ylim3d([y_min, y_max])
    ax0.set_zlim3d([z_min, z_max])

    # add labels
    ax0.set_xlabel('Area')
    ax0.set_ylabel('Age')
    ax0.set_zlabel('Price')

    frames_for_rotation = rotation_dur_s * saving_fps

    # calculate the camera step size to complete 2 full rotations in rotation_dur_s frames
    camera_step = 360 / frames_for_rotation

    # calc meshgrid
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # init items that will be drawn
    surf = None
    mse_line = None

    # create the animation
    anim = animation.FuncAnimation(fig,
                                   prepare_animation(ax0, ax1, surf, mse_line, meshgrid=[x1, x2],
                                                     acc_history=acc_history,
                                                     camera_step=camera_step),
                                   frames=len(acc_history) + frames_for_rotation, interval=1, repeat=True)

    if save:
        anim.save(f"animation_{len(acc_history)}iter_{regr_dur_s+rotation_dur_s}s_{saving_fps}fps.gif", writer='imagemagick',
                  fps=saving_fps)

    plt.show()


def smooth_acc_history(acc_history, window_size=40, num_initial_frames=10, show_info=True):
    # Extract the MSE values from acc_history
    mse_values = [entry[0] for entry in acc_history]

    # Keep the first #num_inital_frames elements always
    # Skip the first element because it is the initial MSE
    reduced_acc_history = acc_history[1:num_initial_frames + 1]

    # Calculate the reduced acc_history for the remaining frames
    for i in range(num_initial_frames, len(acc_history), window_size):
        reduced_acc_history.append((mse_values[i], acc_history[i][1]))

    if show_info:
        print(f"Reduced acc history size from:{len(acc_history)} to {len(reduced_acc_history)}")

    return reduced_acc_history


def new_animate_3d(df, a_vectors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the data
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, -1], c='r', marker='x')

    # find min and max values of the dataset
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    # Make the X,Y meshgrid.
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # set the z axis limits
    ax.set_zlim3d(z_min, z_max)

    a_plane = None
    for i in range(0, len(a_vectors)):
        if a_plane:
            a_plane.remove()

        a = a_vectors[i]
        y_pred = a[0] + a[1] * x1 + a[2] * x2
        a_plane = ax.plot_surface(x1, x2, y_pred, alpha=0.5)
        plt.pause(0.001)


def main():
    df = pd.read_csv('data.csv', header=None)

    # model parameters
    a = np.random.rand(3)
    n_iter = 2000000
    lr = 0.00035
    his_interval_size = 1000

    # animation parameters
    saving_fps = 40
    duration_regression_s = 7
    rotation_dur_s = 5
    num_initial_frames = 0
    skip_first_n = 1
    save = True

    # run the gradient descent
    acc_history, best_index = grad_desc(df, a, lr, n_iter, his_interval_size=his_interval_size, variable_lr=False)

    total_frames_needed = saving_fps * duration_regression_s

    #remove first n frames
    acc_history = acc_history[skip_first_n:]

    # calculate the window size for the acc history
    window_size = int(len(acc_history) / total_frames_needed)

    # sample the acc history
    reduced_acc_history = smooth_acc_history(acc_history, window_size=window_size,
                                             num_initial_frames=num_initial_frames)
    # animate the regression
    animate_grad_desc(df, reduced_acc_history, save=save, regr_dur_s=duration_regression_s,
                        rotation_dur_s=rotation_dur_s, saving_fps=saving_fps)


if __name__ == "__main__":
    main()
