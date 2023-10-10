import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from tqdm import tqdm


class Scheduler:
    def __init__(self, lr=0.1, factor=0.8, patience=10):
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


def prepare_animation(ax_line, ax_error, x_axis, df_0, df_1, acc_list, best_index):
    def animate(frame_number):
        ax_line.clear()

        # in ax_line, scatter plot df_0 and df_1 with different symbols
        ax_line.scatter(df_0[:, 0], df_0[:, 1], s=10, c='b', marker="x", label='0')
        ax_line.scatter(df_1[:, 0], df_1[:, 1], s=10, c='r', marker="o", label='1')

        # also plot the current line
        a = acc_list[frame_number][1]

        # find min and max values of the dataset
        x_min, x_max = ax_line.get_xlim()
        y_min, y_max = ax_line.get_ylim()

        # keep the axis constant
        ax_line.axis([x_min, x_max, y_min, y_max])

        # add the curr frame number
        ax_line.text(0.05, 0.95, f'iter = {frame_number * his_saving_interval}', transform=ax_line.transAxes)

        # add title
        ax_line.set_title('Fixed Increment Single Sample Perceptron')

        # plot the line defined by the weight vector
        x = np.linspace(x_min, x_max, 100)
        y = (-a[1] / a[2]) * x - (a[0] / a[2])
        plot = ax_line.plot(x, y, '-g', label=f"{a[1]:.3f}x + {a[2]:.3f}y + {a[0]:.3f} = 0")

        ax_error.clear()

        # in ax_error, plot the error rate
        portion = acc_list[:frame_number]

        # set the x-axis
        x_axis = np.arange(0, len(portion))

        # add the curr error count
        ax_error.text(0.05, 0.95, f'errors:{acc_list[frame_number][0]}', transform=ax_error.transAxes)

        # plot the 0-frame_number portion of the error rate
        ax_error.plot(x_axis, [x[0] for x in portion], '-b', label='error count')

        # add title
        ax_error.set_title('Animation of Error-Rate')

        if frame_number >= best_index:
            # plot the best line in ax_line dotted
            a = acc_list[best_index][1]
            y = (-a[1] / a[2]) * x - (a[0] / a[2])
            ax_line.plot(x, y, '--g', label=f"{a[1]:.3f}x + {a[2]:.3f}y + {a[0]:.3f} = 0")

            ax_line.text(0.70, 0.95, f"best acc:{1 - (acc_list[best_index][0] / (len(df_0) + len(df_1)))}",
                         transform=ax_line.transAxes, color='r')

            # draw a vertical line at the best error rate
            ax_error.axvline(x=best_index, color='r', linestyle='--', label='best error rate')
            ax_error.text(0.70, 0.95, f'min_errors:{acc_list[best_index][0]}', transform=ax_error.transAxes, color='r')

        return plot

    return animate


def format_dataset(df):
    # split dataset based on the labels column
    df_0 = df[df['labels'] == 0]
    df_1 = df[df['labels'] == 1]

    # drop the labels column
    df_0 = df_0.drop(['labels'], axis=1)
    df_1 = df_1.drop(['labels'], axis=1)

    # drop the id column
    df_0 = df_0.drop(['id'], axis=1)
    df_1 = df_1.drop(['id'], axis=1)

    return np.asarray(df_0), np.asarray(df_1)


def format_y(df_0, df_1):
    # add the bias trick to df_0 and df_1
    df_0 = np.c_[np.ones(len(df_0)), df_0]
    df_1 = np.c_[np.ones(len(df_1)), df_1]

    # normalise the df_1, flip the sign
    df_1 = df_1 * -1

    # concatenate the two classes
    df = np.concatenate((df_0, df_1), axis=0)

    return df


# scatter plot of the dataset
def scatter_plot(ax, df_0, df_1, a=None):
    # plot the dataset
    ax.scatter(df_0[:, 0], df_0[:, 1], s=10, c='b', marker="x", label='0')
    ax.scatter(df_1[:, 0], df_1[:, 1], s=10, c='r', marker="o", label='1')

    # find min and max values of the dataset
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # plot the line defined by the weight vector
    if a is not None:
        x = np.linspace(x_min, x_max, 100)
        y = (-a[1] / a[2]) * x - (a[0] / a[2])
        ax.plot(x, y, '-g', label=f"{a[1]:.3f}x + {a[2]:.3f}y + {a[0]:.3f} = 0")

    # set the labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # set the title
    ax.set_title('Scatter plot of the dataset')

    # show the legend
    ax.legend(loc='upper left')

    return ax


def animate_perceptron(df_0, df_1, best_model, his, save=False):
    # create a 1x3 figure
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # find min and max values of the dataset
    scatter_plot(ax[0], df_0, df_1, best_model['a'])

    # find min,max of dataset
    x_min, x_max = ax[0].get_xlim()

    # set the x-axis based on min,max values
    x_axis = np.linspace(x_min, x_max, len(his))

    anim = animation.FuncAnimation(fig, prepare_animation(ax[1], ax[2], x_axis, df_0, df_1, his, best_model['index']),
                                   frames=len(his), interval=100)

    if save:
        anim.save('var_lr_single_perceptron.gif', writer='imagemagick', fps=8)

    plt.show()


# function implements the single sample perceptron algorithm
# a: weights and bias vector
# y: data samples and 1's (bias trick)
# labels: labels of the data sample
# n_iterations: number of iterations
# lr: learning rate
# variable_lr: if True, learning rate will be decreased by 1/iteration

# Ouput 1: a: trained weights and bias vector
# Output 2: acc_history: list of accuracy values every n_samples iterations
# Output 3: a: best trained weight and bias vector
def train_single_sample(a, y, n_iterations, lr=0.1, variable_lr=False):
    k = 0
    acc_history = []

    # tqdm is used to show a progress bar
    loop = tqdm(
        range(n_iterations),
        desc='Training',
        leave=True
    )

    if variable_lr:
        scheduler = Scheduler()

    for i in loop:  # for each iteration

        k = (k + 1) % len(y)  # get the next pseudo random number, k follows a deterministic sequence
        if np.transpose(a) @ y[k] < 0:  # if the prediction is wrong
            a = a + lr * y[k]  # update the weights and bias vector

            # update the learning rate
            if variable_lr:
                lr = scheduler.get_next_lr()

        # check total error every n_samples iterations
        if i % his_saving_interval == 0:
            var = np.apply_along_axis(lambda x: np.transpose(a) @ x, 1, y)
            # append #count of misclassified samples and the current a vector
            acc_history.append((len(var[var < 0]), a))

    return a, acc_history


# create a perceptron runner
def perceptron(a, df, n_iterations, lr=0.1, variable_lr=False):
    df_0, df_1 = format_dataset(df)

    # format the dataset
    _df = format_y(df_0, df_1)

    # train the model
    a, his = train_single_sample(a, _df, n_iterations, lr, variable_lr)
    # get the index of the minimum error
    min_index = np.argmin([x[0] for x in his])
    # get the a vector with the minimum error
    min_a = his[min_index][1]

    best_model = {'errors': his[min_index][0], 'a': min_a, 'index': min_index}

    print(f"most acc model had {his[min_index][0]} errors, a:{min_a} at iteration:{min_index * his_saving_interval}")

    # calculate the accuracy of the model
    var = np.apply_along_axis(lambda x: np.transpose(min_a) @ x, 1, _df)
    acc = len(var[var > 0]) / len(var)

    print(f"accuracy of the model: {acc}")

    return a, his, best_model


his_saving_interval = 100


def main():
    # import the dataset
    df = pd.read_csv('dataset.csv')
    df_0, df_1 = format_dataset(df)

    # define the model's variables
    a = np.random.rand(3)
    n_iter = 20000
    lr = 1

    # train the model
    #fixed_a, fixed_his, fixed_best_model = perceptron(a, df, 100000, lr, variable_lr=False)
    #animate_perceptron(df_0, df_1, fixed_best_model, fixed_his, save=True)


    var_a, var_his, var_best_model = perceptron(a, df, n_iter, lr, variable_lr=True)
    animate_perceptron(df_0, df_1, var_best_model, var_his, save=True)



if __name__ == '__main__':
    main()
