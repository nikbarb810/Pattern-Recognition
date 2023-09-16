import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import norm
import matplotlib.patches as mpatches





class obj_bundler:
    def __init__(self, ax, axis, est_func, real_func, h_n_vals, mse_vals, best_h_n):
        self.ax = ax
        self.axis = axis
        self.est_func = est_func
        self.real_func = real_func
        self.h_n_vals = h_n_vals
        self.mse_vals = mse_vals
        self.best_h_n = best_h_n


def prepare_animation(objects):
    # render each frame
    def animate(frame_number):

        # if previous frame was best estimation, freeze for 2 seconds to show it
        if (frame_number > 0 and objects.h_n_vals[frame_number - 1] == objects.best_h_n):
            time.sleep(2)

        objects.ax.clear()

        objects.ax.set_title('Animation of Estimation')

        # set axis limits
        objects.ax.set_ylim([0, .2])

        # plot animation of curr_est
        plot = objects.ax.bar(objects.axis, objects.est_func[frame_number])

        # plot real distribution
        objects.ax.plot(objects.axis, objects.real_func, color='red', linewidth=2)

        # plot legend
        colors = ['blue', 'red']
        labels = ['Estimated', 'Real']
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        objects.ax.legend(handles=patches)

        objects.ax.text(0.05, 0.95, f'h_n = {objects.h_n_vals[frame_number]:.1f}', transform=objects.ax.transAxes,
                        fontsize=10,
                        verticalalignment='top')

        objects.ax.text(0.05, 0.9, f'mse = {objects.mse_vals[frame_number]:.4f}', transform=objects.ax.transAxes,
                        fontsize=10,
                        verticalalignment='top')

        # if we are at the best h_n, print it and freeze anim for 1 second
        if (objects.best_h_n == objects.h_n_vals[frame_number]):
            objects.ax.text(0.05, 0.8, f'best h_n', transform=objects.ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        return plot

    return animate


# hypercube window function
def hypercube_window(x, x_i, h_n):
    return 1 if np.abs(x - x_i) <= h_n / 2 else 0


# gaussian window function
def gaussian_window(x, x_i, h_n):
    return (1 / np.sqrt(2 * np.pi * h_n)) * np.exp(-0.5 * ((x - x_i) / h_n) ** 2)


# Calculate the pdf at a single point x_i
# Input:
#   x_i: the point to calculate the pdf at
#   samples: the samples
#   h_n: the window size
#   kernel: 'hypercube' or 'gaussian'

def parzen_pdf(x_i, samples, h_n, kernel):
    ins_cnt = 0
    for x in samples:
        ins_cnt += kernel(x, x_i, h_n)

    return np.asarray(ins_cnt / (len(samples) * h_n))


# Estimate best window size
# Input:
#   samples: the samples
#   real_dis: the real distribution of the samples
#   kernel: 'hypercube' or 'gaussian'
def parzen_window_selection(samples, real_dis, kernel):
    print(f"In window selection, kernel: {kernel.__name__}")
    # define a linspace of x values based on samples
    buffer = 2

    x = np.linspace(np.min(samples) - buffer, np.max(samples) + buffer, 100)

    # define a list of window sizes
    h_n_cands = np.arange(0.2, 10.1, 0.1)

    # define a list to store the pdf values
    pdf_list = []

    # calculate the pdf for each window size
    for h_n in h_n_cands:
        pdf_list.append(np.asarray([parzen_pdf(x_i, samples, h_n, kernel) for x_i in x]))

    # calculate the mean squared error for each window size
    mse_list = []
    for pdf in pdf_list:
        mse_list.append(np.mean((pdf - real_dis) ** 2))
        print(f"mse: {mse_list[-1]}  h_n: {h_n_cands[len(mse_list) - 1]}")

    # find the best window size
    best_h_n = h_n_cands[np.argmin(mse_list)]
    best_dis = pdf_list[np.argmin(mse_list)]

    print(f"Best window size: {best_h_n}")

    # create a 1x3 subplot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # plot the histogram of the samples
    ax[0].hist(samples, bins=100)
    ax[0].set_title('Histogram of Samples')

    # plot real vs best estimated pdf on the same graph

    ax[1].plot(x, real_dis, label='True Distribution', color='red')
    ax[1].plot(x, best_dis, label='Estimated Distribution', color='blue')
    ax[1].set_title('Real vs Estimated Distribution')

    # add a color legend
    red_patch = mpatches.Patch(color='red', label='True Distribution')
    blue_patch = mpatches.Patch(color='blue', label='Estimated Distribution')
    ax[1].legend(handles=[red_patch, blue_patch])


    # animate the distribution estimation
    # each frame is a different window size
    posts = np.arange(0, len(h_n_cands), 1)

    # reshape pdf_list to [-1,]
    pdf_list = [pdf.reshape(-1, ) for pdf in pdf_list]

    anim_objects = obj_bundler(ax[2], x, pdf_list, real_dis, h_n_cands, mse_list, best_h_n)


    anim = animation.FuncAnimation(fig, prepare_animation(anim_objects), frames=posts, interval=300)

    plt.show()

    return best_h_n


# define main function
def main():
    # read csv file
    df = pd.read_csv('dataB_Parzen.csv', header=None)

    samples = np.asarray(df)

    # define a linspace of x values based on samples
    buffer = 2
    x = np.linspace(np.min(samples) - buffer, np.max(samples) + buffer, 100)

    true_dis = norm.pdf(x, loc=1, scale=4)

    # reshape the true_dis
    true_dis = true_dis.reshape(-1, 1)


    rect_h = parzen_window_selection(samples, true_dis, hypercube_window)
    gauss_h = parzen_window_selection(samples, true_dis, gaussian_window)


if __name__ == '__main__':
    main()
