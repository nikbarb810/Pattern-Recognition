import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
import matplotlib.patches as mpatches


def prepare_animation(ax, scatter_points, scatter_labels, xx, yy, posteriors, k_list, best_k):
    def animate(frame_number):
        if frame_number > 0 and k_list[frame_number - 1] == best_k:
            time.sleep(2)

        ax.clear()

        # plot anim, scatter
        plot = ax.contourf(xx, yy, posteriors[frame_number], alpha=.4)
        ax.scatter(scatter_points[:, 0], scatter_points[:, 1], c=scatter_labels, edgecolor='black')

        # add legend and title
        ax.set_title('Animation of KNN')

        colors = ['blue', 'yellow']
        labels = ['class_0', 'class_1']
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        ax.legend(handles=patches)

        ax.text(0.05, 0.95, f'k = {k_list[frame_number]:.1f}', transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top')

        if k_list[frame_number] == best_k:
            ax.text(0.05, 0.9, 'Best k', transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top')

        return plot

    return animate


# calculate euclidean distance between x and all points in train_data
# Input 1: x, a 1x2 array
# Input 2: train_data, a nx2 array
# Output: a nx1 array of distances
def calc_dist(x, train_data):
    # calculate the difference between x and each point in train_data
    diff = x - train_data

    # calculate the square of the difference
    sq_diff = np.square(diff)

    # calculate the sum of the square of the difference
    sum_sq_diff = np.sum(sq_diff, axis=1)

    # calculate the square root of the sum of the square of the difference
    dist = np.sqrt(sum_sq_diff)

    return dist


def import_data(filename):
    data = pd.read_csv(filename, header=None)

    # drop the third column
    labels = data.iloc[:, 2]
    data = data.drop(2, axis=1)
    return data.values, labels


# get the k nearest neighbors of x
# Input 1: x, a 1x2 array
# Input 2: train_data, a nx2 array
# Input 3: k, the number of neighbors
# Output: a kx1 array of indices of the k nearest neighbors
def get_knn(x, train_data, k):
    dist = calc_dist(x, train_data)
    # return the indices of the k smallest distances
    # excluding distance 0, which is the distance between x and itself
    return np.argpartition(dist, k + 1)[1:k + 1]


# calculate probability of each sample belonging to each class
# Input 1: train_data, a nx2 array
# Input 2: train_labels, a nx1 array
# Input 3: test_data, a mx2 array
# Input 4: test_labels, a mx1 array
# Input 5: k, the number of neighbors
def calc_probs(train_data, train_labels, test_data, test_labels, k, debug=False):
    correct = 0
    predictions = []

    # calculate the probability of each sample belonging to each class
    for i in range(len(test_data)):
        # get the k nearest neighbors of the sample
        knn = get_knn(test_data[i], train_data, k)

        # get the labels of the k nearest neighbors
        knn_labels = train_labels[knn]

        # calculate the probability of the sample belonging to each class
        prob_0 = np.count_nonzero(knn_labels == 0) / k
        prob_1 = np.count_nonzero(knn_labels == 1) / k

        if debug:
            # print the probability of the sample belonging to each class
            print('Probability of sample', i + 1, 'belonging to class 0:', prob_0)
            print('Probability of sample', i + 1, 'belonging to class 1:', prob_1)

            # print the actual label of the sample
            print('Actual label of sample', i + 1, ':', test_labels[i])

        predictions.append(np.argmax([prob_0, prob_1]))

        if debug:
            # print the predicted label of the sample
            print('Predicted label of sample', i + 1, ':', predictions[i])
            print()

        if predictions[i] == test_labels[i]:
            correct += 1

    accuracy = correct / len(test_data)
    if debug:
        print(f"Accuracy: {accuracy:.2f}")

    return predictions, accuracy


# plot the data
def plot_data(train_data, train_labels, test_data, test_labels, preds):
    # create a 1x2 subplot
    fig, ax = plt.subplots(1, 2)

    # plot actual labels
    ax[0].scatter(train_data[:, 0], train_data[:, 1], c=train_labels)

    # plot predicted labels
    ax[1].scatter(test_data[:, 0], test_data[:, 1], c=preds)

    # set the title of the plot
    ax[0].set_title('Actual labels')
    ax[1].set_title('Predicted labels')

    plt.show()


def est_best_k(train_data, train_labels, test_data, test_labels, k_cands):
    results = []
    for k in k_cands:
        print('k =', k)
        preds, acc = calc_probs(train_data, train_labels, test_data, test_labels, k)
        results.append((k, preds, acc))

    return results


def getGrid(data, spacing=0.1):
    # create meshgrid based on data
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    return xx, yy


# calculate decision boundaries
def calc_decision_boundaries(samples, labels, xx, yy, k):
    Xgrid = np.c_[xx.ravel(), yy.ravel()]

    # calculate the probability of each point
    preds, acc = calc_probs(samples, labels, Xgrid, np.zeros(len(Xgrid)), k)

    # reshape the predictions
    Z = np.array(preds).reshape(xx.shape)

    return Z


def main():
    train_data, train_labels = import_data('dataC_ΚΝΝtrain.csv')
    test_data, test_labels = import_data('dataC_ΚΝΝtest.csv')

    k_cands = [1, 3, 5, 7, 9, 11, 13, 15]
    results = est_best_k(train_data, train_labels, test_data, test_labels, k_cands)

    best_k = max(results, key=lambda x: x[2])[0]
    print(f"Best k: {best_k}, accuracy: {max(results, key=lambda x: x[2])[2]:.2f}")

    spacing = 0.5
    xx, yy = getGrid(test_data, spacing)

    results = []
    for k in k_cands:
        curr_res = calc_decision_boundaries(train_data, train_labels, xx, yy, k)
        results.append(curr_res)
        print(f"Completed k={k}")

    fig, ax = plt.subplots()

    posts = np.arange(0, len(k_cands), 1)
    anim = animation.FuncAnimation(fig, prepare_animation(ax, test_data, test_labels, xx, yy, results, k_cands, best_k),
                                   len(posts), interval=800, repeat=True)


    plt.show()


if __name__ == '__main__':
    main()
