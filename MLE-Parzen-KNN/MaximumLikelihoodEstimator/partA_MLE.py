
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


class Distribution:
    # Input 1: Name of distribution
    # Input 2: nX2x1 array of samples
    # Constructor
    def __init__(self, _name, _samples):
        self.name = _name
        self.samples = self.format_samples(_samples)
        self.mean = None
        self.cov = None
        self.probs = None

    def run(self):
        self.mean = self.calc_mean()
        self.cov = self.calc_cov()
        self.probs = self.calc_probs()

    # Format the input samples to be a nX2x1 array
    def format_samples(self, _samples):
        np_samples = np.asarray(_samples)
        return np_samples.reshape(-1, 2, 1)

    def calc_mean(self):
        return (1 / len(self.samples)) * np.sum(self.samples, axis=0)

    def calc_cov(self):
        # calculate the difference between each sample and the mean
        diffs = self.samples - self.mean

        sum = 0
        for dif in diffs:
            sum += np.matmul(dif, np.transpose(dif))

        return (1 / len(self.samples)) * sum

    def calc_probs(self, samples=None):

        # if samples is not given, use the samples of the distribution
        if samples is None:
            samples = self.samples

        # calculate the exponent
        exp = np.matmul(np.matmul(np.transpose(samples - self.mean, (0, 2, 1)), np.linalg.inv(self.cov)),
                        (samples - self.mean))

        # calculate the denominator
        denom = np.sqrt(np.linalg.det((2 * np.pi) ** 2 * self.cov))

        # calculate the probability
        prob = np.exp(-0.5 * exp) / denom

        # flatten the array
        return prob.flatten()

    def plot(self):
        # plot the samples
        plt.scatter(self.samples[:, 0], self.samples[:, 1])

        # set the title
        plt.title(self.name)

        # find the min and max of the samples
        min_x = np.min(self.samples[:, 0])
        max_x = np.max(self.samples[:, 0])
        min_y = np.min(self.samples[:, 1])
        max_y = np.max(self.samples[:, 1])

        # create linspace with some padding
        padding = 0.5
        x = np.linspace(min_x - padding, max_x + padding, 100)
        y = np.linspace(min_y - padding, max_y + padding, 100)

        # create meshgrid
        X, Y = np.meshgrid(x, y)
        Xgrid = np.vstack((X.flatten(), Y.flatten())).T

        # calculate the probability using calc_probs
        Z = self.calc_probs(Xgrid.reshape(-1, 2, 1)).reshape(X.shape)

        plt.contour(X, Y, Z)
        plt.show()


def plot_3d(D0, D1, D2, style='3D'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # calculate meshgrid that encompasses all distributions

    # find the min and max of the samples
    min_x = np.min(np.concatenate((D0.samples[:, 0], D1.samples[:, 0], D2.samples[:, 0])))
    max_x = np.max(np.concatenate((D0.samples[:, 0], D1.samples[:, 0], D2.samples[:, 0])))
    min_y = np.min(np.concatenate((D0.samples[:, 1], D1.samples[:, 1], D2.samples[:, 1])))
    max_y = np.max(np.concatenate((D0.samples[:, 1], D1.samples[:, 1], D2.samples[:, 1])))

    # create linspace with some padding
    padding = 0.5
    x = np.linspace(min_x - padding, max_x + padding, 100)
    y = np.linspace(min_y - padding, max_y + padding, 100)
    X, Y = np.meshgrid(x, y)
    Xgrid = np.vstack((X.flatten(), Y.flatten())).T

    # calculate the probability using calc_probs
    Z0 = D0.calc_probs(Xgrid.reshape(-1, 2, 1)).reshape(X.shape)
    Z1 = D1.calc_probs(Xgrid.reshape(-1, 2, 1)).reshape(X.shape)
    Z2 = D2.calc_probs(Xgrid.reshape(-1, 2, 1)).reshape(X.shape)

    if style != '3D':
        max_Z = np.maximum.reduce([Z0, Z1, Z2])
        plt.contourf(X, Y, max_Z, levels=360, cmap='RdYlBu')
    else:
        # calculate the label with the maximum probability for each point in the meshgrid
        max_indices = np.argmax(np.stack((Z0, Z1, Z2), axis=2), axis=2)

        # define the RGB values for each label
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # create a new array where each point is assigned a color based on the label with the maximum probability
        Z_color = colors[max_indices]

        # set the Z value of each point in the meshgrid to the maximum probability among the three distributions
        Z = np.maximum.reduce([Z0, Z1, Z2])

        # plot the surface with the assigned colors
        ax.plot_surface(X, Y, Z, facecolors=Z_color, linewidth=0.7, antialiased=True, alpha=0.55)

    # add a title
    plt.title("3D plot of all distributions")

    # add labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Probability')

    # add label legend based on color (red, green/yellow, blue)
    l0_patch = mpatches.Patch(color='red', label=D0.name)

    if style != '3D':
        l1_patch = mpatches.Patch(color='yellow', label=D1.name)
    else:
        l1_patch = mpatches.Patch(color='green', label=D1.name)

    l2_patch = mpatches.Patch(color='blue', label=D2.name)
    plt.legend(handles=[l0_patch, l1_patch, l2_patch])

    plt.show()


def main():
    # read csv
    samples_data = pd.read_csv("dataA_MLE.csv", header=None)

    # format dataframe, by splitting it into unique labels
    dfs = [group[1].drop(2, axis=1) for group in samples_data.groupby(2)]

    dis0 = Distribution("dis0", dfs[0])
    dis0.run()
    #dis0.plot()

    dis1 = Distribution("dis1", dfs[1])
    dis1.run()
    #dis1.plot()

    dis2 = Distribution("dis2", dfs[2])
    dis2.run()
    #dis2.plot()

    plot_3d(dis0, dis1, dis2, style="3D")
    plot_3d(dis0, dis1, dis2, style="contour")



if __name__ == "__main__":
    main()
