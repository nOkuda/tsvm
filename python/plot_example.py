"""Plotting for example"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run():
    """Plots data"""
    data = []
    with open('../data/example.data') as ifh:
        for line in ifh:
            data.append([float(a) for a in line.strip().split(',')])
    data = np.array(data)
    initialanswer = np.array([-0.3499999989415537, 0.2299999993580931])
    finalanswer = np.array([-1.070895993602519, -0.274627197579686])
    initialbias = 0.011620685479047073
    finalbias = 0.5010333531255504
    xvalues = np.linspace(-0.1, 0.9)
    initialline = [-(a*initialanswer[0]+initialbias)/initialanswer[1] for a in xvalues]
    finalline = [-(a*finalanswer[0]+finalbias)/finalanswer[1] for a in xvalues]
    plt.plot(xvalues, initialline, 'k:')
    plt.plot(xvalues, finalline, 'k-')
    plt.scatter(data[0:2, 0], data[0:2, 1], s=64, c='r', marker='+')
    plt.scatter(data[2:4, 0], data[2:4, 1], s=64, c='r', marker='o')
    plt.scatter(data[4:6, 0], data[4:6, 1], s=64, c='b', marker='o')
    plt.scatter(data[6:8, 0], data[6:8, 1], s=64, c='b', marker='_')
    plt.xlim(-0.1, 0.9)
    plt.ylim(0.2, 1.0)
    plt.savefig('../results/example_plot.png')

if __name__ == '__main__':
    run()

