"""Plotting for example"""
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_data():
    """Extracts training and testing data"""
    data = []
    with open('../data/example.data') as ifh:
        for line in ifh:
            data.append([float(a) for a in line.strip().split(',')])
    return np.array(data)

def extract_results():
    """Extracts results from output"""
    results = []
    with open('../results/example.out') as ifh:
        for line in ifh:
            if line.startswith('#####'):
                results.append({})
                for line in ifh:
                    if line.startswith('predictions'):
                        _, values = line.split()
                        results[-1]['predictions'] = np.array(ast.literal_eval(values))
                        break
                for line in ifh:
                    if line.startswith('weights'):
                        _, values = line.split()
                        results[-1]['weights'] = np.array(ast.literal_eval(values))
                        break
                for line in ifh:
                    if line.startswith('bias'):
                        _, value = line.split()
                        results[-1]['bias'] = float(value)
                        break
    return results

def plot(data, curresult, counter):
    """Plots data"""
    xvalues = np.linspace(-0.1, 0.9)
    colorcoding = {
        -1: 'b',
        1: 'r'
    }
    initialline = [-(a*curresult['weights'][0]+curresult['bias'])/\
        curresult['weights'][1] for a in xvalues]
    plt.plot(xvalues, initialline, 'k:')
    plt.scatter(data[0:2, 0], data[0:2, 1], s=64, c='r', marker='+')
    plt.scatter(data[6:8, 0], data[6:8, 1], s=64, c='b', marker='_')
    for i in range(4):
        plt.scatter(
            data[2+i, 0], data[2+i, 1], s=64,
            c=colorcoding[curresult['predictions'][i]], marker='o')
    plt.xlim(-0.1, 0.9)
    plt.ylim(0.2, 1.0)
    plt.savefig('../results/example_plot.'+str(counter)+'.png')
    plt.close()

def run():
    """Gets and plots data"""
    data = extract_data()
    results = extract_results()
    for i, curresult in enumerate(results):
        plot(data, curresult, i)

if __name__ == '__main__':
    run()

