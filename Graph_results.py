import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt


# This is the function to estimate the win rate of Ray.
def func(x, a, b, c):
    return a / x + c + b * x


def make_Graph():
    x1 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    with open('Training_and_graph_data/lies.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x1.append(float(row[0]))
            y1.append(float(row[1]))
            y2.append(float(row[2]))
            y3.append(float(row[3]))
            y4.append(float(row[4]))
    x1 = np.array(x1)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    ans1 = 0
    ans2 = 0
    aver1 = 0
    aver2 = 0
    for i in range(0, len(y1)):
        aver1 += y1[i] / len(y1)
        aver2 += y2[i] / len(y1)
    print("In average Ray lies: " + str(aver1))
    print("In average Norman lies: " + str(aver2))
    for i in range(0, len(y4)):
        ans1 += y4[i]
    for i in range(0, len(y3)):
        ans2 += y3[i]
    print("The average win rate of Norman vs Ray is: " + str(np.round(ans2 / ans1)))
    plot1 = plt.figure(1)

    plt.scatter(x1, y1 * 100, s=5, marker="o", color="green", label='Q-Learning AI')  # Plots the Q-learning data.

    plt.scatter(x1, 100 * y2, s=5, marker="o", color="blue", label='Neural Network AI')  # Plots the neural network
    # data.

    plt.xlabel('Matches')
    plt.ylabel('% of lies')
    plt.title('lies vs matches')
    plt.legend()
    plt.savefig('Graphs/Amount_of_lies.png')
    plot2 = plt.figure(2)
    plt.scatter(x1, y3 / 100, s=5, marker="o", color="green",
                label='Q-Learning AI Win')  # Plots the Q-learning win rate.
    # The actual curve fitting happens here
    optimizedParameters, pcov = opt.curve_fit(func, x1, y3/100)

    print("The value of the constants in the 1/x function is: " + str(optimizedParameters))
    # Use the optimized parameters to plot the best fit
    plt.plot(x1, func(x1, *optimizedParameters))

    plt.scatter(x1, y4 / 100, s=5, marker="o", color="blue", label='neural network AI Win')  # Plots the neural
    # network win rate. 
    plt.xlabel('Matches')
    plt.ylabel('Win rate')
    plt.title('Win rate AI')
    plt.legend()
    plt.savefig('Graphs/win_rate.png')
    plt.show()

