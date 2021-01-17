import numpy as np


# With a Q-learning algorithm returns how good is each response.
def player0(data, Q, player, valid, learning_rate, feedback):
    actual = Q[data[player][0]][data[player][1]][data[player][2] - 1][data[player][3]][
        int(np.log2(data[player][4]))]  # How much it weights the actual state.
    p0 = 0  # The amount of points if choose 0.
    p1 = 0  # The amount of points if choose 1.
    p2 = 0  # The amount of points if choose 2.
    p3 = 0  # The amount of points if choose 3.
    if feedback == 0:
        # The probability is 0 because it is an illegal move.
        if valid == 0:
            p1 = actual
            p3 = 0

        else:
            p1 = 0

        # If it is a legal move then the value of playing 0 is the same to raise the points_hand variable,
        # Else it's 0.
        if data[0][4] == 8:
            p0 = 0
        else:
            p0 = Q[data[player][0]][data[player][1]][data[player][2] - 1][data[player][3]][
                int(np.log2(data[player][4])) + 1]

        # If the points of the opponent is greater than 15 then it shouldn't be an option.
        try:
            p2 = Q[data[player][0]][data[player][1] + data[player][4]][data[player][2] - 1][data[player][3]][
                int(np.log2(data[player][4]))]
        except:
            p2 = 0
    else:  # If it is call because of feedback then update the current state
        Q[data[player][0]][data[player][1]][data[player][2] - 1][data[player][3]][
            int(np.log2(data[player][4]))] += learning_rate * (feedback - actual)
    return [p0, p1, p2, p3]


# Taking the probability of the 4 possible inputs, returns the final response.
def players(probability):
    p0 = probability[0]
    p1 = probability[1]
    p2 = probability[2]
    p3 = probability[3]
    normal_const = p0 + p1 + p2 + p3
    chance = np.array([p0, p1, p2, p3]) / normal_const
    choose = np.random.random()
    if choose < chance[0]:
        return 0
    elif choose < chance[0] + chance[1]:
        return 1
    elif choose < chance[0] + chance[1] + chance[2]:
        return 2
    else:
        return 3
