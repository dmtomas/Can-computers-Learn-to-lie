import numpy as np
import time
import csv
import tensorflow as tf
from Higher_or_lower.QAI import player0, players
from Higher_or_lower.neuralAI import train_Neural


def mentiras(mentira, partidas, win_rate):
    lista = [partidas, mentira[0], mentira[1], win_rate[0], win_rate[1]]
    try:
        with open("Training_and_graph_data/lies.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(lista)
    except:
        return 0
    return 0


def data_writer(data, valid, d):
    resp = list(data[1])
    resp.append(valid)
    # The following code determines the best play knowing all the information.
    if valid == 0 and data[1][2] > data[0][2]:
        if data[1][4] < 8:
            resp.append(0)
        else:
            resp.append(1)
    elif valid == 0 and data[0][2] <= 5:
        resp.append(0)
    elif valid == 0 and data[0][2] > 5:
        resp.append(2)
    elif valid == 1 and data[1][2] > data[0][2]:
        if data[1][2] < 8:  # This is to make the AI notice that 3 is also a good option in some cases.
            resp.append(3)
        elif data[1][4] < 8:
            resp.append(0)
        else:
            resp.append(1)
    elif valid == 1 and data[0][2] < 5:
        resp.append(0)
    elif valid == 1 and data[0][2] >= 5:
        resp.append(2)
    else:
        if valid == 1:
            resp.append(3)
        else:
            resp.append(1)
    name = "Training_and_graph_data/neural_network_data" + str(d) + ".csv"
    with open(name, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(resp)

    return 0


def hand_maker():
    hand = []
    for i in range(0, 2):
        hand.append([np.random.randint(1, 13)])
        hand[i].append(np.random.randint(1, 5))

    # The cards can't be repeated.
    while hand[0] == hand[1]:
        hand[1][0] = np.random.randint(1, 13)
        hand[1][1] = np.random.randint(1, 5)
    return hand


def game(points, player, data, Q, learning_rate, d, player1):
    # 0 = increase the points
    # 1 = throw a card.
    # 2 = give up
    # 3 = accept the increment.
    feedback = 0  # How good it was this match for the AI.
    valid = 0  # 0 if in the actual state it is possible to throw a card.
    hand = hand_maker()  # This is going to be the hand that both players have.
    points_hand = 1  # How many points the winner is going to gain.
    table = [0, 0]  # Which cards are on the table.
    # With this loop, we give each player 1 of the 48 available cards on the deck.

    # This loop is the one that decides the match.
    while True:

        started = player  # With this variable, we can be sure that the player that starts can't change.

        # If it is the first time playing add the information of the table.
        if len(data[player]) == 2 and len(data[player - 1]) == 2:
            data[player].extend([hand[player][0], table[player - 1], points_hand])
            data[player - 1].extend([hand[player - 1][0], table[player], points_hand])

        # If the data is already complete, update it.
        elif len(data[player]) == 5 and len(data[player - 1]) == 5:
            data[player][3] = table[player - 1]
            data[player - 1][3] = table[player]
            data[player][4] = data[player - 1][4] = points_hand

        # Create the data for the neural network
        neural_data = list(data[1])
        if len(neural_data) == 5:
            neural_data.append(valid)

        # The action that the player is going to make.
        # If the player1 is available then use the neural network, otherwise, just run the Q-Learning algorithm.
        if player == 0:
            action = int(players(player0(data, Q, player, valid, learning_rate, feedback)))
        elif player1 != 0:
            neural_data[5] = valid
            ans = player1(tf.Variable([np.array(neural_data)], trainable=False, dtype=tf.float64))
            action = int(players(ans[0]))
        else:
            action = int(players(player0(data, Q, player, valid, learning_rate, feedback)))

        data_writer(data, valid, d)

        while action == 0 and points_hand < 8:
            points_hand *= 2
            valid = 1  # You can't throw a card right now.

            if player == 0:
                action = int(players(player0(data, Q, player, valid, learning_rate, feedback)))
            elif player1 != 0:
                neural_data[5] = valid
                ans = player1(tf.Variable([np.array(neural_data)], trainable=False, dtype=tf.float64))
                action = int(players(ans[0]))
            else:
                action = int(players(player0(data, Q, player, valid, learning_rate, feedback)))

            if action == 2:  # If the player wants to give up not showing the cards.
                points[player - 1] += points_hand // 2
                return 0
            elif action == 0:
                data[player][4] = points_hand
                data[player - 1][4] = points_hand
                data_writer(data, valid, d)
            elif action == 3:
                break

        player = started  # The player that has to throw is the same the started the game.
        valid = 0  # It is valid again to throw a card.
        neural_data[5] = valid  # Update the valid data for the neural network.

        if action == 2:
            # If someone give up then give feedback to the first player.
            feedback = (data[0][0] + 1) / (16 * (data[0][1] + 1))
            player0(data, Q, player, valid, learning_rate, feedback)
            points[player - 1] += points_hand
            return 0

        # If a card is thrown, change the table.
        if action == 1:
            table[player] = hand[player][0]

        # if both players have thrown their cards then choose the winner.
        if table[player] != 0 and table[player - 1] != 0:
            if hand[player] > hand[player - 1]:
                feedback = (data[0][0] + 1) / (16 * (data[0][1] + 1))
                player0(data, Q, player, valid, learning_rate, feedback)
                points[player] += points_hand
                return 0
            else:  # The second player has an advantage so if they tie the first player wins.
                feedback = (data[0][0] + 1) / (16 * (data[0][1] + 1))
                player0(data, Q, player, valid, learning_rate, feedback)
                points[player - 1] += points_hand
                return 0

        if player == 0:  # Every time they make a decision, then the other player has to answer.
            player = 1
        else:
            player = 0


# This function is going to train both AI.
def complete_training():
    Q = []  # This matrix is going to have all the possible values for each state.

    # If possible load both AI in other case, create them.
    try:
        with open('AI_Matrix/AI5.npy', 'rb') as f:
            Q = np.load(f)

    except:
        # This is going to tell initially how much does it weight each state.
        # It was chosen to give extra points if win and to see some pattern about high cards and high chances to win.
        for points1 in range(0, 16):
            Q.append([])
            for points2 in range(0, 16):
                Q[points1].append([])
                for cards in range(0, 12):
                    Q[points1][points2].append([])
                    for tables in range(0, 13):
                        Q[points1][points2][cards].append([])
                        for points_match in range(0, 4):
                            # This way the values can't be negatives
                            Q[points1][points2][cards][tables].append((points1 + 1) / (16 * (points2 + 1)))
    try:
        with open('Training_and_graph_data/lies.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                matches = int(row[0])
    except:
        matches = 0  # Amount of games played by both AI.
    player1 = 0  # When the AI is not trained he's value is 0.

    # All the variables that the game needs.
    lie = [0, 0]
    previous = 0  # Amount of games player until the last match.
    n_iterations = 500000  # How many times is going to train.
    learning_rate = 0.25  # How much the new data is going to weight.
    start_time = time.time()
    last_time = 0
    steps = 25  # Steps for reducing the learning rate.
    files = 100  # How many files with training for the neural network you need.
    div = 100  # How many times players play without the neural network training.
    victories = np.array([0, 0])  # How many games each player won on the last set.

    change = n_iterations // steps

    for i in range(1, n_iterations + 1):
        # Update how fast it updates the table and restarts the variables.
        if i % change == 0:
            learning_rate *= 0.9
        points = [0, 0]
        player = 0
        if i % (n_iterations // files) == 0:  # Every time the other players finish, train the neural network with
            # that data.
            train_Neural((i // (n_iterations // files)) - 1)
            player1 = tf.keras.models.load_model('saved_model/my_model2')  # Load the trained neural network.

        if i % (n_iterations // files) == div:  # When they finish their matches make the graph data.
            lie[0] /= (matches - previous)
            lie[1] /= (matches - previous)
            mentiras(lie, matches, victories)
            previous = matches
            victories = np.array([0, 0])

        while points[0] < 15 and points[1] < 15:
            data = [[points[0], points[1]], [points[0], points[1]]]

            # They are going to play 50 matches against each other and then the Q-learning is going to play 50
            # Alone so the neural network can train.

            if i % (n_iterations // files) < div:
                game(points, player, data, Q, learning_rate, (i // (n_iterations // files)), player1)
                if points[0] >= 15:
                    victories[0] += 1
                elif points[1] >= 15:
                    victories[1] += 1
                matches += 1
                if data[0][4] > 1 and data[0][3] < 6:
                    lie[0] += 1
                elif data[0][4] > 1 and data[1][3] < 6:
                    lie[1] += 1
                if player == 0:
                    player = 1
                else:
                    player = 0
            else:
                game(points, player, data, Q, learning_rate, (i // (n_iterations // files)), 0)
                lie = [0, 0]

        # Counts on a shorter lapse of time to keep track.
        if i % 25000 == 0:
            print(
                "Ya pasaron: " + str(i) + "       --- %s seconds ---" % np.round((time.time() - start_time - last_time),
                                                                                 3))
            last_time = time.time() - start_time

        # Saves the Q data on a npy file.
        if i % (n_iterations // steps) == 0:
            with open("AI_Matrix/AI.npy", 'wb') as f:
                np.save(f, Q)

            print(
                "The matrix was saved on: " + str(
                    i) + " iterations, and it took " "       --- %s seconds ---" % np.round((time.time() - start_time),
                                                                                            3))

