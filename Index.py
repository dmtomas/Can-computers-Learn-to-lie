from Higher_or_lower.Graph_results import *
from Higher_or_lower.Higher_Lower import *

while True:
    action = int(input("Decide your action (0/1/2) \n 3 for instructions: "))
    if action == 0:
        break
    if action == 1:
        complete_training()
    if action == 2:
        try:
            make_Graph()
        except:
            print("You don't have data to graph: ")
    if action == 3:
        print("press 0 to exit")
        print("press 1 if you want to train the AIs")
        print("press 2 if you want to graph the data")



