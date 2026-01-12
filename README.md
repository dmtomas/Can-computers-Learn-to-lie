> [!NOTE]
> My first AI project.

# Motivation and conclusion
I investigated whether an AI model could learn when to bluff in situations with incomplete information. To test this, I trained the model using a higher-or-lower game with variable stakes. The AI competed against itself, with rewards determined by an arbiter that had access to complete information. The results showed that the AI effectively learned to bluff in certain situations.

## Configure a Personal training

1- If you want to configure another AI, change the current AI in Higher_Lower.py on lines 110 and 125.

2- If you want to change the amount of games, amount of games against the AIs, etc. Change variables on line 207 on Higher_Lower.py



## Run the default Training

1- Run index.py.

2- Press 1 to start the training of the AIs.

3- Press 2 to Graph all the data that the training made.

4- Exit pressing 0.


The default Training takes about 1h with an fx8350 a gtx970 and 8gb of ram.

## Requirements

-Tensor Flow

-Matplotlib

-Numpy

-Time

-csv

-scipy.optimize
