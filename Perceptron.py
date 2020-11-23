
import numpy as np

"""
This is a preliminary version of a Perceptron Learning Algorithm.

 * Raphael, I still have some research/verification to do on the content in and related to this project.  (Planning to finish up before our meeting)
 - This was just an intuitive first draft to play around with.  I believe my indexing for w-updates is probably not corrent and that not all values are updated.
 - The sign equation is probably not correct.  This was just an intuitive guess.  I will look more carefully at it later.
 - Part D from assignment is not included.  However, the loop to create a new data set and run the experiment multiple times is present.  
 - Average accuracy is printed.  It becomes clear that for differnt P and N ratios, there are different accuracies, so part D makes sense.
 - Have not looked at the bonus questions.

"""

A = 0       # initialize average accuracy counter
runs = 10   # How many experiments to run

for z in range(runs):

    P = 50                              # number of feature vectors 
    N = 500                             # number of features 
    X = np.random.normal(0,1,(N,P))     # randomly generated feature vector matrix
    Y = np.random.randint(2, size=P)    # randomly generated binary labels 
    n_max = 100                         # how many sweeps through data

    for i in range(len(Y)):             # converting binary to plus/minus 1
        if Y[i] == 0:
            Y[i] = -1
        
    W = np.zeros((N,P))                 # initialize weights to zero

    for n in range(n_max):              # sweeping loop

        for p in range(P-1):                # feature vector loop

            E = np.dot(np.transpose(W[:,p]),X[:,p]) * Y[p]      # dotting weights with features and multiplying with label sign (1-D real number)

            if E <= 0:                                          # check if local potential is less than zero
                W[:,p+1] =  W[:,p] + (1/N)*X[:,p]*Y[p]          # update next weight if necessary


    sign = np.sign((np.dot(np.transpose(W),X)).sum(axis=0))     # sign of sum of dotted columns (probably should fix this!)

    correct = 0                     # check accuracy against labels
    for i in range(len(Y)):
        if Y[i] == sign[i]:
            correct += 1
        accuracy = correct/len(Y)

    print(accuracy)

    A += accuracy

A = A/runs # average accuracy across all runs 

print("AVG ACC = ", A)  # depending on values of N,P, and n-max, usually achieves .75-.9


