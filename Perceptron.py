
import numpy as np
import matplotlib.pyplot as plt

"""
This is a preliminary version of a Perceptron Learning Algorithm.

 * Raphael, I still have some research/verification to do on the content in and related to this project.  (Planning to finish up before our meeting)
 - This was just an intuitive first draft to play around with.  I believe my indexing for w-updates is probably not corrent and that not all values are updated.
 - The sign equation is probably not correct.  This was just an intuitive guess.  I will look more carefully at it later.
 - Part D from assignment is not included.  However, the loop to create a new data set and run the experiment multiple times is present.  
 - Average accuracy is printed.  It becomes clear that for different P and N ratios, there are different accuracies, so part D makes sense.
 - Have not looked at the bonus questions.

"""

# perform a run on a single set of data
def run_rosenblatt(N, P, n_max):
    # a) generate data
    X = np.random.normal(0, 1, (N, P))          # randomly generated feature vector matrix
    Y = np.random.choice([1, -1], size=P)       # randomly generated plus/minus 1 labels
    W = np.zeros((N, P))                # initialize weights to zero

    # b) sequential training
    for n in range(n_max):              # epoch loop        
        E_list = []
        for p in range(P):              # feature vector loop

            # c) Rosenblatt algorithm
            E = np.dot(np.transpose(W[:, p]), X[:, p]) * Y[p]       # dot weights with features and multiply with label sign (1-D real number)
            E_list.append(E)
            if E <= 0:
                W[:, p] = W[:, p] + (1/N) * X[:, p] * Y[p]          # check if local potential is less than zero and update weight if necessary

        if all([e > 0 for e in E_list]): break                      # end training if all E > 0

    # check accuracy against labels
    sign = np.sign((np.dot(np.transpose(W), X)).sum(axis=0))     # sign of sum of dotted columns
    correct = 0
    for i in range(len(Y)):
        if Y[i] == sign[i]: correct += 1
    accuracy = correct/len(Y)

    return accuracy

def main():
    N = 20           # number of features
    alpha = np.arange(0.75, 3.25, 0.25).tolist()
    mean_acc = 0    # initialize average accuracy counter
    n_D = 50        # number of experiments to run
    n_max = 100     # max number of epochs (sweeps through data)

    # d) train on multiple randomized data sets
    collect_acc = []
    for run in alpha:                       # change parameters
        P = int(run*N)                         
        for rep in range(n_D):              # given the parameters average over n_D runs
            accuracy = run_rosenblatt(N, P, n_max)
            collect_acc.append(accuracy)

        mean_acc = np.mean(collect_acc)     # average accuracy across all runs
        print(f"AVG ACC for P={P} is {mean_acc}")

main()