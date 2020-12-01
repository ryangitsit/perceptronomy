
import numpy as np
import matplotlib.pyplot as plt
import math

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
    W = np.zeros((N, 1))                # initialize weights to zero

    # b) sequential training
    for n in range(n_max):              # epoch loop        
        E_list = []
        for p in range(P):              # feature vector loop

            # c) Rosenblatt algorithm
            E = np.dot(np.transpose(W[:, 0]), X[:, p]) * Y[p]       # dot weights with features and multiply with label sign (1-D real number)
            E_list.append(E)
            if E <= 0:
                W[:, 0] = W[:, 0] + (1/N) * X[:, p] * Y[p]          # check if local potential is less than zero and update weight if necessary

        if all([e > 0 for e in E_list]): break                      # end training if all E > 0

    # check accuracy against labels
    #sign = np.sign((np.multiply(W, X)).sum(axis=0))        # sign of sum of dotted columns
    sign = np.sign(np.dot(np.transpose(W), X))
    sign = sign.reshape((P,1))
    #print(f"X = {sign} \nY = {Y} \n\n")

    correct = 0
    for i in range(len(Y)):
        if Y[i] == sign[i]: correct += 1
    accuracy = correct/len(Y)
    #print(np.sum(np.dot(W,np.transpose(W))))
    return accuracy

def plot_alpha(alpha, y):
    plt.plot(alpha, y)
    plt.xlabel('Accuracy')
    plt.ylabel('Alpha')
    plt.title("Q_ls(alpha)")
    plt.show()

# # see equation 3.42
# def get_P_ls(P, N):
#     if P <= N:
#         P_ls = 1
#     else:
#         binomial = []
#         for i in range(N):
#             binomial.append(math.comb((P-1), i))
#         P_ls = (2**(1-N))*sum(binomial)
#     return P_ls

def main():
    N = 20          # number of features
    alpha = np.arange(0.25, 4.25, 0.25).tolist()
    mean_acc = 0    # initialize average accuracy counter
    n_D = 50        # number of experiments to run
    n_max = 100     # max number of epochs (sweeps through data)

    # d) train on multiple randomized data sets
    acc_per_p = []
    P_ls_collect = []
    for run in alpha:                       # change parameters
        P = int(run*N)                         
        rep_acc = []
        for rep in range(n_D):              # given the parameters average over n_D runs
            accuracy = run_rosenblatt(N, P, n_max)
            rep_acc.append(accuracy)

        # NOTE: ls is linearly separable
        Q_ls = np.mean(rep_acc)     # average accuracy across all runs
        acc_per_p.append(Q_ls)

        #P_ls_collect.append(get_P_ls(P,N))

        #print(f"Q_l.s. (mean accuracy) for P={P} is {Q_ls}")
    
    plot_alpha(alpha, acc_per_p)

main()