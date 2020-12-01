
import numpy as np
import matplotlib.pyplot as plt
import math

"""
This is a preliminary version of a Perceptron Learning Algorithm.

 * Rafael, I still have some research/verification to do on the content in and related to this project.  (Planning to finish up before our meeting)
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
                W[:,0] = W[:,0] + (1/N) * X[:, p] * Y[p]            # check if local potential is less than zero and update weight if necessary

        if all([e > 0 for e in E_list]): break                      # end training if all E > 0

    # check accuracy against labels
    sign = np.sign(np.dot(np.transpose(W), X))
    sign = sign.reshape((P,1))
    print(sign.shape)



    correct = 0
    for i in range(P):
        if Y[i] == sign[i]: correct += 1
    accuracy = correct/P
    return accuracy

def plot_alpha(alpha, y, N):
    plt.plot(alpha, y, label = "N=" + str(N))
    plt.xlabel('Alpha')
    plt.ylabel('Q_ls')
    #plt.ylim(0, 1)
    plt.title("Q_ls(alpha)")

def main():
    N = [20]    # number of features
    alpha_step = 0.25

    alpha = np.arange(0.75, 5 + alpha_step, alpha_step).tolist()
    mean_acc = 0    # initialize average accuracy counter
    n_D = 50        # number of experiments to run
    n_max = 100     # max number of epochs (sweeps through data)

    # d) train on multiple randomized data sets
    def run_single_N(N):
        acc_per_alpha = []
        for run in alpha:                       # change parameters
            P = int(run*N)                         
            rep_acc = []
            for rep in range(n_D):              # given the parameters average over n_D runs
                accuracy = run_rosenblatt(N, P, n_max)
                rep_acc.append(accuracy)
            acc_per_alpha.append(np.mean(rep_acc))
            # NOTE: ls is linearly separable
            Q_ls = acc_per_alpha

        return Q_ls
    
    for N_i in N:
        N_i_acc = run_single_N(N_i)
        plot_alpha(alpha, N_i_acc, N_i)

    plt.legend()
    plt.show()

main()