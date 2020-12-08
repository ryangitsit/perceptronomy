
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
def run_rosenblatt(N, P, n_max, Q_tot):
    # a) generate data
    X = np.random.normal(0, 1, (N, P))          # randomly generated feature vector matrix
    Y = np.random.choice([1, -1], size=P)       # randomly generated plus/minus 1 labels
    W = np.zeros((N, 1))                # initialize weights to zero
    embedding = np.zeros((N,1))
    
    Q_count = 0
    Q_tot += 1
    E_list = np.zeros((1,P))



    # b) sequential training
    for n in range(n_max):              # epoch loop 
            
        for p in range(P):              # feature vector loop

            # c) Rosenblatt algorithm
            E = np.dot(np.transpose(W[:, 0]), X[:, p]) * Y[p]       # dot weights with features and multiply with label sign (1-D real number)
            E_list[0,p] = E

            if E <= 0:
                W[:,0] = W[:,0] + (1/N) * X[:, p] * Y[p]            # check if local potential is less than zero and update weight if necessary
                embedding[:,0] = embedding[:,0] + E

        if all([E_list[0,e] > 0 for e in range(P)]): 
            Q_count += 1
            break                      # end training if all E > 0
            

    # check accuracy against labels
    sign = np.sign(np.dot(np.transpose(W), X))
    sign = sign.reshape((P,1))
    #print(sign.shape)



    correct = 0
    for i in range(P):
        if Y[i] == sign[i]: correct += 1

    Q_ls_single = Q_count/Q_tot

    return Q_ls_single, embedding

def plot_alpha(alpha, y, N):
    plt.plot(alpha, y, label = "N=" + str(N))
    plt.xlabel('Alpha')
    plt.ylabel('Q_ls')
    #plt.ylim(0, 1)
    plt.title("Q_ls(alpha)")

def main():

    N = [5, 20, 100]    # number of features
    alpha_step = 0.25
    
    alpha = np.arange(0.25, 4 + alpha_step, alpha_step).tolist()
    n_D = 50        # number of experiments to run
    n_max = 100     # max number of epochs (sweeps through data)

    # d) train on multiple randomized data sets
    def run_single_N(N):
        Q_per_alpha = []
        embed_per_alpha = []
        for run in alpha:                       # change parameters
            P = int(run*N)                         
            rep_Q = []
            rep_embed = []
            for rep in range(n_D):              # given the parameters average over n_D runs
                Q_tot = 0
                Q_ls_single , embedding = run_rosenblatt(N, P, n_max, Q_tot)
                rep_Q.append(Q_ls_single)
                rep_embed.append(embedding)
            Q_per_alpha.append(np.mean(rep_Q))
            embed_per_alpha.append(np.mean(rep_embed))
            # NOTE: ls is linearly separable
            Q_ls = Q_per_alpha
            embed = embed_per_alpha
        return Q_ls, embed
    
    for N_i in N:
        N_i_Q, N_i_embed = run_single_N(N_i)
        plot_alpha(alpha, N_i_Q, N_i)

    plt.legend()
    plt.show()

    # for N_i in N:
    #     N_i_Q, N_i_embed = run_single_N(N_i)
    #     plot_alpha(alpha, N_i_embed, N_i)

    # plt.legend()
    # plt.show()
    

main()