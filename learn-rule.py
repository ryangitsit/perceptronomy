
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import math


"""
- what is meant by |w_star|^2 = N ?  How to define it?
- what is w(t_max)?  How can we use it as part of the stopping criterion if we don't know it yet?
"""


# perform a run on a single set of data
def run_rosenblatt(N, P, n_max):
    # a) generate data
    X = np.random.normal(0, 1, (N, P))               # randomly generated feature vector matrix
    w_star = np.ones((N,1)) #np.random.dirichlet(np.ones(N), size=N)
    Y = np.transpose(np.sign(np.dot(np.transpose(w_star), X)))     # randomly generated plus/minus 1 labels
    W =  np.zeros((N, 1))
    E_list = np.zeros((1,P))                         # Initialize vector to hold local potentials


    # b) sequential training
    for n in range(n_max):              # epoch loop 
            
        for p in range(P):              # feature vector loop
            
            E = np.dot(np.transpose(W[:, 0]), X[:, p]) * Y[p]       # dot weights with features and multiply with label sign (1-D real number)

            E_list[0,p] = E                                         # keep all local potentials in a list

            kappa_min = np.argmin(E_list)

            W[:,0] = W[:,0] + (1/N) * X[:, kappa_min] * Y[kappa_min]    # check if local potential is less than zero and update weight if necessary

    w_t = W[:,0].reshape((N,1))

    g_e = (1/3.14)*np.arccos(np.dot(np.transpose(w_t),w_star)/(np.linalg.norm(w_t)*np.linalg.norm(w_star)))
        
    # sign = np.sign(np.dot(np.transpose(W), X)) # determine the sign for each training instance
    # sign = sign.reshape((P,1))

    # correct = 0
    # for i in range(P):
    #     if Y[i] == sign[i]: correct += 1
    # acc = correct/P

    Q_ls = g_e
    return Q_ls


def plot_alpha(alpha, y, N):
    plt.plot(alpha, y, label = "N=" + str(N))
    plt.xlabel('Alpha=P/N')
    plt.ylabel('e_g(t_max)')
    #plt.ylim(0, 1)
    plt.title("Learning Curve (Average Over N_D=10")

def main():

    N = [5, 20, 50]    # number of features
    
    alpha_step = 0.25
    
    alpha = np.arange(0.1, 5 + alpha_step, alpha_step).tolist()
    n_D = 10        # number of experiments to run
    n_max = 100     # max number of epochs (sweeps through data)

    # d) train on multiple randomized data sets
    def run_single_N(N):
        Q_per_alpha = []

        for run in alpha:                       # change parameters
            P = int(run*N)                      # iterate through different P values (as ratios of alpha)   
            rep_Q = []                          # intialize list of all Q values

            for rep in range(n_D):                                      # iteratate through n_D random datasets
                Q_ls_single = run_rosenblatt(N, P, n_max)               # run Rosenblatt for interating parameters
                rep_Q.append(Q_ls_single)                               # append results

            Q_per_alpha.append(np.mean(rep_Q))                          # take avarage of results
            Q_ls = Q_per_alpha         

        return Q_ls
    
    for N_i in N:
        N_i_Q = run_single_N(N_i)
        plot_alpha(alpha, N_i_Q, N_i)

    plt.legend()
    plt.show()



main()
