
import numpy as np
import matplotlib.pyplot as plt
import math
import random

"""
- what is meant by |w_star|^2 = N ?  How to define it?
- what is w(t_max)?  How can we use it as part of the stopping criterion if we don't know it yet?
"""

# initialize w* to satisfy |w|² = N
def init_w_star(N):
    '''
    This function was derived as follows:

    |w|² = N
    |w| = sqrt(N)
    sqrt(x1² + x2² + ... + xn²) = sqrt(N)
    x1² + x2² + ... + xn² = N
    x1 + x2 + ... xn = sqrt(N)
    '''

    w = np.zeros(N)
    for i in range(N):
        if N <= 0: break
        if N == i: w[i] = math.sqrt(N)
        w[i] = random.uniform(0, math.sqrt(N))
        N -= w[i]**2
    return w

# perform a run on a single set of data
def run_rosenblatt(N, P, n_max, stopped):
    # a) generate data
    X = np.random.normal(0, 1, (N, P))               # randomly generated feature vector matrix
    #w_star = np.ones((N,1)) #np.random.dirichlet(np.ones(N), size=N)
    w_star = init_w_star(N)


    y = np.transpose(np.sign(np.dot(np.transpose(w_star), X)))     # randomly generated plus/minus 1 labels
    W =  np.zeros((N, 1))
    E_list = np.zeros((1,P))                         # Initialize vector to hold local potentials
    kappa_list = np.zeros((1,P))

    # b) sequential training
    for n in range(n_max):              # epoch loop 
            
        for p in range(P):              # feature vector loop
            
            E = np.dot(np.transpose(W[:, 0]), X[:, p]) * y[p]       # dot weights with features and multiply with label sign (1-D real number)


            E_list[0,p] = E                                         # keep all local potentials in a list

            kappa_list[0,p] = E/np.linalg.norm(W[:, 0])

            kappa_min = np.argmin(E_list)

            last_w = W[:,0]

            W[:,0] = W[:,0] + (1/N) * X[:, kappa_min] * y[kappa_min]    # check if local potential is less than zero and update weight if necessary

        w_t = W[:,0].reshape((N,1))
        ang_change = (1/3.14)*np.arccos(np.dot(np.transpose(last_w),w_t)/(np.linalg.norm(last_w)*np.linalg.norm(w_t)))

        if np.abs(ang_change) < .000000005:
            stopped.append(1)
        else:
            stopped.append(0)

        stable = 10
        stability = 0
        if n > stable:
            for i in range(stable):
                stability += stopped[n-i]
                
            if stability == stable:
                break 
        

        
    w_t = W[:,0].reshape((N,1))

    e_g = (1/3.14)*np.arccos(np.dot(np.transpose(w_t),w_star)/(np.linalg.norm(w_t)*np.linalg.norm(w_star)))


        
    # sign = np.sign(np.dot(np.transpose(W), X)) # determine the sign for each training instance
    # sign = sign.reshape((P,1))

    # correct = 0
    # for i in range(P):
    #     if y[i] == sign[i]: correct += 1
    # acc = correct/P
    return e_g , kappa_list, stopped


def plot_alpha(alpha, y, N):
    plt.plot(alpha, y, label = "N=" + str(N))
    plt.xlabel('Alpha=P/N')
    plt.ylabel('e_g(t_max)')
    #plt.ylim(0, 1)
    plt.title("Learning Curve (Averaged Over N_D=10)")

def main():

    N = [5, 20, 50]    # number of features
    
    alpha_step = 0.25
    
    alpha = np.arange(0.25, 5 + alpha_step, alpha_step).tolist()
    n_D = 10        # number of experiments to run
    n_max = 100     # max number of epochs (sweeps through data)

    # d) train on multiple randomized data sets
    def run_single_N(N):
        e_g_per_alpha = []

        for run in alpha:                       # change parameters
            P = int(run*N)                      # iterate through different P values (as ratios of alpha)   
            rep_e_g = []                          # intialize list of all Q values

            for rep in range(n_D):
                stopped = []                                     
                e_g_single, k_list, stopped = run_rosenblatt(N, P, n_max, stopped)               # run Rosenblatt for interating parameters
                rep_e_g.append(e_g_single)                               # append results
                print(stopped)
                #rep_k.append(k_list)
                #plt.hist(k_list, bins = 20)
            e_g_per_alpha.append(np.mean(rep_e_g))                          # take avarage of results
                    
            #plt.show()
        return e_g_per_alpha 

    # stopped = 0
    # e_gg, k_list, stopped = run_rosenblatt(20, 5, 1, stopped)
    # print (stopped)
    # plt.hist(k_list, bins = 10)
    # plt.show()
    
    for N_i in N:
        N_i_e_g = run_single_N(N_i)
        plot_alpha(alpha, N_i_e_g, N_i)

    plt.legend()
    plt.show()



main()
