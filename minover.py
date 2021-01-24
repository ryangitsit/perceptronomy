
import numpy as np
import matplotlib.pyplot as plt
import math
import random


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
def run_minover(N, P, n_max, stopped, stable_crit):

    # generate data
    X = np.random.normal(0, 1, (N, P))                          # randomly generated P examples of N dimeions
    w_star = init_w_star(N)                                     # initialize teacher vector 
    y = np.transpose(np.sign(np.dot(np.transpose(w_star), X)))  # define labes on basis of teacher making correct classifications via sign(w* \cdot X)
    W =  np.zeros((N, 1))                                       # initialize student vector of N-dimensions as zero vector
    E_list = np.zeros((1,P))                                    # Initialize vector to hold local potentials

    # sequential training
    for n in range(n_max):                                      # epoch loop 
        last_w = W[:,0]
        for p in range(P):                                      # feature vector loop
            
            E = np.dot(np.transpose(W[:, 0]), X[:, p]) * y[p]   # dot weight vector with feature vector (for pth example) and multiply with the label

            E_list[0,p] = E                                     # keep all local potentials in a list

            kappa_min = np.argmin(E_list)                       # stability is defined by nearest example to threshold (lowest local potential), store the index of this example

            W[:,0] = W[:,0] + (1/N) * X[:, kappa_min] * y[kappa_min]    # update weight with lowest E-index sample vector times label

        # measure the angular change between the previous weight vector and the current one
        w_t = W[:,0].reshape((N,1))
        ang_change = (1/3.14)*np.arccos(np.dot(np.transpose(last_w),w_t)/(np.linalg.norm(last_w)*np.linalg.norm(w_t)))

        # if the angular change of the last iteration is approximately zero, append 1, else 0
        if np.abs(ang_change) < .00001:
            stopped.append(1)
        else:
            stopped.append(0)

        stability = 0   # Counter for zero-angular-change of stable_crit many past iterations

        if n > stable_crit:
            for i in range(stable_crit):
                stability += stopped[n-i]   # accumulate past zero-change iterations
                
            if stability == stable_crit:    # if no change has occured for stable_crit iterations, break
                break 


    # measure generalized error of training process
    e_g = (1/3.14)*np.arccos(np.dot(np.transpose(w_t),w_star)/(np.linalg.norm(w_t)*np.linalg.norm(w_star))) 

    return e_g , stopped


def plot_alpha(alpha, y, N, stable_crit):
    plt.plot(alpha, y, label = "N=" + str(N))
    plt.xlabel(r'$\alpha=P/N$')
    plt.ylabel(r'$\epsilon_g(t_{max})$')
    condition = "Strong" if stable_crit == 15 else "Weak"
    plt.title(r"Learning Curve (Averaged Over $N_D=10$): " + condition + " Stopping")

def plot_SC(alpha, y, N, stable_crit):
    plt.plot(alpha, y, label = "Stopping Criterion = " + str(stable_crit))
    plt.xlabel(r'$\alpha=P/N$')
    plt.ylabel(r'$\epsilon_g(t_{max})$')
    plt.title(r"Learning Curve N = 20 (Averaged Over $N_D=10$)")

def main():

    #N = [5, 20, 50]                                                # number of features
    alpha_step = 0.25                                               # Alpha step size
    alpha = np.arange(0.25, 5 + alpha_step, alpha_step).tolist()    # Alpha stepping across given range
    n_D = 10                                                        # number of experiments to run on new datasets
    n_max = 100                                                     # max number of epochs (sweeps through data)

    # Define required consecutive iterations of approx zero angular change to meet stopping criterion
    #stable_crit = 1    # 15 denotes strong, 5 denotes weak stopping criterion

    # train on multiple randomized data sets for different values of alpha
    def run_single_N(N, SC):
        e_g_per_alpha = []

        for run in alpha:                      
            P = int(run*N) # iterate through different P values (as ratios of alpha)   
            rep_e_g = []   # initialize a list to store generalized errors for different experiments

            for rep in range(n_D): # iterate of n_D datasets
                stopped = []                                     
                e_g_single, stopped = run_minover(N, P, n_max, stopped, SC)             # run minover for interating parameters
                rep_e_g.append(e_g_single)                                              # append results
                #print(len(stopped))                                                    # print e_g to observe stopping
            e_g_per_alpha.append(np.mean(rep_e_g))                                      # take avarage of results
                    
        return e_g_per_alpha                                                            # return an e_g for each alpha


    # choice "features": iterates MinOver algorithm over different values of N-feature length, with a fixed stopping criterion sc = 10
    # choice "stopping": iterates MinOver algorithm over different valuess of stopping criterion, with a fixed feature length N = 20
    exp = "features" 
    
    if exp == "features":
        N = [5, 20, 50] 
        stable_crit = 15
        for N_i in N:                                       # iterate for different values of N (and thus P and alpha)
            N_i_e_g = run_single_N(N_i, stable_crit)        # run the minover algorithm for a single N accross multiple datasets for diff alphas
            plot_alpha(alpha, N_i_e_g, N_i, stable_crit)    # plot the results!

    if exp == "stopping":
        SC = [1,5,10,50]                        # list of different requirements of consecutive stable iterations (stopping criterion)
        for sc_i in SC:                         # iterate over these different criterion
            N_i_e_g = run_single_N(20, sc_i)    # run the minover algorithm for a single stopping criterion accross multiple datasets for diff alphas
            plot_SC(alpha, N_i_e_g, 20, sc_i)   # plot the results!

    plt.legend()
    plt.show()



main()
