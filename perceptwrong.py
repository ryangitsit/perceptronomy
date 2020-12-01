import numpy as np
import matplotlib.pyplot as plt
import math

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
                W[:,0] = W[:,0] + (1/N) * X[:, p] * Y[p]          # check if local potential is less than zero and update weight if necessary

        if all([e > 0 for e in E_list]): break                      # end training if all E > 0

    # check accuracy against labels
    sign = np.sign(np.dot(np.transpose(W), X))
    sign = sign.reshape((P,1))
    print(f"sign = {np.transpose(sign)} \n Y = {Y.reshape(1,P)}")

    correct = 0
    for i in range(P):
        if Y[i] == sign[i]: correct += 1
    accuracy = correct/P
    return accuracy

print (run_rosenblatt(10, 50, 100))

