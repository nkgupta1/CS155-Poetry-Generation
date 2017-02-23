#!/usr/bin/env python3
########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import unsupervised_HMM
from Utility import Utility

shakespeare = '../data/shakespeare.txt'
spenser = '../data/spenser.txt'


def unsupervised_learning(n_states, n_iters):
    '''
    Trains an HMM using unsupervised learning.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    observations, obs_map = Utility.load(shakespeare)

    # Train the HMM.
    HMM = unsupervised_HMM(observations, n_states, n_iters)

    return HMM, obs_map

def sequence_generator(HMM, obs_map, k, M):
    '''
    Arguments:
        HMM:        Trained HMM.
        K:          Number of sequences to generate.
        M:          Length of emission to generate.
    '''
    print()
    print()
    # Generate k input sequences.
    for i in range(k):
        # Generate a single input sequence of length m.
        x = HMM.generate_emission(M)
        for obs in x:
            print(obs_map[obs], end=' ')
        print()

        # Print the results.
        # print("{:30}".format(x))

if __name__ == '__main__':
    HMM, obs_map = unsupervised_learning(10, 1)
    sequence_generator(HMM, obs_map, 14, 10)

