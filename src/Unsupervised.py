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

def sequence_generator(HMM, obs_map, n, l, w):
    '''
    Arguments:
        HMM:        Trained HMM.
        n:          Number of poems to generate.
        l:          Number of lines in each poem.
        w:          Number of words in each lines.
    '''
    print()
    print()
    # Generate k input sequences.
    for _ in range(n):
        for i in range(l):
            # Generate a single input sequence of length m.
            x = HMM.generate_emission(w)
            for obs in x:
                print(obs_map[obs], end=' ')
            print()
        print()

if __name__ == '__main__':
    HMM, obs_map = unsupervised_learning(30, 25)
    sequence_generator(HMM, obs_map, 5, 14, 10)

