########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Avishek Dutta
# Description:  Set 5
########################################

import string

class Utility:
    '''
    Utility for the problem files.
    '''

    def __init__():
        pass

    @staticmethod
    def load(filename):
        '''
        Returns:
            observations:   Sequences of observations, i.e. a list of lists.
                            Each entry is a line
            obs_map:        A hash map that maps each observation to a word
        '''

        # keeps track of the observations parsed
        observations = []
        # maps words to observations
        word_map = {}
        # keeps a count of how many words we have seen
        observation_counter = 0

        with open(filename) as f:
            for line in f:
                line = line.strip()

                if line == '' or line.isdigit():
                    continue

                line = ''.join(l for l in line if l not in string.punctuation)

                word_seq = line.split()

                for word in word_seq:
                    if word not in word_map:
                        word_map[word] = observation_counter
                        observation_counter += 1

                observation_seq = []

                for word in word_seq:
                    observation_seq.append(word_map[word])

                observations.append(observation_seq)

        # map an observation to a word
        obs_map = {v: k for k, v in word_map.items()}


        return observations, obs_map
