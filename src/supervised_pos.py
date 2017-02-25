"""
line by line data with part of speech tagging
"""

import string
import nltk

def load_pos():
    '''
    Returns:
        observations:   Sequences of observations, i.e. a list of lists.
                        Each entry is a line
        parts_of_speed: states for each of the observation based on parts of 
                        speed
        obs_map:        A hash map that maps each observation to a word
        pos_map:        maps the state to its part of speech description
    '''

    # keeps track of the observations parsed
    observations = []
    # maps words to observations
    word_map = {}
    # keeps a count of how many words we have seen
    observation_counter = 1

    # keeps track of parts of speech parsed
    parts_of_speech = []
    # part of speech map
    pos_map = {}
    # keeps a count of how many parts of speech we have seen
    pos_counter = 1

    with open('../data/shakespeare.txt') as f:
        for line in f:
            line = line.strip()

            if line == '' or line.isdigit():
                continue

            line = ''.join(l for l in line if l not in string.punctuation)

            words_pos = nltk.pos_tag(line.lower().split())

            # populate the word map
            for word, pos in words_pos:
                if word not in word_map:
                    word_map[word] = observation_counter
                    observation_counter += 1

                if pos not in pos_map:
                    pos_map[pos] = pos_counter
                    pos_counter += 1

            observation_seq = []
            pos_seq = [] 

            # convert each word to a number
            for word, pos in words_pos:
                observation_seq.append(word_map[word])
                pos_seq.append(pos_map[pos])


            # add the words as a line
            observations.append(observation_seq)
            parts_of_speech.append(pos_seq)

    # map an observation to a word
    obs_map = {v: k for k, v in word_map.items()}
    # map a state to a part of speed
    pos_map = {v: k for k, v in pos_map.items()}


    return observations, parts_of_speech, obs_map, pos_map