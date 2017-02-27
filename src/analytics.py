#!/usr/bin/env python3

import nltk

import matplotlib.pyplot as plt
from graphviz import Digraph

import numpy as np

from Unsupervised import unsupervised_learning
from super_cache_syl_rym import sylco
from nltk.corpus import cmudict

# Dictionary of parts of speech
parts = {
    'CC' : 'Conj',
    'CD' : 'Noun',
    'DT' : 'Adj',
    'EX' : 'Noun',
    'IN' : 'Conj',
    'JJ' : 'Adj',
    'JJR' : 'Adj',
    'JJS' : 'Adj',
    'LS' : 'Noun',
    'MD' : 'Adverb',
    'NN' : 'Noun',
    'NNP' : 'Noun',
    'NNS' : 'Noun',
    'PDT' : 'Adj',
    'PRP' : 'Pronoun',
    'PRP$' : 'Pronoun',
    'RB' : 'Adverb',
    'RBR' : 'Adverb',
    'RBS' : 'Adverb',
    'TO' : 'Prep',
    'VB' : 'Verb',
    'VBD' : 'Verb',
    'VBG' : 'Verb',
    'VBN' : 'Verb',
    'VBP' : 'Verb',
    'VBZ' : 'Verb',
    'WDT' : 'Adj',
    'WP' : 'Pronoun',
    'WP$' : 'Pronoun',
    'WRB' : 'Adverb'
}

d = cmudict.dict()

def nsyl(word):
    ''' 
    Number of syllables.
    '''

    return str(sylco(word))

def stacked_bar_chart(list_freqs, labels, title, filename):
    '''
    Creates stacked bar charts for part of speech.

    Input is a list of frequencies for each of n states
    '''

    states = len(list_freqs)
    parts = len(labels)

    labels.reverse()
    state_names = [str(i+1) for i in range(states)]

    np_freqs = np.rot90(np.asarray(list_freqs))
    ind = np.arange(states)
    width = 0.5

    starts = [0 for i in range(states)]
    legend_cols = []
    for i in range(parts):
        vals = np_freqs[i]

        p = plt.bar(ind, vals, width, bottom=starts)

        for j in range(len(starts)):
            starts[j] += vals[j]

        legend_cols.append(p[0])

    plt.ylabel('Frequencies')
    plt.title(title)
    plt.xticks(ind, state_names)
    plt.legend(legend_cols, labels, loc='upper center', 
               bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)

    plt.show()
    return

def run(num_states):
    '''
    Runs main body of code for a given number of states.
    '''

    HMM, obs_map = unsupervised_learning(num_states, 100)
    A = HMM.A
    O = HMM.O
    D = HMM.D

    # Let's see what the most common words are
    for i in range(len(O)):
        # Keep track of top ten
        most_common = []
        for j in range(D):
            if len(most_common) <= 10:
                most_common.append((j, O[i][j]))
            else:
                min_val = 2
                min_idx = 0
                for k in range(len(most_common)):
                    if most_common[k][1] < min_val:
                        min_val = most_common[k][1]
                        min_idx = k
                if O[i][j] > min_val:
                    most_common[min_idx] = (j, O[i][j])
        most_common = sorted(most_common, key=lambda x: x[1])
        print('For state {0}, the most common words are: '.format(str(i+1)))
        for j in range(10):
            word = obs_map[most_common[j][0]]
            pos = parts[nltk.pos_tag(nltk.word_tokenize(word))[0][1]]
            print('{0}, {1}, {2}'.format(word, pos, nsyl(word)))
        print('')

    # Most common parts of speech
    list_freqs_pos = []
    labels_pos = []

    for i in range(len(O)):
        freq = {}
        graph_freqs = []
        for j in range(D):
            word = obs_map[j]
            pos = parts[nltk.pos_tag(nltk.word_tokenize(word))[0][1]]
            if pos in freq:
                freq[pos] += O[i][j]
            else:
                freq[pos] = O[i][j]

        for k in freq:
            graph_freqs.append(freq[k])

            # Also keep track of labels
            if i == 0:
                labels_pos.append(k)
        list_freqs_pos.append(graph_freqs)

    # Graph some data
    stacked_bar_chart(list_freqs_pos, labels_pos, 'Part of Speech Frequencies',
                      'results/part_of_speech_{0}'.format(str(num_states)))

    # Most common syllable lengths
    list_freqs_syl = []
    labels_syl = []

    for i in range(len(O)):
        freq = {}
        graph_freqs = []
        for j in range(D):
            word = obs_map[j]
            syl = nsyl(word)
            if syl in freq:
                freq[syl] += O[i][j]
            else:
                freq[syl] = O[i][j]

        for k in freq:
            graph_freqs.append(freq[k])

            # Also keep track of labels
            if i == 0:
                labels_syl.append(k)
        list_freqs_syl.append(graph_freqs)

    # Graph syllable data too
    stacked_bar_chart(list_freqs_syl, labels_syl, 'Syllable Frequencies',
                      'results/syllables_{0}'.format(str(num_states)))

    # Transitions likelihoods
    labels_trans = ["To State {0}".format(i+1) for i in range(len(A))]
    stacked_bar_chart(A, labels_trans, 'Transition Probabilities', 
                      'results/transitions_{0}'.format(str(num_states)))

    # Create graph
    create_graph(A, list_freqs_pos, labels_pos, 'results/graph_{0}'
                 .format(str(num_states)))

    return

def create_graph(A, list_freqs_pos, labels_pos, filename):
    '''
    Creates a graph of these attributes.
    '''

    # Find which parts of speech are uncommonly well represented
    # for each node
    np_freqs = np.asarray(list_freqs_pos)
    means = np.mean(np_freqs, axis=0)
    stds = np.std(np_freqs, axis=0)
    labels_pos.reverse()

    well_represented = []
    for i in range(len(list_freqs_pos)):
        highest_scoring = []

        # See if any are a std above expected
        for j in range(len(list_freqs_pos[0])):
            if np_freqs[i][j] >= (means[j] + (stds[j] / 2)):
                highest_scoring.append(labels_pos[j])

        # If not, choose whichever one's closest
        if len(highest_scoring) == 0:
            best = 0
            best_word = ''
            for j in range(len(list_freqs_pos[0])):
                ratio = np_freqs[i][j] / (means[j] + stds[j])
                if ratio > best:
                    best = ratio
                    best_word = labels_pos[j]
            highest_scoring.append(best_word)

        well_represented.append(highest_scoring)

    g = Digraph()
    for i in range(len(list_freqs_pos)):
        g.node(str(i+1), str(i+1) + ' : ' + ' '.join(well_represented[i]))

    trans_mean = np.mean(A)

    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] >= trans_mean:
                g.edge(str(i+1), str(j+1))

    g.render(filename, view=True)

    return


if __name__=='__main__':
    # Try for differing number of states
    for i in [5, 10, 15, 20]:
        run(i)