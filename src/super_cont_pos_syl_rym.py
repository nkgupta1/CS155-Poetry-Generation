import numpy as np
import nltk
import pickle


class Supervised:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, Xmap, X, Y):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state. 

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''
        self.L = max(Y)
        self.D = len(set(X))
        self.Y = Y
        self.Xmap = Xmap
        self.get_rhyme_and_syllables(no_rhyme=0.01)
        self.X = X


    def get_rhyme_and_syllables(self, no_rhyme=0.2, datafile='shakespeare'):
        # adjust rhyme mat to no_rhyme if not a rhyme and 1 if rhyme
        self.rhyme_mat = pickle.load(open('rhyme_mat3147' + datafile + '.pkl', 'rb'))
        assert(self.rhyme_mat.shape[0] == self.D)
        self.rhyme_mat = self.rhyme_mat.astype(np.float32)
        self.rhyme_mat[self.rhyme_mat != 0.] = 1.
        self.rhyme_mat = (self.rhyme_mat * ((1. - no_rhyme) / no_rhyme)) + 1.
        self.rhyme_mat *= no_rhyme
        self.syllables = pickle.load(open('syllables3147' + datafile + '.pkl', 'rb'))
        assert(self.syllables.shape[0] == self.D)



    def supervised_learning(self):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.

        print('learning supervised...')
        X_arr, Y_arr = np.array(self.X), np.array(self.Y)

        self.O = np.zeros((self.L, self.D), dtype=np.float64)
        for y in range(0, self.L):
            for x in range(0, self.D):
                y_correct = (Y_arr == y)
                xy_correct = (X_arr == x) * y_correct
                sum_xy = np.sum(xy_correct)
                if sum_xy != 0:
                    self.O[y, x] = sum_xy / np.sum(y_correct)

        self.A = np.zeros((self.L, self.L), dtype=np.float64)
        for y1 in range(0, self.L):
            for y2 in range(0, self.L):
                num_y1 = 0
                num_y1y2 = 0
                for yi in range(1, len(self.Y)):
                    if self.Y[yi - 1] == y1:
                        num_y1 += 1
                        if self.Y[yi] == y2:
                            num_y1y2 += 1
                if num_y1y2 != 0:
                    self.A[y1, y2] = num_y1y2 / num_y1


    def generate_emission(self):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''
        print('generating emission...')
        emission = ''

        # generate 140 words should be enough to parse into a sonnet
        M = 140

        state = np.random.randint(self.L)
        states = [state]
        for m in range(0, M - 1):
            weighted_probs = self.A[state]
            # print('total prob weight (should be ~1):', np.sum(weighted_probs))
            weighted_probs /= np.sum(weighted_probs)  # probs sum to 1
            state = np.random.choice(self.L, 1, p=weighted_probs)[0]
            states.append(state)
        # print(states)

        lines = []
        line = []
        previous_state = 0
        s = 0
        rhyming_obs = np.zeros(14)
        while len(lines) < 14:
            syllables_left = 10
            while syllables_left > 0:
                weighted_probs = self.O[state] * (self.syllables <= syllables_left)
                if (((len(lines) // 2) + 1) % 2  == 0):
                    # do rhyming on last word
                    # print('rhyming', self.Xmap[int(rhyming_obs[len(lines) - 2])])
                    # print(rhyming_obs[len(lines) - 2], self.rhyme_mat.shape)
                    # print(self.syllables.shape[0], weighted_probs.shape)
                    weighted_probs[(self.syllables == syllables_left)] *= (self.rhyme_mat[int(rhyming_obs[len(lines) - 2])])[(self.syllables == syllables_left)]
                elif len(lines) == 13:
                    # print('rhyming', self.Xmap[int(rhyming_obs[len(lines) - 1])])
                    weighted_probs[self.syllables == syllables_left] *= (self.rhyme_mat[int(rhyming_obs[len(lines) - 1])])[(self.syllables == syllables_left)]
                weighted_probs[previous_state] = 0  # don't duplicate words. does fine w/o this
                weighted_probs /= np.sum(weighted_probs)  # probs sum to 1
                # print(len(line), len(lines))
                obs = np.random.choice(self.D, 1, p=weighted_probs)[0]
                word = self.Xmap[obs]
                line.append(word)
                syllables_left -= self.syllables[obs]
                previous_state = state
                s += 1
                state = states[s]
            rhyming_obs[len(lines)] = obs  # store last word to rhyme with later
            lines.append(line)
            line = []
        emission = '\n'.join(' '.join(line) for line in lines)
        return emission


datafile = 'shakespeare'
Xmap, X, Y = pickle.load(open('Xm.X.Y_' + datafile + '.pkl', 'rb'))

supervised_model = Supervised(Xmap, X, Y)
supervised_model.supervised_learning()
for i in range(0, 3):
    print(supervised_model.generate_emission())


