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
        self.X = X


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


    def generate_emission(self, M):
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

        state = np.random.randint(self.L)
        states = [state]
        for m in range(0, M - 1):
            weighted_probs = self.A[state]
            # print('total prob weight (should be ~1):', np.sum(weighted_probs))
            weighted_probs /= np.sum(weighted_probs)
            state = np.random.choice(self.L, 1, p=weighted_probs)[0]
            states.append(state)

        # make observations that would fit these states
        emission = ''
        line = ''
        for state in states:
            obs = np.random.choice(self.D, 1, p=self.O[state])[0]
            line += self.Xmap[obs] + ' '
            if len(line) > 60:
                emission += line + '\n'
                line = ''
        return emission


datafile = 'shakespeare'
Xmap, X, Y = pickle.load(open('Xm.X.Y_' + datafile + '.pkl', 'rb'))

supervised_model = Supervised(Xmap, X, Y)
supervised_model.supervised_learning()
print(supervised_model.generate_emission(100))


