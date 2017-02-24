import numpy as np
import pickle


class Super_CONT_POS:
    '''
    Class implementation supervised learning using part of speech tags. 
    The training data is in a continuous (CONT) format such that all 
    lines and sonnets are flattened.
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
        Xmap:  See parameter Xmap
        X: See parameter X
        Y: See parameter Y

        Parameters:
        L:      Number of states.
        D:      Number of observations.
        A:      The transition matrix. Not initialized.
        O:      The observation matrix. Not initialized.
        X:      Observations for training
        Y:      States for training
        Xmap:   Mapping of observations in X to words.
            
        '''
        self.L = max(Y)
        self.D = len(set(X))
        self.Y = Y
        self.X = X
        self.Xmap = Xmap


    def supervised_learning(self):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        '''
        # Calculate each element of A using the M-step formulas.

        print('learning supervised...')
        X_arr, Y_arr = np.array(self.X), np.array(self.Y)

        # make observation matrix by finding probabilites of observations 
        # in self.Y given states in self.X (for every state)
        self.O = np.zeros((self.L, self.D), dtype=np.float64)
        for y in range(0, self.L):
            for x in range(0, self.D):
                y_correct = (Y_arr == y)
                xy_correct = (X_arr == x) * y_correct
                sum_xy = np.sum(xy_correct)
                if sum_xy != 0:
                    self.O[y, x] = sum_xy / np.sum(y_correct)

        # make state transition matrix by finding probabilites of states 
        # in self.Y given the previous state in self.Y (for every state)
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

        # create list of states using weighted probabilities from self.A
        state = np.random.randint(self.L)
        states = [state]

        for m in range(0, M - 1):
            weighted_probs = self.A[state]
            # check discrepencies in sum(prob_weight), which should be ~1:
            # print('total prob weight (should be ~1):', np.sum(weighted_probs))
            weighted_probs /= np.sum(weighted_probs)
            state = np.random.choice(self.L, 1, p=weighted_probs)[0]
            states.append(state)

        # create list of observations that would fit these states using 
        # weighted probabilities from self.O
        emission = ''
        line = ''
        for state in states:
            weighted_probs = self.O[state]
            weighted_probs /= np.sum(weighted_probs)
            obs = np.random.choice(self.D, 1, p=weighted_probs)[0]
            line += self.Xmap[obs] + ' '
            # delimit each line by 60 characters. then start a new line
            if len(line) > 60:
                emission += line + '\n'
                line = ''
        return emission


datafile = 'both'  # datafile.txt to read use
Xmap, X, Y = pickle.load(open('cache/Xm.X.Y_' + datafile + '.pkl', 'rb'))

supervised_model = Super_CONT_POS(Xmap, X, Y)
supervised_model.supervised_learning()  # learn the model
print(supervised_model.generate_emission(100))  # generate emission


