import numpy as np
import pickle


class Super_CONT_POS_SYL_RYM:
    '''
    Class implementation supervised learning using part of speech tags 
    (POS), syllables (SYL), and rhyming (RYM) data. The training data 
    is in a continuous (CONT) format such that all lines and sonnets 
    are flattened.
    '''
    def __init__(self, datafile, strength=2):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state. 

        Arguments:
        datafile: str: datafile.txt to open and use
        strength: int: strength of rhyming matrix to use. correlates to strictness 
        of what is considered a rhyme.

        Parameters:
        L:      Number of states.
        D:      Number of observations.
        A:      The LxL transition matrix. Uninitialized.
        O:      The LxD observation matrix. Uninitialized.
        X:      Observations for training
        Y:      States for training
        Xmap:   Mapping of observations in X to words.
        rhyme_mat: DxD matrix of rhyming matrix between observations. 
        Contains data values to multiply against original probabilities.
            
        '''
        # laod Xmap, X, and Y from picled objects
        self.Xmap, self.X, self.Y = pickle.load(open('cache/Xm.X.Y_' + datafile + '.pkl', 'rb'))
        self.L = max(self.Y)
        self.D = len(set(self.X))
        # load rhyme and syllables from pickled objects
        self.get_rhyme_and_syllables(0.01, datafile, strength)


    def get_rhyme_and_syllables(self, no_rhyme, datafile, strength):
        '''
        no_rhyme: the relative probability of a word that doesn't rhyme when 
        it should. relative to 1.
        datafile: datafile.txt to open and use
        strength: strength of rhyming matrix to use. correlates to strictness 
        of what is considered a rhyme.
        '''
        # load rhyme_mat and syllables for pickled objects and verfiy shape
        print('loading rhyming with strength:', strength)
        prefix = '2' if strength == 2 else ''
        self.rhyme_mat = pickle.load(open('cache/' + prefix + 'rhyme_mat_' + datafile + '.pkl', 'rb'))
        assert(self.rhyme_mat.shape[0] == self.D)
        # change values of rhyme_mat to no_rhyme if not a rhyme (0) 
        # use floating probabilities instead of np.uint8 used in saving
        self.rhyme_mat = self.rhyme_mat.astype(np.float32)
        # rescale [0, 1] -> [no_rhyme, 1]
        self.rhyme_mat[self.rhyme_mat != 0.] = 1.
        self.rhyme_mat = (self.rhyme_mat * ((1. - no_rhyme) / no_rhyme)) + 1.
        self.rhyme_mat *= no_rhyme
        # load syllables and verfiy shape
        self.syllables = pickle.load(open('cache/syllables_' + datafile + '.pkl', 'rb'))
        assert(self.syllables.shape[0] == self.D)



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

        # generating 140 words is enough to parse into a sonnet of
        # 140 syllables
        M = 140

        # create list of states using weighted probabilities from self.A
        state = np.random.randint(self.L)
        states = [state]
        for m in range(0, M - 1):
            weighted_probs = self.A[state]
            # check discrepencies in sum(prob_weight), which should be ~1:
            # print('total prob weight (should be ~1):', np.sum(weighted_probs))
            weighted_probs /= np.sum(weighted_probs)  # probs sum to 1
            state = np.random.choice(self.L, 1, p=weighted_probs)[0]
            states.append(state)

        # create list of observations that would fit these states using 
        # weighted probabilities from self.O, rhmying data from 
        # self.rhyme_mat and syllable data from self.syllables
        lines = []
        line = []
        previous_obs = 0  # keep track of last observation to not repeat
        s = 0  # index in states to draw state from
        rhyming_obs = np.zeros(14)
        while len(lines) < 14:  # create sonnet of 14 lines
            syllables_left = 10  # each line contains exactly 10 syllables
            while syllables_left > 0:
                # get weighted probabilites, while zeroing words w/ too many syllables
                weighted_probs = self.O[state] * (self.syllables <= syllables_left)
                # apply rhyminging probabilities if last word in sentence and 
                # correct (see online) line in sonnet
                if (((len(lines) // 2) + 1) % 2  == 0):
                    weighted_probs[(self.syllables == syllables_left)] *= (self.rhyme_mat[int(rhyming_obs[len(lines) - 2])])[(self.syllables == syllables_left)]
                elif len(lines) == 13:
                    weighted_probs[self.syllables == syllables_left] *= (self.rhyme_mat[int(rhyming_obs[len(lines) - 1])])[(self.syllables == syllables_left)]
                weighted_probs[previous_obs] = 0  # don't duplicate words. does okay w/o this
                weighted_probs /= np.sum(weighted_probs)  # probs sum to 1
                obs = np.random.choice(self.D, 1, p=weighted_probs)[0]
                line.append(self.Xmap[obs])  # add word for emission string
                syllables_left -= self.syllables[obs]
                previous_obs = obs  # keep track of last observation/word
                # get next state 
                s += 1
                state = states[s] 
            rhyming_obs[len(lines)] = obs # store last word to rhyme with later
            lines.append(line)
            line = []
        # join into a single string
        emission = '\n'.join(' '.join(line) for line in lines)  
        return emission


supervised_model = Super_CONT_POS_SYL_RYM('both', strength=1)
supervised_model.supervised_learning()   # learn the model
# generate emissions from model
for i in range(0, 3):
    print(supervised_model.generate_emission())
    print('\n')


