import nltk
import pickle

'''
pre-processes datafile and saves as a pickled list object containing 
[Xm.X.Y]. Xm maps observation-numbers as ints to words. X is a list of 
observation-numbers as ints for every word in the data flattened into one 
dimension (while remaining ordered). Y is a list of part of speech state 
values as ints that correspond to the obersvation numbers in X.
'''


for datafile in ['shakespeare', 'spenser']:
    print('processing', datafile)
    # open text as [lines]
    with open('../data/' + datafile + '.txt') as f:
        lines = f.readlines()

    XY_tagged = []  # list[tuple(observation as word, state as part of speech tag)]
    # get part of speech tagging for each word and save into XY_tagged
    for l, line in enumerate(lines):
        words = nltk.word_tokenize(line.lower())  # split line into words
        # remove lines shorter than 3 words
        if len(words) < 3:
        	continue
        # remove wierd characters
        words = [word for word in words if word not in 
            [',', ':', '.', ';', '!', '?', ')', '(', "'", "'s"]]
        XY_tagged += nltk.pos_tag(words)  # tag words by part of speech

    X, Y_tagged = zip(*XY_tagged)  # unpack [(t1, t2)] -> [t1], [t2]

    # now we have X and Y_tagged. so generate state-numbers that correspond to 
    # part of speech tagging and save into Y.

    Y_tagged_set = set(Y_tagged)  # distinct POS
    Y_tagged_map = {k: v for v, k in enumerate(Y_tagged_set)}  # dict[POS]=state
    print('total number of distinct states is', len(Y_tagged_set))
    Y = [Y_tagged_map[y] for y in Y_tagged]  # Y <- lst[states]

    # map observations to numbers and save into X_nums. save mapping as X_map
    X_set = set(X)  # distinct words
    X_map = {k: v for v, k in enumerate(X_set)}  # dict[word] = observation-num
    print('total number of distinct observations is', len(X_set))
    X_nums = [X_map[x] for x in X]  # X_nums <- lst[obersvation-nums]
    X_map = dict(enumerate(X))  # prepare for mapping in reverse direction

    print('length of observations/states:', len(X), len(Y))

    # save X_map, X_nums, and Y as a pickled list
    with open('Xm.X.Y_' + datafile + '.pkl', 'wb') as f:
        pickle.dump([X_map, X_nums, Y], f, pickle.HIGHEST_PROTOCOL)


