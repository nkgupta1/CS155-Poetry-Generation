import nltk
import pickle


for datafile in ['shakespeare', 'spenser']:
    print('processing', datafile)
    # open text as [lines]
    with open('../data/' + datafile + '.txt') as f:
        lines = f.readlines()

    #Y_iambic = []
    XY_tagged = []  # observations, states
    # get state for each word
    for l, line in enumerate(lines):
        # print(line)
        words = nltk.word_tokenize(line.lower())
        # remove lines shorter than 3 words
        if len(words) < 3:
        	continue
        # remove weird characters
        words = [word for word in words if word not in 
            [',', ':', '.', ';', '!', '?', ')', '(', "'", "'s"]]
        #iambic = [(w % 2) for w, word in enumerate(words)]
        #iambic[0] = 2  # first word in line gets a special state
        #Y_iambic += iambic
        XY_tagged += nltk.pos_tag(words)

    X, Y_tagged = zip(*XY_tagged)

    # now we have X, Y_tagged, and Y_iambic. so generate Y_tagged * Y_iambic  
    # states and save into Y

    Y_tagged_set = set(Y_tagged)
    # Y_iambic ranges from [0, 2]
    num_states = len(Y_tagged_set)
    Y_tagged_map = {k: v for v, k in enumerate(Y_tagged_set)}
    print('total number of distinct states is', num_states)

    Y = [Y_tagged_map[y] for y in Y_tagged]

    # map observations to numbers
    X_set = set(X)
    num_obs = len(X_set)
    X_map = {k: v for v, k in enumerate(X_set)}
    print('total number of distinct observations is', num_obs)
    X_nums = [X_map[x] for x in X]
    X_map = dict(enumerate(X))  # prepare for mapping in reverse direction


    print('length of observations/states:', len(X), len(Y))

    with open('Xm.X.Y_' + datafile + '.pkl', 'wb') as f:
        pickle.dump([X_map, X_nums, Y], f, pickle.HIGHEST_PROTOCOL)


