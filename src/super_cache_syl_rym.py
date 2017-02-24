import nltk
import re
import numpy as np
import pickle

'''
pre-processes datafile using saved pickled file 'Xm.X.Y_****' 
to create cached rhyme_mat** and syllable** objects that 
contain a matrix of word to word rhmying probability (0 or 1) 
and a list of syllables counts for all words. follows the 
format of X and the words in Xm to create these objects.
'''

# We did not write rhyme() and sylco(). It was found online courtesy of 
# http://eayd.in/?p=232
def rhyme(inp, level=3):
    '''
    param inp: str: string to find rhymes of
    param level: int: strictness of rhyming. 3 looks good. 2 is reasonable for ML.
    return: set(strs): set of rhymes belonging to inp
    '''
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)


def sylco(word):
    '''
    param word: str: word to find number of syllables of
    return: int: number of syllables in word
    '''
    word = word.lower()
 
    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables
 
    exception_add = ['serious','crucial']
    exception_del = ['fortunately','unfortunately']
 
    co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
    co_two = ['coapt','coed','coinci']
 
    pre_one = ['preach']
 
    syls = 0 #added syllable number
    disc = 0 #discarded syllable number
 
    #1) if letters < 3 : return 1
    if len(word) <= 3 :
        syls = 1
        return syls
 
    #2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)
 
    if word[-2:] == "es" or word[-2:] == "ed" :
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
            else :
                disc+=1
 
    #3) discard trailing "e", except where ending is "le"  
 
    le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']
 
    if word[-1:] == "e" :
        if word[-2:] == "le" and word not in le_except :
            pass
 
        else :
            disc+=1
 
    #4) check if consecutive vowels exists, triplets or pairs, count them as one.
 
    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
    disc+=doubleAndtripple + tripple
 
    #5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]',word))
 
    #6) add one if starts with "mc"
    if word[:2] == "mc" :
        syls+=1
 
    #7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1
 
    #8) add one if "y" is surrounded by non-vowels and is not in the last word.
 
    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1
 
    #9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.
 
    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1
 
    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1
 
    #10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
 
    if word[-3:] == "ian" : 
    #and (word[-4:] != "cian" or word[-4:] != "tian") :
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else :
            syls+=1
 
    #11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.
 
    if word[:2] == "co" and word[2] in 'eaoui' :
 
        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
            syls+=1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
            pass
        else :
            syls+=1
 
    #12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.
 
    if word[:3] == "pre" and word[3] in 'eaoui' :
        if word[:6] in pre_one :
            pass
        else :
            syls+=1
 
    #13) check for "-n't" and cross match with dictionary to add syllable.
 
    negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]
 
    if word[-3:] == "n't" :
        if word in negative :
            syls+=1
        else :
            pass  
 
    #14) Handling the exceptional words.
 
    if word in exception_del :
        disc+=1
 
    if word in exception_add :
        syls+=1    
 
    # calculate the output
    return numVowels - disc + syls


def check_rhyming(datafile='shakespeare', level=1):
    ''' (deprecated)
    param datafile: str: name of datafile to use
    return none: allows user to verify the rhyming process of make_rhyming()
    '''
    # open datafile
    with open('../data/' + datafile + '.txt') as f:
        lines = f.readlines()
    # reformat lines by removing wierd characters and short lines
    new_lines = []
    for l, line in enumerate(lines[:50]):
        # convert line to list of words
        words = nltk.word_tokenize(line.lower())
        # remove lines shorter than 3 words
        if len(words) < 3:
            continue
        # remove wierd characters
        words = [word for word in words if word not in 
            [',', ':', '.', ';', '!', '?', ')', '(', "'", "'s"]]
        new_lines.append(words)

    # print stanza rhyming process
    for i1, i2 in [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 13]]:
        w1, w2 = new_lines[i1][-1], new_lines[i2][-1]
        print('rhymes', w1, w2,('yes' if w1 in rhyme(w2, level) else 'no'))

def make_rhyming(datafile='shakespeare', strength=2):
    '''
    param datafile: str: name of datafile to use
    param strength: int: strength of rhyming matrix to use. correlates to 
    strictness of what is considered a rhyme.
    return: none: creates and saves rhyming matrix as for use by super2.py
    '''
    Xmap, X, Y = pickle.load(open('cache/Xm.X.Y_' + datafile + '.pkl', 'rb'))
    size = len(set(X))
    # create matrix that maps observations (nums) to observations (nums) 
    rhyme_mat = np.zeros((size, size), dtype=np.uint8)
    # for every pair of observations value in matrix as 1 if they rhyme
    for d1 in range(0, size):
        d1_rhymes = set(rhyme(Xmap[d1], strength)) # set has O(1) lookup
        print('processing %d/%d' % (d1, size))
        for d2 in range(0, d1): # to d1: don't process duplicate pairs or itself
            rhyme_mat[d1, d2] = (1 if Xmap[d2] in d1_rhymes else 0)
    rhyme_mat += rhyme_mat.T  # add transpose to fill in duplicate pairs
    
    # save rhyme matrix using pickle
    prefix = '2' if strength == 2 else ''
    with open('cache/' + prefix + 'rhyme_mat_' + datafile + '.pkl', 'wb') as f:
        pickle.dump(rhyme_mat, f, pickle.HIGHEST_PROTOCOL)
    return rhyme_mat

def analyze_rhyming(datafile='shakespeare'):
    '''
    param datafile: str: name of datafile to use
    return: none: verifies make_rhyming() by printing number of rhymes per 
    word and total number of rhymes for all words.
    '''
    rhyme_mat = pickle.load(open('cache/rhyme_mat_' + datafile + '.pkl', 'rb'))
    for row in rhyme_mat:
        print('num rhymes for word:', np.sum(row))
    print('total num pairs rhymes:', np.sum(rhyme_mat) // 2)

def make_syllables(datafile='shakespeare'):
    '''
    param datafile: str: name of datafile to use
    return syllables: list[int]: list of number of syllables for every word 
    in X. has the length of X. it is saved/pickled.
    '''
    Xmap, X, Y = pickle.load(open('cache/Xm.X.Y_' + datafile + '.pkl', 'rb'))
    syllables = []
    size = len(set(X))
    for d in range(0, size):
        syllables.append(sylco(Xmap[d]))  # use sylco to get number of syllables
    # save list using pickle
    with open('cache/syllables_' + datafile + '.pkl', 'wb') as f:
        pickle.dump(np.array(syllables), f, pickle.HIGHEST_PROTOCOL)
    return syllables




