#!/usr/bin/env python3

import string

if __name__=='__main__':
    words = {}

    with open('../data/shakespeare.txt', 'r') as f:
        for line in f:
            line = "".join(l for l in line if l not in string.punctuation)
            lst = line.split()
            if len(lst) <= 2:
                continue

            for word in lst:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

    word_counts = {}
    count = 0
    for key in words:
        if words[key] in word_counts:
            word_counts[words[key]] += 1
        else:
            word_counts[words[key]] = 1
        count += words[key]

    print(word_counts)
    print(count)