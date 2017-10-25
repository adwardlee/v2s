import numpy as np
import csv
import operator

vocabulary = dict()
number_words = 500
with open('sent_train_file','r') as f:
    for line in f:
        line = line.strip()
        id_sent = line.split('\t')
        for word in id_sent[1].split():
            if not word in vocabulary:
                vocabulary[word] = 1
	    if word in vocabulary:
		vocabulary[word] = vocabulary[word] + 1

with open('train_most_freq_vocab_500.txt','w') as file_write:
    sorted_vocab = sorted(vocabulary.items(), key = operator.itemgetter(1), reverse=True)
    for i in xrange(number_words):
        file_write.write(sorted_vocab[i][0])
        file_write.write('\n')


