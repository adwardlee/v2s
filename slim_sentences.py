import os
import numpy as np
import string
import re

max_length = 35
max_sent_length = 0
file1 = './sents_train.txt'
file2 = './sents_val.txt'
out1 = './train_nolc.txt'
out2 = './val_nolc.txt'

lines1 = []
lines2 = []

character = set(string.punctuation)

out1_file = open(out1,'w')
out2_file = open(out2,'w')

def line_to_stream(sentence):  ###// compute all the words position in one sentence
    stream = []
    for word in sentence.split():
        if word == '--':
	   continue
        word = word.strip()
        re.split('\W+',word)
	word = word.strip('.').lower()
        stream.append(word)
    # increment the stream -- 0 will be the EOS character
    stream = ' '.join(s for s in stream)#[s for s in stream]  ####get the position of each word
    return stream



with open(file1, 'r') as sentfd:  ###read all sentences in the sentfile
    for linesent in sentfd:
        line = linesent
        line = line.strip()
        id_sent = line.split('\t')
        former = id_sent[0]
        latter = id_sent[1]
        latter = line_to_stream(latter)
	out1_file.write(id_sent[0])
	out1_file.write('\t')
	out1_file.write(latter)
	out1_file.write('\n')

with open(file2, 'r') as sentfd1:  ###read all sentences in the sentfile
    for linesent in sentfd1:
        line = linesent
        line = line.strip()
        id_sent = line.split('\t')
        former = id_sent[0]
        latter = id_sent[1]
        latter = line_to_stream(latter)
	out2_file.write(id_sent[0])
	out2_file.write('\t')
	out2_file.write(latter)
	out2_file.write('\n')

out1_file.close()
out2_file.close()
