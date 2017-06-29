import os
import numpy as np

max_length = 25
max_sent_length = 0
file1 = './backup/my_sents_train_lc.txt'
file2 = './backup/my_sents_test_lc.txt'
out1 = './truncated_train_lc.txt'
out2 = './truncated_test_lc.txt'

lines1 = []
lines2 = []


def line_to_stream(sentence):  ###// compute all the words position in one sentence
    stream = []
    for word in sentence.split():
        word = word.strip()
        stream.append(word)
    # increment the stream -- 0 will be the EOS character
    stream = [s for s in stream]  ####get the position of each word
    return stream



with open(file1, 'r') as sentfd:  ###read all sentences in the sentfile
    for linesent in sentfd:
        line = linesent
        line = line.strip()
        id_sent = line.split('\t')
        former = id_sent[0]
        latter = id_sent[1]
        latter = line_to_stream(latter)
        if len(latter) > max_sent_length:
            max_sent_length = len(latter)
        if len(id_sent) < 2:
            num_empty_lines += 1
            continue
        if len(latter) < max_length:
            lines1.append(linesent)  ####append video id and sentences to lines
    print 'the max length of sentences is %d' % max_sent_length

with open(file2, 'r') as sentfd1:  ###read all sentences in the sentfile
    for linesent in sentfd1:
        line = linesent
        line = line.strip()
        id_sent = line.split('\t')
        former = id_sent[0]
        latter = id_sent[1]
        latter = line_to_stream(latter)
        if len(latter) > max_sent_length:
            max_sent_length = len(latter)
        if len(id_sent) < 2:
            num_empty_lines += 1
            continue
        if len(latter) < max_length:
            lines2.append(linesent)  ####append video id and sentences to lines
    print 'the max length of sentences is %d' % max_sent_length

with open(out1,'w') as outfile1:
    for line in lines1:
        outfile1.write('{}'.format(line))

with open(out2,'w') as outfile2:
    for line in lines2:
        outfile2.write('{}'.format(line))
