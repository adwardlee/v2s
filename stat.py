import numpy as np
from operator import itemgetter

file1_lines = []
file2_lines = {}
a = open('generate.txt','w')
b = open('test1.txt','w')
with open('ep12.txt','r') as file1:
  for line in file1:
    line = line.strip()
    id_sent = line.split('\t')    
    file1_lines.append((id_sent[0], id_sent[1]))
file1_lines = sorted(file1_lines, key=itemgetter(0) )

with open('sents_test_lc_nopunc.txt','r') as file2:
  for line in file2:
    line = line.strip()
    id_sent = line.split('\t')
    if id_sent[0] not in file2_lines:
      file2_lines[id_sent[0]] = []
    file2_lines[id_sent[0]].append(len(id_sent[1].split()))
#file2_lines.sort(key=natural_keys)

for i in xrange(len(file1_lines)):
  vid, sent = file1_lines[i]
  a.write('%s %s\n' %(vid,len(sent.split())))

for key in sorted(file2_lines.iterkeys()):
  b.write('%s ' %key)
  num = 0
  for x in file2_lines[key]:
    #b.write('%s ' %x)
    num += x
  num = (num+0.00001)/len(file2_lines[key])
  b.write('%s\n' % num)
