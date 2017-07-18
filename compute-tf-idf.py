import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

counts = defaultdict(int)
def precook(s, counts, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    for k in xrange(1,n+1):
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


sentences =  list()
with open('/media/llj/storage/processed-data-ms/sents_train_lc_nopunc.txt','r') as f:
  for line in f:
    line = line.strip()
    id_sent = line.split('\t')
    sentences.append(id_sent[1])
for x in xrange(len(sentences)):
  precook(sentences[x], counts)
#print 'ngrams: ',counts

#tf = CountVectorizer(ngram_range=(1,4))
#X = tf.fit_transform(sentences)
#frequencies = sum(X).toarray()[0]
#print 'x: ',tf.vocabulary

# write python dict to a file
#mydict = {'a': 1, 'b': 2, 'c': 3}
output = open('msvd.p', 'wb')
pickle.dump(counts, output)
output.close()

