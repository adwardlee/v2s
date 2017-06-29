import numpy as np
import os
from caption import *
from framefc7_text_to_hdf5_data import *

rgb_score = np.load('rgb_score.npy').item()
flow_score = np.load('flow_score.npy').item()
alpha = 0.75
VOCAB_FILE = '/media/llj/storage/processed-data-ms/yt_coco_mvad_mpiimd_vocabulary.txt'
vocab_file = VOCAB_FILE
text_out_filename = 'fusion'

whole_caption = OrderedDict()
for i in sorted(rgb_score):
    rgb_caption = rgb_score[i][0]['caption']
    rgb_all_probs = rgb_score[i][0]['all_probs']
    flow_caption = flow_score[i][0]['caption']
    flow_all_probs = flow_score[i][0]['all_probs']
    rgb_prob = rgb_score[i][0]['prob']
    flow_prob = flow_score[i][0]['prob']
    rgb_length = len(rgb_caption)
    flow_length = len(flow_caption)
    sent_length = 0
    one_caption = []
    caption = {}
    if rgb_length <= flow_length:
        for num in xrange(rgb_length):
            one_prob = alpha * rgb_all_probs[num] + (1 - alpha) * flow_all_probs[num]
            if num == 0:
                caption['caption']=[int(one_prob.argsort()[-1:])]
                caption['prob']=[one_prob[int(one_prob.argsort()[-1:])]]
            else:
                caption['caption'].append(int(one_prob.argsort()[-1:]))
                caption['prob'].append(one_prob[int(one_prob.argsort()[-1:])])
        if caption['caption'][-1] != 0:
            for num1 in xrange(rgb_length, flow_length):
                caption['caption'].append(flow_caption[num1])
                caption['prob'].append(flow_prob[num1])
    else:
        for num in xrange(flow_length):
            one_prob = alpha * rgb_all_probs[num] + (1 - alpha) * flow_all_probs[num]
            if num == 0:
                caption['prob']=[one_prob[int(one_prob.argsort()[-1:])]]
                caption['caption']=[int(one_prob.argsort()[-1:])]
            else:
                caption['prob'].append(one_prob[int(one_prob.argsort()[-1:])])
                caption['caption'].append(int(one_prob.argsort()[-1:]))
        if caption['caption'][-1] != 0:
            for num1 in xrange(flow_length, rgb_length):
                caption['caption'].append(rgb_caption[num1])
                caption['prob'].append(rgb_prob[num1])
    caption['gt'] = rgb_score[i][0]['gt']
    #caption['prob'] = rgb_score[i][0]['prob']
    caption['source'] = rgb_score[i][0]['source']
    one_caption.append(caption)
    whole_caption[i] = one_caption
FRAMEFEAT_FILE_PATTERN = '/home/llj/caffe/models/inception/out_feature_file'
SENTS_FILE = None  # optional#'./my_sents_test_lc.txt'
filenames = [(FRAMEFEAT_FILE_PATTERN,
                      SENTS_FILE)]
fsg = fc7FrameSequenceGenerator(filenames, BUFFER_SIZE,
                                vocab_file, max_words=MAX_WORDS, align=False, shuffle=False,
                                pad=False, truncate=False)
eos_string = '<EOS>'
# add english inverted vocab
vocab_list = [eos_string] + fsg.vocabulary_inverted
text_out_types = to_text_output(whole_caption, vocab_list)
for strat_type in text_out_types:
    text_out_fname = text_out_filename + strat_type + '.txt'
    text_out_file = open(text_out_fname, 'a')
    text_out_file.write(''.join(text_out_types[strat_type]))
    text_out_file.close()