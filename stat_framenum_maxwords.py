import os
import numpy as np
import glob
import argparse
import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)



parser = argparse.ArgumentParser(description="read frames in folder")
parser.add_argument("-video_dir", type = str, default = "/media/llj/storage/microsoft-corpus/video-frames")
parser.add_argument("-sent_file", type = str, default = "/media/llj/storage/processed-data-ms/my_sents_test_lc.txt")
parser.add_argument("-out_file", type = str, default = "/media/llj/storage/processed-data-ms/max_min_framenum_wordnum_test.txt")
args = parser.parse_args()

video_dir = args.video_dir
sentfile = args.sent_file
out_file = args.out_file
max_wordnum = 0
min_wordnum = 200
max_framenum = 0
min_framenum = 10000
max_video = 'a'
min_video = 'a'
max_sent = 0
min_sent = 0

for root,subfolders, filename in os.walk(video_dir):
	subfolders = natural_sort(subfolders)
	for folders in subfolders:
		files = os.listdir(os.path.join(root,folders))
		if len(files) > max_framenum:
			max_framenum = len(files)
			max_video = folders
		if len(files) < min_framenum:
			min_framenum = len(files)
			min_video = folders
with open(sentfile,'r') as thesentfile:
	for line in thesentfile:
		line = line.strip()		
		sent = line.split('\t')[1]
		number = sent.split()
		if len(number) > max_wordnum:
			max_wordnum = len(number)
			max_sent = line
		if len(number) < min_wordnum:
			min_wordnum = len(number)
			min_sent = line
with open(out_file,'w') as outp_file:
	outp_file.write('max_framenum: {}\n min_framenum: {}\n max_wordnum: {}\n \
					min_wordnum: {}\n max_video: {}\n min_video: {}\n max_sent: {}\n min_sent: {}\n'.format(\
					max_framenum, min_framenum, max_wordnum, min_wordnum, max_video, min_video, max_sent, min_sent))
	
