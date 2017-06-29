import numpy as np
import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rgb_file', type=str, help="caffemodel name",
					default='/media/llj/storage/processed-data-ms/rgb.txt')
parser.add_argument('--flow_file', type=str, help="video ximage list",
					default='/media/llj/storage/processed-data-ms/test_10flowx_list.txt')
parser.add_argument('--out_file', type=str, help="output fusion feature",
					default='/media/llj/storage/processed-data-ms/fusion_feature.txt')
args = parser.parse_args()

out_file = open(args.out_file,'w')
def float_line_to_stream(self, line):
    return map(float, line.split(','))

args = parser.parse_args()
rgb_video_id = list()
rgb_features = list()
flow_video_id = list()
flow_features = list()
with open(args.rgb_file,'r') as rgb_f:
	rgb_feature = list(csv.reader(rgb_f))
	for line in rgb_feature:
		rgb_features.append(line[1:])
		rgb_video_id.append(line[0])
		
with open(args.flow_file,'r') as flow_f:
	flow_feature = list(csv.reader(flow_f))
	for line in flow_feature:
		flow_video_id.append(line[0])
		flow_features.append(line[1:])
assert len(rgb_video_id) == len(flow_video_id) == len(rgb_features) == len(flow_features)
for i in xrange(len(rgb_video_id)):
	rgb_feature = float_line_to_stream(rgb_features[i])
	flow_feature = float_line_to_stream(flow_features[i])
	feature = (np.array(rgb_feature).reshape(1, len(rgb_feature)) + \
		np.array(flow_feature).reshape(1, len(flow_feature)))/2
	
	out_file.write(rgb_video_id[i]+','+ ','.join(str(x) for x in feature.tolist())+ '\n')
