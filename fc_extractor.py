import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
import caffe

DEVICE_ID = 1

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, help="caffemodel name",
					default='/home/llj/caffe/models/bn_inception_rgb_init.caffemodel')
parser.add_argument('--data_file', type=str, help="video image list",
					default='/media/llj/storage/processed-data-ms/origin_per15_test_video_list.txt')
parser.add_argument('--deploy', type=str, default='/home/llj/caffe/models/inception/deploy.prototxt',
                    help="deploy prototxt")
parser.add_argument('--layer', type=str, default='global_pool', help='layer name')
parser.add_argument('--out_file', type=str, default='./yt_allframes_inception_globalpool_test_origin.txt')
parser.add_argument("--caffe_path", type=str, default='/home/llj/caffe', help='path to the caffe toolbox')
parser.add_argument("--gpus", type=int, nargs='+', default=None, help='specify list of gpu to use')
args = parser.parse_args()

print args

sys.path.append(os.path.join(args.caffe_path, 'python'))

def read_source_file(data_file):
	vid = []
	image_dirs = {}
	with open(data_file,'r') as files:
		for x in files:
			vid_name = x.strip().split(',')[0]
			if vid_name not in vid:
				vid.append(vid_name)
			if vid_name not in image_dirs:
				image_dirs[vid_name] = []
			image_dirs[vid_name].append(x.strip().split(',')[1])
	assert len(vid) == len(image_dirs)
	return vid,image_dirs


if DEVICE_ID >=0:
	caffe.set_device(DEVICE_ID)
	caffe.set_mode_gpu()
else:
	caffe.set_mode_cpu()
print "setting up caffe net"
net = caffe.Net(args.deploy, args.model_file,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
#transformer.set_raw_scale('data', 255)
transformer.set_mean('data', np.array([104, 117, 123]))
if args.layer not in net.blobs:
	raise TypeError("Invalid layer name: "+args.layer)
vid, image_dirs = read_source_file(args.data_file)
output_file = open(args.out_file,'w')
with open(args.data_file, 'r') as in_file:
	for index in vid:
		frame_num = 1
		for y in image_dirs[index]:
			frame = cv2.imread(y, cv2.IMREAD_COLOR)
			np_frame = np.array(frame)
			data = transformer.preprocess('data', np_frame)
			net.blobs['data'].reshape(1, 3, 224, 224)
			net.blobs['data'].data[...] = data
			net.forward()#feature =
			#out_feature = feature[args.layer][0].flatten()
			out_feature = net.blobs[args.layer].data[0].flatten()
			output_file.write(index + '_frame_' + str(frame_num) + ',' +\
							  ','.join(str(x) for x in out_feature.tolist()) + '\n')
			frame_num += 1
