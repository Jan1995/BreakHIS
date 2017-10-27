#!/bin/env/python
# -*- encoding: utf-8 -*-
"""
===============================================================================
BreakHIS Data Extraction
===============================================================================
author=hal112358
"""
from __future__ import print_function
import numpy as np
import argparse
import zlib
import cv2
import sys
import os

if int(sys.version[0]) ==2:
	import cPickle as pickle
else:
	import pickle

path_sep = lambda: '/' if os.name != 'nt' else '\\'

training_data = []
training_labels = []

classifications = ['MDC','MLC','MMC','MPC','BA','BF','BPT','BTA']

def cvt2tensor(data):
	return (np.reshape(data,(1,-1))).tolist()[0]

def create_1HV(i):
	hv = [0]*len(classifications)
	hv[i] = 1
	return hv

def data_dir():
	global output_path, label_path
	parser = argparse.ArgumentParser()
	parser.add_argument("--d","--data")
	parser.add_argument("--o","--output")
	parser.add_argument("--l","--labels")
	args = parser.parse_args()
	directory = args.d
	output_path = args.o
	label_path = args.l
	return directory

def create_label(image_name):
	image_char = image_name.split('-')
	tumor_info = image_char[0].split('_')
	biopsy, t_class, t_type = tumor_info[0],tumor_info[1],tumor_info[2]
	tumor ="{}{}".format(t_class,t_type)
	index = classifications.index(tumor)
	encoding_vector = create_1HV(index)
	return encoding_vector

def process_path(path):
	image_data = None
	label = None
	if os.path.exists(path):
		image_suffix = ['.png','.jpg']
		is_image = False
		for s in image_suffix:
			if path.endswith(s):
				is_image = True
				break
		if is_image:
			image_name = path.split(path_sep())[-1]
			label = create_label(image_name)
			image_data = cv2.imread(path)
			image_data = cv2.cvtColor(image_data,cv2.COLOR_BGR2GRAY)
	return image_data,label

def data_path(directory):
	for root, dirs, files in os.walk(directory,topdown=False):
		for name in files:
			path = os.path.normpath(os.path.join(root,name))
			image,label = process_path(path)
			if isinstance(image,list) and isinstance(label,list):
				image_tensor = cvt2tensor(image)
				training_data.append(image_tensor)
				training_labels.append(label)

def extract_BreakHis_data():
	data_path(data_dir())
	with open(output_path,"wb") as data_file:
		pickle.dump(training_data,data_file)
	with open(label_path,"wb") as label_file:
		pickle.dump(training_labels,label_file)

if __name__ == "__main__":
	extract_BreakHis_data()