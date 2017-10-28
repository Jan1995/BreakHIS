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

class data_extract():

	def __init__(self):
		self.classes = ['MDC','MLC','MMC','MPC','BA','BF','BPT','BTA']
		self.training_data = []
		self.training_labels = []
		self.data_dir()
		self.sys_config()

	def sys_config(self):
		data_path = os.path.join(self.output_data_directory,'BreakHIS')
		if not os.path.isdir(data_path):
			os.mkdir(data_path)
		self.training_data_file = os.path.join(data_path,'data.p')
		self.training_label_file = os.path.join(data_path,'labels.p')

	def cvt2tensor(self,data):
		return (np.reshape(data,(1,-1))).tolist()[0]

	def create_1HV(self,i):
		hv = [0]*len(self.classes)
		hv[i] = 1
		return hv

	def data_dir(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("--d","--data")
		parser.add_argument("--o","--output")
		args = parser.parse_args()
		self.input_data_directory = args.d
		self.output_data_directory = args.o


	def create_label(self,image_name):
		image_char = image_name.split('-')
		tumor_info = image_char[0].split('_')
		biopsy, t_class, t_type = tumor_info[0],tumor_info[1],tumor_info[2]
		tumor ="{}{}".format(t_class,t_type)
		index = self.classes.index(tumor)
		encoding_vector = self.create_1HV(index)
		return encoding_vector

	def process_path(self,path):
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
				label = self.create_label(image_name)
				image_data = cv2.imread(path)
				image_data = cv2.cvtColor(image_data,cv2.COLOR_BGR2GRAY)
		return image_data,label

	def data_path(self,directory):
		for root, dirs, files in os.walk(directory,topdown=False):
			for name in files:
				path = os.path.normpath(os.path.join(root,name))
				image,label = self.process_path(path)
				if isinstance(image,np.ndarray) and isinstance(label,list):
					image_tensor = self.cvt2tensor(image)
					self.training_data.append(image_tensor)
					self.training_labels.append(label)

	def extract_BreakHis_data(self):
		self.data_dir()
		self.data_path(self.input_data_directory)
		with open(self.training_data_file,"wb") as data_file:
			pickle.dump(self.training_data,data_file)
		with open(self.training_label_file,"wb") as label_file:
			pickle.dump(self.training_labels,label_file)

if __name__ == "__main__":
	e = data_extract()
	e.extract_BreakHis_data()