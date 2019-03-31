#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
#  eigenbrain1.py
#  
#  Copyright 2019 Marcelo Ruan Moura Araújo <marceloruan.moura@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
# 
'''
		                 uuuuuuu
		             uu$$$$$$$$$$$$uu
		          uu$$$$$$$$$$$$$$$$$$uu
		         u$$$$$$$$$$$$$$$$$$$$$$u
		        u$$$$$$$$$$$$$$$$$$$$$$$$u
		       u$$$$$$$$$$$$$$$$$$$$$$$$$$u
		       u$$$$$$$$$$$$$$$$$$$$$$$$$$u
		       u$$$$$$"   "$$$$"   "$$$$$$u
		       "$$$$"      u$$u       $$$$"
		        $$$u       u$$u       u$$$
		        $$$u      u$$$$u      u$$$
		         "$$$$uu$$$$   $$$uu$$$$"
		          "$$$$$$$$"   "$$$$$$$"
		            u$$$$$$$$u$$$$$$$u
		             u$"$"$"$$"$"$"$u      
		  uuu        $$u$ $ $ $ $u$$       uuu
		 u$$$$        $$$$$u$u$u$$$       u$$$$
		  $$$$$uu      "$$$$$$$$$"     uu$$$$$$
		u$$$$$$$$$$$uu    """""    uuuu$$$$$$$$$$
		$$$$"""$$$$$$$$$$uuu   uu$$$$$$$$$"""$$$"
		 """      ""$$$$$$$$$$$uu ""$"""
		           uuuu ""$$$$$$$$$$uuu
		  u$$$uuu$$$$$$$$$uu ""$$$$$$$$$$$uuu$$$
		  $$$$$$$$$$""""           ""$$$$$$$$$$$"
		   "$$$$$"                      ""$$$$""
		     $$$"                         $$$$"

				   ▀██▀█ █▀██▀   ▀██▀▀▀▀▄
				    ██ ▀▄▀ ██     ██    █
				    ██  █  ██     ██▀▀▀▀▄ 
				   ▄██▄   ▄██▄ ▄ ▄██    █▄
'''

import time
ini =time.time() 
import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import random
import sys
import itk
import csv
import pandas as pd 

def load_images_per_directory(path):
	file_dic = {}
	sub_directory = []
	image_list = []
	id_list = []
	for subdir, dirs, files in sorted(os.walk(path)):
		for file in sorted(files):
			split_list = file.split("_")
			id = split_list[0]         
			id_list.append(id)
			if file.find("nii") > -1:
				if subdir in file_dic:
					file_dic[subdir].append(os.path.join(subdir, file))                    
				else:
					sub_directory.append(subdir)
					file_dic[subdir] = [os.path.join(subdir, file)]
	return file_dic, sub_directory, id_list

def load_images(path):
	file_dic = {}
	sub_directory = []
	image_list = []
	id_list = []
	for subdir, dirs, files in os.walk(path):
		for file in files:
			split_list = file.split("_")
			id = split_list[0]
			#A = file.split(".nii")[0].split("_")[1]           
			id_list.append(id)
			#id_proc.append(A)
			if file.find("nii") > -1:
				if subdir in file_dic:
					file_dic[subdir].append(os.path.join(subdir, file))                    
					PixelType = itk.ctype("float")
					image = (itk.imread(os.path.join(subdir, file),PixelType))
					InputType = type(image)
					input_dimension = image.GetImageDimension()
					OutputType = itk.Image[itk.F, input_dimension]
					castFilter = itk.CastImageFilter[InputType, OutputType].New()
					castFilter.SetInput(image)
					image = itk.GetArrayFromImage(castFilter)
					image_list.append(image)
				else:
					sub_directory.append(subdir)
					file_dic[subdir] = [os.path.join(subdir, file)]
					PixelType = itk.ctype("float")
					image = (itk.imread(os.path.join(subdir, file),PixelType))
					InputType = type(image)
					input_dimension = image.GetImageDimension()
					OutputType = itk.Image[itk.F, input_dimension]
					castFilter = itk.CastImageFilter[InputType, OutputType].New()
					castFilter.SetInput(image)
					image = itk.GetArrayFromImage(castFilter)
					#image = image/np.amax(image) #normalize 0-1
					image_list.append(image)
	image_shape = image_list[0].shape
	return image_list, file_dic, sub_directory, id_list, image_shape





def save_images_all(image_list,id_list,k):
	for i in range(len(image_list)):
		# Convert back to itk, data is copied
		brain_img = image_list[i]
		brain_img = np.float32(brain_img)
		brain_img = itk.GetImageFromArray(brain_img)
		InputType = type(brain_img)
		dimension = brain_img.GetImageDimension()
		OutputType = itk.Image[itk.F, dimension]
		castFilter2 = itk.CastImageFilter[InputType, OutputType].New()
		castFilter2.SetInput(brain_img)
		itk.imwrite(brain_img, "%s_%d.nii"%(id_list[i],k))


def calculate_image_vector_matrix(image_list):
	# =============================================================
	#               calculating BRAIN VECTOR
	# =============================================================
	val_1, val_2, val_3 = image_list[0].shape
	rows = val_1 * val_2 * val_3
	image_vec_matrix = np.zeros((rows, len(image_list)))
	i = 0
	for image in image_list:
		vector = image.flatten()
		vector = np.asmatrix(vector)
		image_vec_matrix[:, i] = vector # Each colomn is a BRAIN
		i += 1
	return image_vec_matrix



def calculate_main_brain(image_vec_matrix):
	# =============================================================
	#              calculating MEAN BRAIN VECTOR
	# =============================================================
	mean_brain_vec = np.mean(image_vec_matrix, axis=1)
	mean_brain_vec = np.array(mean_brain_vec)
	# =============================================================
	#               calculating MEAN BRAIN IMAGE
	# =============================================================
	mean_brain_img = mean_brain_vec.reshape((sizex, sizey, sizez)) #reshape to save
	mean_brain_img = np.float32(mean_brain_img)
	mean_brain_itk = itk.GetImageFromArray(mean_brain_img)
	InputType = type(mean_brain_itk)
	dimension = brain_img.GetImageDimension()
	OutputType = itk.Image[itk.F, dimension]
	castFilter2 = itk.CastImageFilter[InputType, OutputType].New()
	castFilter2.SetInput(mean_brain_itk)
	itk.imwrite(castFilter2, "mean_brain.nii")
	return mean_brain_vec



def get_zero_mean_brain_matrix(image_vec_matrix, mean_brain_vec):
	# =============================================================
	#      calculating zero_mean_brain_matrix and Covariance
	#==============================================================
	zero_mean_brain_matrix = np.zeros((sizex, sizey, sizey))
	zero_mean_brain_matrix_test = np.zeros((sizex, sizey, sizey)) # test this
	count = 0
	for i in range(image_vec_matrix.shape[1]):
		image_col_vector = image_vec_matrix[:,i]
		image_col_vector = (image_col_vector - mean_brain_vec)/np.std(image_col_vector,axis=0)
		zero_mean_brain_matrix_test[:,i]= image_col_vector
		if count == 0: #if this work del this 
			zero_mean_brain_matrix = image_col_vector
			count += 1
		else:
			zero_mean_brain_matrix = np.vstack((zero_mean_brain_matrix, image_col_vector))
	zero_mean_brain_matrix = zero_mean_brain_matrix.T #until here
	return zero_mean_brain_matrix



def vectorize_matrix_normalize(norm_mean_brain_matrix):
	listToNorm = []
	for i in range(0,norm_mean_brain_matrix.shape[1]):
		listToNorm.append(norm_mean_brain_matrix[:,i].reshape(sizex,sizey,sizez))
	return listToNorm


def calculate_covariance(matrix):
	return np.cov(matrix, rowvar=False)

def get_eig_vectors(zero_mean_brain_matrix, mean_brain_vec):
	covariance = calculate_covariance(zero_mean_brain_matrix)
	# =============================================================
	#      			   calculating Eigen Brains
	# =============================================================
	eig_values, eig_vectors = np.linalg.eig(covariance)
	eig_brains_vec = np.dot(zero_mean_brain_matrix, eig_vectors)
	#Optional reconstrucation
	eig_brain = [] #listed eigenbrains with 1st image with first retaind informations and secondo image with the secondo, so on.
	eig_brain = vectorize_matrix_normalize(zero_mean_brain_matrix)
	# =============================================================
	#      			   Saving Eigen Brain
	# =============================================================
	#save_images_all(eig_brain,MR_id_list)
	# =============================================================
	#      				Sorting Eigen Values and Vectors
	# =============================================================
	eig_values = sorted(eig_values, reverse=True)
	eig_vectors = -np.sort(-eig_vectors)
	return eig_vectors

def reconstruct(k, eig_vec, zero_mean_brain_matrix, mean_brain_vec):
	# =============================================================
	#              Pick first K EIGEN VECTOR
	# =============================================================
	k_eig_vec = eig_vec[0: k, :]#split k eigenvectors to each interation
	#==============================================================
	#          Calculate EIGEN BRAIN from K EIGEN VECTOR
	# =============================================================
	k_eigen_brains_vec = np.dot(zero_mean_brain_matrix, k_eig_vec.T) #32256 col and k lines to do a matrix operation the number of
	#col in the frist must be the same number os lines in the second
	k_eigen_brains_vec = np.transpose(k_eigen_brains_vec) #invert order
	train_eigen_brain_vec[k] = k_eigen_brains_vec
	# =============================================================
	#          				Calculate WEIGHTS
	# =============================================================
	k_weights = np.dot(k_eigen_brains_vec, zero_mean_brain_matrix)
	k_weights = np.transpose(k_weights)
	train_weights[k] = k_weights
	#sheetsToCSV = {'IDPatiance': MR_id_list[i],'Weights':  list(train_weights[i]) }
	#df = pd.DataFrame(sheetsToCSV, columns= ['IDPatiance', 'Weights'])
	#export_csv = df.to_csv (r'%d.csv'%(i), sep=',',index = None, header=True)
	# ==============================================================
	#          		       Perform Reconstruction (optional)
	# ==============================================================
	'''
	k_brains = [] #list with eigenbrains with K componentes plus mean
	if k == 3:
		k_reconstruction = mean_brain_vec + np.dot(k_weights, k_eigen_brains_vec)
        #we need mean BRAIN back to reconstrution
		k_reconstruction = np.transpose(k_reconstruction)
		k_brains = vectorize_matrix_normalize(k_reconstruction)
		# =============================================================
		#          		       Saving Eigen BRAIN
		# =============================================================
		#save_images_all(k_brains,MR_id_list,k)
		'''

#Weights to data test set
'''
def store_weights(k, zero_mean_brain_matrix): #K is componentes from PCA
	k_eigen_brains_vec = train_eigen_brain_vec[k] #weights from trainning set
	# =============================================================
	#          				Calculate WEIGHTS
	# =============================================================
	print(zero_mean_brain_matrix.shape)
	k_weights = np.dot(k_eigen_brains_vec, zero_mean_brain_matrix)
	k_weights = np.transpose(k_weights)
	test_weights[k] = k_weights
'''

#Reading
input_path = "/home/marcelo/Seafile/marceloGAPIS/Marcelo/Dataset/EADC/BasicTest"

file_dict, sub_dic, identify_list = load_images_per_directory(input_path)

for i in range(len(sub_dic)):
	# =========================================================================================================
	#      						Reading all images directory per directory
	# =========================================================================================================
	
	MR_image_list, MR_file_dic, MR_sub_directory, MR_id_list, shape_MR = load_images(paths) 
	sizex, sizey,sizez = shape_MR
	
	#*****************************************************************
	
	train_weights = {} #Dictonary to store weights calculate from training set
	test_weights = {} #Dictonary to store weights calculate from testin set
	train_eigen_brain_vec = {} #Dictonary to eigen eigen vector from training set
	eigen_vec_error_dict = {} #Dictonary to erro calculate between training and testing set
	#************************************************************************
	#Vectorize
	brain_image_vec_matrix = calculate_image_vector_matrix(MR_image_list)
	#Mean brain
	mean_brain_vector =   calculate_main_brain(brain_image_vec_matrix)
	#Normalize datas
	MR_mean_brain_matrix = get_zero_mean_brain_matrix(brain_image_vec_matrix, mean_brain_vector)
	#Eigenvectors
	eig_vectors = get_eig_vectors(MR_mean_brain_matrix, mean_brain_vector)
	for i in range(MR_mean_brain_matrix.shape[1]):
		reconstruct(i+1, eig_vectors, MR_mean_brain_matrix, mean_brain_vector)
	x = np.asarray(train_weights[3]) #train
	#y = np.asarray(test_weights[3]) #test 

	#model1 = KMeans(n_clusters=2)
	#model1.fit(x)
	#distance1 = model1.fit_transform(x)
	#labels1 = model1.labels_
	
	#cc1 = model1.cluster_centers_#[0]
	#cc2 = model1.cluster_centers_[1]
	
	
	#cc1 =np.asarray(cc1)
	#cc2 =np.asarray(cc2)
	print(x)
	#print(y)
	#predicted_label1 = model1.predict(y)
	#print(predicted_label1)
end =time.time()
print(end-ini)
