import sys
import numpy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

class PCActy(object)
	
	
	def fit (data,num_features):
		meanVals=numpy.mean(data,axis=0) #figure out mean
		normalized_data=(data-meanVals)/numpy.std(data) #calculate normalized value
		covData=numpy.cov(normalized_data,rowvar=0) 
		e,EV = eigsh(covData,num_features,which='LM')
		return normalized_data,EV
	
	def transform (data,normalized_data,EV):
		lowDData=numpy.dot(normalized_data,EV)
		recData=numpy.dot(lowDData,EV.T)*numpy.std(data)+numpy.mean(data,axis=0)
