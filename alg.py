import csv
import numpy as np
import pandas as pd
import urllib
from numpy import array
import sklearn
from sklearn.naive_bayes import *
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

def ReadIn(filename):
	num = 0
	lines = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			if num >=1:
				line = line.strip("\n").strip("\r")
				name = line.split(';')
				lines.append(name)
			num += 1
	
	return lines
	
	
def RemoveNulls(csv_file):
	for row in csv_file:
		for i, x in enumerate(row):
			if len(x) < 1:
				x = row[i] = 0
	
	return csv_file
	

def Classify(gross):
	i = 0
	for item in gross:
		if item <= 5000000:
			gross[i] = 1
		elif item> 5000000 and item <= 30000000:
			gross[i] = 2
		elif item > 30000000 and item <= 80000000:
			gross[i] = 3
		elif item > 80000000:
			gross[i] = 4
		i+=1;
	return gross
	
	
def NormCalc(x, min_val, max_val):
	return (x - min_val) / (max_val - min_val)
	
	
def Normalise(csv_file):
	values = []
	gross = []
		
	for line in csv_file:   
	    for i in range(2,10):
		    line[i] = float(line[i])
		for j in range(3,10):
		    line[i] = NormCalc(line[i], MinCalc(j, csv_file), MaxCalc(j, csv_file))
		
		values.append(line[3::])
		gross.append(line[2])
		
	return values, gross

	
def SplitCount(gross):
	one = 0
	two = 0
	three = 0
	four = 0
	
	for i in gross:
		if i == 1:
			one+=1
		elif i == 2:
			two+=1
		elif i == 3:
			three+=1
		elif i == 4:
			four+=1
	return one, two, three, four
	

def main():
	csv_file = ReadIn('imdb_data.csv')			
	csv_file_parsed = RemoveNulls(csv_file)						
	values, gross = Normalise(csv_file_parsed)	
	gross = Classify(gross)
	
	print "split:", SplitCount(gross)
		
	# Naive Bayes
	values_train, values_test, gross_train, gross_test = train_test_split(values, gross, test_size = .15, random_state = 20)

	#Bernoulli
	BernNB = BernoulliNB()
	BernNB.fit(values_train, gross_train)
	gross_expect = gross_test
	gross_predict = BernNB.predict(values_test)
	print "BernoulliNB:", accuracy_score(gross_expect, gross_predict)
	
	#Gaussian
	Gauss = GaussianNB()
	Gauss.fit(values_train, gross_train)
	gross_expect = gross_test
	gross_predict = Gauss.predict(values_test)
	print "GaussianNB:", accuracy_score(gross_expect, gross_predict)
	
	#Multinomial
	Multi = MultinomialNB()
	Multi.fit(values_train, gross_train)
	gross_expect = gross_test
	gross_predict = Multi.predict(values_test)
	print "MultinomialNB:", accuracy_score(gross_expect, gross_predict)
		

if __name__ == '__main__':
	main()
