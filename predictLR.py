#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:03:30 2017

@author: fubao
"""

#black friday purchase amount prediction
#using linear regression
'''
--Data Handling
Importing Data with Pandas
Cleaning Data
Exploring Data through Visualizations with Matplotlib
--Data Analysis
Linear Regression Model here
Plotting results
--Valuation of the Analysis
K-folds cross validation to valuate results locally
Output the results

'''
# pandas usage reference:
#    http://nbviewer.jupyter.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/01%20-%20Lesson.ipynb


from sklearn import linear_model
import pandas as pd
import numpy as np

class predictLR:
 
    def __init__(self):
      pass


    def readAnalyseInputData(self, inputFile):
        df = pd.read_csv(inputFile)
        print ("df: ", df.head())
        print ("cnt: ", df["Gender"].value_counts())
        print(df['Purchase'].describe())
        #df['Purchase'].plot.bar()
        
        #show NaN ratio
        #for col in df:
        #    print (' col: ' , col, ": ", df[col].value_counts(dropna=False))
        print ("nan: ", (len(df)-df.count())/len(df))
            
    def trainModel(self):
        x = 1
        
        
        
    

def main():
    preLRObj = predictLR()
    inputFile = "../input_data1/train.csv"
    preLRObj.readAnalyseInputData(inputFile)
    
if __name__== "__main__":
  main()