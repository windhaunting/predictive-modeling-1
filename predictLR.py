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


from sklearn import linear_model
import pandas as pd

class predictLR:
 
    def __init__(self):
      pass


    def readInputData(self, inputFile):
        df = pd.read_csv(inputFile)
        print ("df: ", df)
        
    def trainModel(self):
        x = 1
        
        
        
    

def main():
    preLRObj = predictLR()
    inputFile = "../input_data1/train.csv"
    preLRObj.readInputData(inputFile)
    
if __name__== "__main__":
  main()