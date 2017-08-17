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
import matplotlib.pyplot as plt

class predictLR:
 
    
    def __init__(self):
      pass


    def readCleanInputData(self, inputFile):
        df = pd.read_csv(inputFile)
        print ("df: ", df.head())
        print ("cnt: ", df["Gender"].value_counts())
        print(df['Purchase'].describe())
        #df['Purchase'].plot.bar()
        
        #show NaN ratio
        #for col in df:
        #    print (' col: ' , col, ": ", df[col].value_counts(dropna=False))
        print ("nan: ", len(df), (len(df)-df.count())/len(df))
        
        #show unique 
        print ('unq: ', len(df.Product_Category_3.unique()))
        
        #drop column
        df = df.drop(['Product_Category_3'], axis=1) 
        
        #drop na
        df = df.dropna()
        print ("nan2: ", len(df), (len(df)-df.count())/len(df))
        
        return df
    def plotExploreData(self, df):
        
        '''
        # specifies the parameters of our graphs
        fig = plt.figure(figsize=(18,6), dpi=1600) 
        alpha=alpha_scatterplot = 0.2 
        alpha_bar_chart = 0.55
        
        #plot many diffrent shaped graphs together 
        # distribution of histogram
        print ("cnt: ", df["Purchase"].value_counts().sort_values())
        ax1 = plt.subplot2grid((2,3),(0,0))
        df.Purchase.value_counts().plot(figsize=(15,5))
        ax1.set_xlim(-1, len(df.Purchase.value_counts()))
        plt.title("Distribution of Purchase")
        
        plt.subplot2grid((2,3),(0,1))
        df['Purchase'].plot(figsize=(15,5));


        plt.subplot2grid((2,3),(0,2))
        plt.scatter(df.Occupation, df.Purchase, alpha=alpha_scatterplot)
        plt.ylabel("Purchase")
        plt.show()
        '''
        
        df.plot(df['Age'], df['Purchase'])
        #plt.show()
        
    def trainModel(self):
        x = 1
        
        
        
    

def main():
    preLRObj = predictLR()
    inputFile = "../input_data1/train.csv"
    df = preLRObj.readCleanInputData(inputFile)
    preLRObj.plotExploreData(df)
    
if __name__== "__main__":
  main()