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

#http://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/

from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from commons import get_series_ids

class predictLR:
 
    
    def __init__(self):
      pass


    #simply clean and dummy coding;    no effective feature selection methods used
    def readPreprocessData(self, inputFile):
        df = pd.read_csv(inputFile)
        #print ("readCleanInputData df head: ", df.head(), df.dtypes)
        
        #for col in df.columns:
        #    print ("readCleanInputDataunique: ", col, len(df[col].unique()))
            #print ("val_count:", df[col].value_counts())
        
        #print ("describe: ", df.describe())
        print (df.head(3))
        #print("readCleanInputData: pur: ", df['Purchase'].describe())
        #df['Purchase'].plot.bar()
        
        #show NaN ratio
        #for col in df:
        #    print ('readCleanInputData col: ' , col, ": ", df[col].value_counts(dropna=False))
        print (" NaN ratio: ", len(df), (len(df)-df.count())/len(df))
        
        #show unique 
        print ('Product_Category_3 len: ', len(df.Product_Category_3.unique()))
        
        
        # limit to categorical data using df.select_dtypes()
        X = df.select_dtypes(include=[object])
        #df.shape
        print ("X head: ", X.head(3))
        
        #http://www.ritchieng.com/machinelearning-one-hot-encoding/
        # 1. INSTANTIATE
        # encode labels with value between 0 and n_classes-1.
        le = preprocessing.LabelEncoder()
        # 2/3. FIT AND TRANSFORM
        # use df.apply() to apply le.fit_transform to all columns
        X_2 = X.apply(le.fit_transform)
        print ("X_2 head: ", X_2.head(3))

        #OneHotEncoder
        #Encode categorical integer features using a one-hot aka one-of-K scheme.
        # 1. INSTANTIATE
        enc = preprocessing.OneHotEncoder()
        
        # 2. FIT
        enc.fit(X_2)
        
        # 3. Transform
        onehotlabels = enc.transform(X_2)
        print ("onehotlabels.shape: ", onehotlabels.shape)

        '''
        #drop column Product_Category_3 due to too many nan
        df = df.drop(['Product_Category_3'], axis=1) 
        
      
        #crete dummy variable   #or df factorize();    vs scikit-learn preprocessing.LabelEncoder
        dfGender = pd.get_dummies(df['Gender'])
        df = df.drop(['Gender'], axis=1) 
        
        dfMarital = pd.get_dummies(df['Marital_Status'])
        df = df.drop(['Marital_Status'], axis = 1)
        
        dfAge = pd.get_dummies(df['Age'])
        df = df.drop(['Age'], axis=1) 
        
        dfCity = pd.get_dummies(df['City_Category'])
        df = df.drop(['City_Category'], axis=1) 
        #dfProd1 = pd.get_dummies(df['Product_Category_1'])
        #dfProd2 = pd.get_dummies(df['Product_Category_2'])

        df = df.join([dfGender, dfCity, dfMarital, dfAge])
        
        #tranfer to float for object type
        #df = df.apply(pd.to_numeric, errors='ignore')
        #df["Product_ID"] = get_series_ids(df["Product_ID"])
        #df['Product_ID'] = df['Product_ID'].str.replace(',','').astype(np.int64)
        #df['Product_ID'] = df['Product_ID'].astype('str').apply(lambda x: x[1:]).astype(int)         # it works
        labelsProd, levels  = pd.factorize(df['Product_ID'])         #not correct here?  remove this feature or use dummy 
        #df['Product_ID'] = pd.to_numeric(df['Product_ID'])
        df['Product_ID'] = labelsProd
        #drop na
        labelsStYear, levels  = pd.factorize(df['Stay_In_Current_City_Years'])        #can not factorize? affect test accuracy
        df['Stay_In_Current_City_Years'] = labelsProd
        
        df = df.dropna()
        
        #print ("readCleanInputData nan2: ", len(df), (len(df)-df.count())/len(df))
        #print ("readCleanInputData df labels: ", labels, levels)
        
        #Transforms features by scaling each feature to a given range.


        # Standardize features by removing the mean and scaling to unit variance
        #hey might behave badly if the individual feature do not more or less
        #look like standard normally distributed data
        scaled_features = StandardScaler().fit_transform(df)
        #print("standard scaler: ", df.mean_)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

        #Transforms features by scaling each feature to a given range.
        scaled_features = MinMaxScaler().fit_transform(df)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
        #print ("after preprocessing df head2: ", df.head(), df.dtypes)
        
        '''
        return df
    
    
    #use correlation statistics to do feature selection
    def featureSelection01(self, inputFile):
        x = 1
        
    
    #analyse and visualize data before training
    def plotExploreDataPreTrain(self, df):
        
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
        
        '''
        print ("matplotlib.__version__: ", matplotlib.__version__)

        df.plot(x='Age', y='Purchase', style = 'o')
        plt.xlabel('Age')
        plt.show()
        
        plt.figure()
        plt.scatter(df['Occupation'], df['Purchase'])
        plt.xlabel('Occupation')

        plt.show()
        '''
        plt.figure()
        df['Purchase'].plot()

        '''
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
        fig.subplots_adjust(hspace=1.0) ## Create space between plots
        df.plot(x='Age', y='Purchase', ax = axes[0,0])
        df.plot(x='Occupation', y='Purchase', ax = axes[0,1], style = 'o')
        # Add titles
        axes[0,0].set_title('Age')
        axes[0,1].set_title('Occupation')
        '''
    
    #analyse and visualize data after training
    def plotExploreDataAfterTrain(self, y_pred, y_true):
        #plot residual plot
        plt.rcParams['agg.path.chunksize'] = 10000
        print ("len y_pred, y_true: ", len(y_pred), len(y_true))
        #plt.scatter(x_test, y_test,  color='black')
        plt.plot(y_pred, y_true-y_pred, color='blue', linewidth=3)
        plt.show()
        
    #use data df to train model;  data[-1] is the train ground truth y values
    def trainModelData(self,df):
        trainX = df.drop(['Purchase'], axis=1) 
        
        trainY = df.Purchase
        lm = linear_model.LinearRegression(normalize=True, n_jobs=2)

        lm.fit(trainX, trainY)
        print("Estimated intercept: ", lm.intercept_, "coeff len: ", len(lm.coef_))
        
        #construct a data frame that contains features and estimated coefficients.
        featureCoeffDf = pd.DataFrame(list(zip(trainX.columns, lm.coef_)), columns = ["feature", "estimatedCoeffcients"])
        print ("trainModel,featureCoeffDf df  ", featureCoeffDf)
        print ("trainModel r2 score: ", lm.score(trainX, trainY))
        
        y_pred = lm.predict(trainX)
        #get mean squared error
        print (" means squared error: ", mean_squared_error(trainY, y_pred))          
        #plot residual
        #self.plotExploreDataAfterTrain(y_pred, trainY)
        return lm
    
    #split original input data to tain and test data to do cross validation etc
    def validationModel(self, df):
        #use cross validation; split the data 8:2 ratio?
        x = 1
    
    
    #final test output for the previous trained model
    def testOutputModelFinal(self, testInFile, lm):
        df = self.readCleanInputData(testInFile)
        testX = df                               #.drop(['Purchase'], axis=1) 
        #testYReal = df['Purchase']
        testYEstimate = lm.predict(testX)
        print ("testOutputModel testYEstimate : ", testYEstimate)
        
        
def main():
    preLRObj = predictLR()
    inputFile = "../input_data1/train.csv"
    df = preLRObj.readPreprocessData(inputFile)
    #preLRObj.plotExploreData(df)
    #lm = preLRObj.trainModelData(df)
    
    #testInFile = "../input_data1/test.csv"
    #preLRObj.testOutputModel(testInFile, lm)
if __name__== "__main__":
  main()
