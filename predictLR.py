#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:03:30 2017

@author: fubao
"""

#black friday purchase amount prediction
#use linear regression here to look at the performance
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
# http://nbviewer.jupyter.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/01%20-%20Lesson.ipynb
#http://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
#https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
#http://www.ritchieng.com/machinelearning-one-hot-encoding/
#https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish

import pandas as pd
import numpy as np
import matplotlib

from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

from sklearn.model_selection import GridSearchCV

from commons import get_series_ids

from preprocessing import dummyEncodeMethod1, dummyEncodeMethod2, preprocessNANMethod, preprocessTransform, preprocessScaler
from featureSelection import featureSelectionFilterVariance01, featureSelectionFilterCorrelation02, featureSelectionFilterMutualInfo03,featureSelectionWrapperKBest

from visualizePlot import plotExploreDataPreTrain, plotResidualAfterTrain, plotCommonAfterTrain

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
        print ("original shape ", df.head(3), df.shape)
        
        #print("readCleanInputData: pur: ", df['Purchase'].describe())
        #df['Purchase'].plot.bar()
        #print("np mean: ",df.Purchase.describe())
        #show NaN ratio
        #for col in df:
        #    print ('readCleanInputData col: ' , col, ": ", df[col].value_counts(dropna=False))
        print (" NaN ratio: ", len(df), (len(df)-df.count())/len(df))
        
        #show unique 
        #print ('Product_Category_3 len: ', len(df.Product_Category_3.unique()))
       
        #drop column Product_Category_3 due to too many nan
        df.drop(['User_ID', 'Product_ID', 'Product_Category_3'], axis=1, inplace=True) 
       
        #fill na or drop na
        print ("drop first shape ", df.shape)
        
        print ("dropna ttdf: ",  df.shape)
        #df = dummyEncodeMethod1(df)
        df = dummyEncodeMethod2(df)
        #print ("after preprocessing df head2: ", df.describe())      
        print ("dummyEncodeMethod2 df: ", df.shape, df.head(2))
        
        array = preprocessNANMethod(df)
        print ("dropna df shape ", df.shape)
       

        array = preprocessTransform(array) 

        #Array = self.preprocessScaler(array)               #scaling is sensitive to linear regression
        
        df = pd.DataFrame(array, index=df.index, columns=df.columns)
        print ("after preprocessing df head2: ", df.shape, df.head())          #df.dtypes
        
        #begin feature slection 
        df = featureSelectionFilterVariance01(df, 0)
        
        print ("after feature selection df head3: ", df.shape, df.head())

        #df = featureSelectionFilterCorrelation02(df, 0.8)
        
        #df = featureSelectionFilterMutualInfo03(df, 0.8)
        df = featureSelectionWrapperKBest(df, 10)
        return df
    

    #use data df to train general linear regression model;  data[-1] is the train ground truth y values
    def trainLinearRegModelData(self, df):
        
        trainX = df.drop(['Purchase'], axis=1)         #inplace false all X data
        
        trainY = df.Purchase
        lm = LinearRegression(normalize=True, n_jobs=2)

        lm.fit(trainX, trainY)
        print("Estimated intercept: ", trainX.shape, lm.intercept_, "coeff len: ", len(lm.coef_))
        
        #construct a data frame that contains features and estimated coefficients.
        featureCoeffDf = pd.DataFrame(list(zip(trainX.columns, lm.coef_)), columns = ["feature", "estimatedCoeffcients"])
        print ("trainModel,featureCoeffDf df  ", featureCoeffDf)
        print ("trainModel r2 score: ", lm.score(trainX, trainY))
        
        y_pred = lm.predict(trainX)
        #get mean squared error
        print (" means squared error: ", mean_squared_error(trainY, y_pred))    #mean squared error
        print (" root means squared error: ", mean_squared_error(trainY, y_pred)**0.5)     # root mean squared error

        #plot residual
        #plotExploreDataAfterTrain(y_pred, trainY)
        
   
        #get mean squared error
       # print (" after lasso means squared error: ", mean_squared_error(trainY, y_pred))    #mean squared error
       # print (" after lasso root means squared error: ", mean_squared_error(trainY, y_pred)**0.5)     # root mean squared error
        return lm
    
        
     #use regularization lasso linear regression model;  data[-1] is the train ground truth y values
     #lasso does not work in this data, why?
    def trainLinearRegModelDataWithLasso(self, df):
                
        trainX = df.drop(['Purchase'], axis=1)         #all X data
        
        trainY = df.Purchase
        
        #cross validation
        #self.crossValidation(trainX, trainY, lm)t
        cf = self.crossValidationGridLasso(trainX, trainY)
        
        #after lasso feature selection
        alpha = 0.1          # 0.2...1.  cf.best_params_['alpha']        
        
        lmLasso = Lasso(alpha = alpha, normalize=True)

        lmLasso.fit(trainX, trainY)
        print("Estimated intercept: ", lmLasso.intercept_, "coeff len: ", len(lmLasso.coef_))
        
        #construct a data frame that contains features and estimated coefficients.
        featureCoeffDf = pd.DataFrame(list(zip(trainX.columns, lmLasso.coef_)), columns = ["feature", "estimatedCoeffcients"])
        print ("trainModel,featureCoeffDf df  ", featureCoeffDf)
        print ("trainModel r2 score: ", lmLasso.score(trainX, trainY))
        
        y_pred = lmLasso.predict(trainX)
        #get mean squared error
        print ("Lasso means squared error: ", mean_squared_error(trainY, y_pred))    #mean squared error
        print ("Lasso root means squared error: ", mean_squared_error(trainY, y_pred)**0.5)     # root mean squared error

        #plot residual
        #plotExploreDataAfterTrain(y_pred, trainY)
        
        return lmLasso
    
    #genearl cross validation 
    def crossValidation(self, x, y, lm):
        #X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        n_folds = 5
        kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)
        p = np.zeros_like(y)
        for train_index, test_index in kf.split(x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x.loc[train_index], x.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            #print ("crossValidation train shape: ", x_train.shape, x_test.shape)
            #print ("crossValidation test shape: ", y_train.shape, y_test.shape)
            lm.fit(x_train, y_train)
            p[test_index] = lm.predict(x_test)
         
        rmse_cv = np.sqrt(mean_squared_error(p, y))   #root mean square error
        print('RMSE on 5-fold CV: {:.2}'.format(rmse_cv))
        #plotCommonAfterTrain(p, y)
        #plotResidualAfterTrain(p, y)
        
    #lasso exhaustive grid cross validation ;  ElasticNet;  lasso could also help feature selection
    def crossValidationGridLasso(self, x, y):
        lasso = Lasso(random_state=0)
        alphas = np.logspace(-4, 0, 30)
        tuned_parameters = [{'alpha': alphas}]              #here alphas 
        n_folds = 5
        clf = GridSearchCV(lasso, tuned_parameters, n_jobs=4, cv=n_folds, refit=True)   #4 core
        clf.fit(x, y)
        scores = clf.cv_results_['mean_test_score']
        scores_std = clf.cv_results_['std_test_score']
        
        print('crossValidationGridLasso score', clf.best_params_, clf.best_score_)

               
        '''
        plt.figure().set_size_inches(8, 6)
        plt.semilogx(alphas, scores)

        # plot error lines showing +/- std. errors of the scores
        std_error = scores_std / np.sqrt(n_folds)

        plt.semilogx(alphas, scores + std_error, 'b--')
        plt.semilogx(alphas, scores - std_error, 'b--')

        # alpha=0.2 controls the translucency of the fill color
        plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel('alpha')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        plt.xlim([alphas[0], alphas[-1]])
        '''
        
        return clf
    
    
    
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
    #preLRObj.plotExploreDataPreTrain(df)
    lm = preLRObj.trainLinearRegModelData(df)
     #lmLasso = preLRObj.trainLinearRegModelDataWithLasso(df)
    
    #test final test data
    #testInFile = "../input_data1/test.csv"
    #preLRObj.testOutputModel(testInFile, lm)
if __name__== "__main__":
  main()
