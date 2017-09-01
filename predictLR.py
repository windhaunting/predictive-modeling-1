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
from numpy import log1p

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

from sklearn.model_selection import GridSearchCV

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
        print("np mean: ",df.Purchase.describe())
        #show NaN ratio
        #for col in df:
        #    print ('readCleanInputData col: ' , col, ": ", df[col].value_counts(dropna=False))
        print (" NaN ratio: ", len(df), (len(df)-df.count())/len(df))
        
        #show unique 
        print ('Product_Category_3 len: ', len(df.Product_Category_3.unique()))
        
        #drop column Product_Category_3 due to too many nan
        df = df.drop(['User_ID', 'Product_ID'], axis=1) 
        df = df.drop(['Product_Category_3'], axis=1) 
        
        #fill na or drop na

        #df = self.dummyEncodeMethod1(df)
        df = self.dummyEncodeMethod2(df)
        #print ("after preprocessing df head2: ", df.describe())      
           
        array = self.preprocessNANMethod(df)
        print ("dropna df shape ", df.shape)
        
        array = self.preprocessTransform(array) 

        #Array = self.preprocessScaler(array)               #scaling is sensitive to linear regression
        
        df = pd.DataFrame(array, index=df.index, columns=df.columns)
        print ("after preprocessing df head2: ", df.shape, df.head())          #df.dtypes
        
        #begin feature slection 
        df = self.featureSelectionVariance01(df)
        
        print ("after feature selection df head3: ", df.shape, df.head())

        return df
    
    
    #use scikit-learn label and oneHotEncoder  -- method 1 
    # --has bug  ValueError: setting an array element with a sequence when call this function

    def dummyEncodeMethod1(self, df):
       # limit to categorical data using df.select_dtypes()
        X = df.select_dtypes(include=[object])
        #df.shape
        print ("X head: ", X.head(3))
        
        # encode labels with value between 0 and n_classes-1.
        le = LabelEncoder()
        # use df.apply() to apply le.fit_transform to all columns
        X_2 = X.apply(le.fit_transform)
        print ("X_2 head: ", X_2.head(3))

        #*** drop previous categorical columns
        #X.columns
        df.drop(X.columns, axis=1, inplace=True)

        #OneHotEncoder
        #Encode categorical integer features using a one-hot aka one-of-K scheme.
        enc = OneHotEncoder()
        onehotlabels = enc.fit_transform(X_2)
        
        dfX2 = pd.DataFrame(onehotlabels, index=range(0,onehotlabels.shape[0]), columns = range(0,onehotlabels.shape[1]), dtype=object)       #random index here
        
        df2 = pd.concat([df, dfX2], axis=1)        
        print ("onehotlabels.shape: ", onehotlabels.shape[1], df.shape, df2.shape, type(df2))
        return df2
        
    
    #use pands get_dummies  -- method 2
    def dummyEncodeMethod2(self, df):
        
       # limit to categorical data using df.select_dtypes()
        categoDf = df.select_dtypes(include=[object])
        #df.shape
        print ("categoDf head: ", categoDf.head(3))
        dfDummy = pd.get_dummies(categoDf)      #crete dummy variable or df factorize();    vs scikit-learn preprocessing Encoder

        #drop previous categorical columns
        df1 = df.drop(categoDf, axis=1) 

        df = pd.concat([df1, dfDummy], axis=1)

        return df
        
    #process missing value here
    def preprocessNANMethod(self, df):
        #drop rows with all NaN
        df = df.dropna(axis=0, how='all', thresh=2)       #Keep only the rows with at least 2 non-na values:
        imputedArray = Imputer(missing_values="NaN", strategy='mean').fit_transform(df)
        
        #df = pd.DataFrame(imputedArray, index=df.index, columns=df.columns)
        #fill na
        
        return imputedArray
    
    #data transform, polynomial, log or exponential. etc.
    def preprocessTransform(self, array):
        
        transArray = FunctionTransformer(log1p).fit_transform(array)

        return transArray
        
    def preprocessScaler(self, array):
        #Transforms features by scaling each feature to a given range.
        # Standardize features by removing the mean and scaling to unit variance
        stanScalerArray = StandardScaler().fit_transform(array)
        #print("standard scaler: ", df.mean_)
        #df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

        #Transforms features by scaling each feature to a given range.
        rangeScalerArray = MinMaxScaler().fit_transform(stanScalerArray)
        
        return rangeScalerArray
        
    #use variance statistics to do feature selection
    def featureSelectionVariance01(self, df):
        #filter method
        #use variance:
        varSelector = VarianceThreshold()                #threshold=0.1) select features variances bigger than threshold 
        
        varSelector.fit_transform(df)
        #idxs = varSelector.get_support(indices=True)
        #print("featureSelection01 varArray: ", idxs)
        
        df = df.iloc[:, varSelector.get_support(indices=False)]
        #return varArray, idxs
        return df
    
    #use correlation statistics to do feature selection
    def featureSelectionCorrelation02(self, df):
        x = 2
        
    
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
    
    #analyse and visualize data after training; residualPlot
    def plotResidualAfterTrain(self, y_pred, y_true):
        #plot residual plot
        plt.rcParams['agg.path.chunksize'] = 10000
        #print ("len y_pred, y_true: ", len(y_pred), len(y_true))
        #plt.scatter(x_test, y_test,  color='black')
        plt.plot(y_pred, y_true-y_pred, color='blue', linewidth=3)  #
        plt.show()
        
    #plot general figure common AfterTrain
    def plotCommonAfterTrain(self, y_pred, y_true):
        plt.scatter(y_true, y_pred)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")

    #use data df to train general linear regression model;  data[-1] is the train ground truth y values
    def trainLinearRegModelData(self, df):
        trainX = df.drop(['Purchase'], axis=1)         #all X data
        
        trainY = df.Purchase
        lm = LinearRegression(normalize=True, n_jobs=2)

        lm.fit(trainX, trainY)
        print("Estimated intercept: ", lm.intercept_, "coeff len: ", len(lm.coef_))
        
        #construct a data frame that contains features and estimated coefficients.
        featureCoeffDf = pd.DataFrame(list(zip(trainX.columns, lm.coef_)), columns = ["feature", "estimatedCoeffcients"])
        print ("trainModel,featureCoeffDf df  ", featureCoeffDf)
        print ("trainModel r2 score: ", lm.score(trainX, trainY))
        
        y_pred = lm.predict(trainX)
        #get mean squared error
        print (" means squared error: ", mean_squared_error(trainY, y_pred))    #mean squared error
        print (" root means squared error: ", mean_squared_error(trainY, y_pred)**0.5)     # root mean squared error

        #plot residual
        #self.plotExploreDataAfterTrain(y_pred, trainY)
        
   
        #get mean squared error
       # print (" after lasso means squared error: ", mean_squared_error(trainY, y_pred))    #mean squared error
       # print (" after lasso root means squared error: ", mean_squared_error(trainY, y_pred)**0.5)     # root mean squared error
        return lm
    
    
    #def train
    
     #use regularization lasso linear regression model;  data[-1] is the train ground truth y values
    def trainLinearRegModelDataWithLasso(self, df):
                
        trainX = df.drop(['Purchase'], axis=1)         #all X data
        
        trainY = df.Purchase
        
             #cross validation
        #self.crossValidation(trainX, trainY, lm)t
        cf = self.crossValidationGridLasso(trainX, trainY)
        
        #after lasso feature selection
        alpha = 0          # cf.best_params_['alpha']        
        
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
        #self.plotExploreDataAfterTrain(y_pred, trainY)
        
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
        #self.plotCommonAfterTrain(p, y)
        #self.plotResidualAfterTrain(p, y)
        
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
    #preLRObj.plotExploreData(df)
    lm = preLRObj.trainLinearRegModelData(df)
    lmLasso = preLRObj.trainLinearRegModelDataWithLasso(df)
    #testInFile = "../input_data1/test.csv"
    #preLRObj.testOutputModel(testInFile, lm)
if __name__== "__main__":
  main()
