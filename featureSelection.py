#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 18:38:42 2017

@author: fubao
"""

#feature selection module
#show feature selection methods here
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from itertools import combinations, chain


            
#Filter use variance statistics to do feature selection, select ones bigger than bigger than threshold
def featureSelectionFilterVariance01(df, threshold):
    #filter method
    #use variance:
    varSelector = VarianceThreshold(threshold)        # threshold = 0.1;  select features variances bigger than threshold 
    
    varSelector.fit_transform(df)
    #idxs = varSelector.get_support(indices=True)
    #print("featureSelection01 varArray: ", idxs)
    
    df = df.iloc[:, varSelector.get_support(indices=False)]
    #return varArray, idxs
    print("featureSelectionVariance01 df: ", df.shape)
    return df


#Filter use linear correlation statistics to do feature selection;  suitable only for linear relationship
def featureSelectionFilterCorrelation02(df, threshold):
    #get column list
    # 1st calculate the feature pair; 2nd use X -> y;   df contains X and y
    print ("featureSelectionFilterCorrelation02 df column", df.columns, len(df.columns))
    y = df['Purchase']      # keep the column name df.iloc[:,-1]       #y column
    #dfX = df.iloc[:, :-1]      #create a view , not to delete;  df.drop(df.columns[[-1,]], axis=1, inplace=True)
    #df.drop(df.columns[[-1,]], axis=1, inplace=True) df.drop(df.index[2])
    
    df.drop(df.columns[-1], axis=1, inplace=True)         
    correlations = {}
    columns = df.columns.tolist()
    
    for col_a, col_b in combinations(columns, 2):
        correlations[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])

    dfCorr = pd.DataFrame.from_dict(correlations, orient='index')
    dfCorr.columns = ['PCC', 'p-value']
    
    print ("featureSelectionFilterCorrelation02 result1: ", dfCorr, dfCorr.shape)
    
    #select one of the features in the feature pair with the absolute PCC larger than threshold
    dfCorr = dfCorr[dfCorr['PCC'] <= threshold]
    print ("featureSelectionFilterCorrelation02 result2: ", dfCorr.index.tolist(), dfCorr.shape)
    
    colLsts = [f.split("__") for f in dfCorr.index.tolist()]
    cols = list(set(chain(*colLsts)))
    print ("cols:    ", len(cols))
    df = pd.concat([df[cols], y], axis=1)
    print("featureSelectionFilterCorrelation02 df: ", df.shape, y) #df['Purchase'])
    
#use mutual information to do feature selection.
#calculate all feature pairs with normalized mutual information(NMI); too cost for big feature set
#calculate feature vs predict value for regression model, filter too low NMI value
def featureSelectionFilterMutualInfo03(df):
    x = 1
    