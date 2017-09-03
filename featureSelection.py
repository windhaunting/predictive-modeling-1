#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 18:38:42 2017

@author: fubao
"""

#feature selection module
#show feature selection methods here
from sklearn.feature_selection import VarianceThreshold


            
#Filter use variance statistics to do feature selection
def featureSelectionFilterVariance01(self, df):
    #filter method
    #use variance:
    varSelector = VarianceThreshold()                #threshold=0.1) select features variances bigger than threshold 
    
    varSelector.fit_transform(df)
    #idxs = varSelector.get_support(indices=True)
    #print("featureSelection01 varArray: ", idxs)
    
    df = df.iloc[:, varSelector.get_support(indices=False)]
    #return varArray, idxs
    print("featureSelectionVariance01 df: ", df.shape)
    return df


#Filter use linear correlation statistics to do feature selection;  suitable only for linear relationship
def featureSelectionFilterCorrelation02(self, df):
    x = 2
    
    
#use mutual information to do feature selection.
#calculate all feature pairs with normalized mutual information(NMI); too cost for big feature set
#calculate feature vs predict value for regression model, filter too low NMI value
def featureSelectionMutualInfo03(self, df):
    x = 1
    