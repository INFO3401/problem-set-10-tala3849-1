

import pandas as pd 
import csv 

#PART B
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import r2_score 

import matplotlib.pyplot as plt 

#PART A 
class AnalysisData:
    
    def __init__(self):
        self.dataset = [] 
        self.variables = []
        
#QUESTION 1     
    def parseFile(self, filename): 
        self.dataset = pd.read_csv(filename)
        self.variables = []
        
        for column in self.dataset.columns.values:
            if column != "competitorname": 
                self.variables.append(column) 


    



#PART B 
#linear analysis 

class LinearAnalysis: 
    
    def __init__(self, targetY_input): 
        self.bestX = ""
        self.targetY = targetY_input  
        self.fit = ""
        
    def runSimpleAnalysis(self, data):
        regr = LinearRegression()
        best_r2 = -1 
        best_var = ""
        for column in data.variables:
            if column != self.targetY:
                independent_var = data.dataset[column].values
                #reshaping from 1D to 2D
                independent_var = independent_var.reshape(len(independent_var),1)
                regr = LogisticRegression()
                regr.fit(independent_var, data.dataset[self.targetY])
                pred = regr.predict(independent_var)
                r_score = r2_score(data.dataset[self.targetY],pred)
                if r_score > best_r2:
                    best_r2 = r_score
                    best_car = column
        self.bestX = best_var
        print(best_var, best_r2)
    

        
        
        
        
#string put quotes, going to be integer put 0 or -1, dictionary are {}, lists are [] 

#PART C

#Logisitc anaylsis 

class LogisticAnalysis: 
    
    def __init__(self, targetY_input): 
        self.bestX = ""
        self.targetY = targetY_input
        self.fit = ""
        
        
#QUESTION 1             
data = AnalysisData()
data.parseFile("candy-data.csv")

#QUESTION 2 
#shown in part B and C 



#WEDNESDAY WORK 
#under part B 


#linearAnalysis = LinearAnalysis('sugarpercent')
#linearAnalysis.runSimpleAnalysis(data)

logisticAnalysis = LogisticAnalysis('chocolate')
logisticAnalysis.runSimpleAnalysis(data)





