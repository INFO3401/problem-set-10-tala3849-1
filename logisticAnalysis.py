##########################
#problem set 10 
#worked with hannah and marissa 

##########################

import pandas as pd 
import csv 

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import r2_score 

import matplotlib.pyplot as plt 


class AnalysisData:
    
    def __init__(self):
        self.dataset = [] 
        self.variables = []
   
    def parseFile(self, filename): 
        self.dataset = pd.read_csv(filename)
        self.variables = []
        
        for column in self.dataset.columns.values:
            if column != "competitorname": 
                self.variables.append(column) 


    




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


############################################
#Monday & Wednesday 
#Question 1 
############################################
#Logisitc anaylsis 

class LogisticAnalysis: 
    
    def __init__(self, targetY_input): 
        self.bestX = ""
        self.targetY = targetY_input
        self.fit = ""
        
    def runSimpleAnalysis(self, data):
       
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
                    best_var = column
        self.bestX = best_var
        print(best_var, best_r2)
        
        
    
    
###########################################
#Monday & Wednesday 
#Question 2 
############################################
def runMultipleRegressrion(self,data):
   
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
                    best_var = column
    self.bestX = best_var
    print(best_var, best_r2)
        
        
    
        
             
data = AnalysisData()
data.parseFile("candy-data.csv")








#linearAnalysis = LinearAnalysis('sugarpercent')
#linearAnalysis.runSimpleAnalysis(data)

logisticAnalysis = LogisticAnalysis('chocolate')
logisticAnalysis.runSimpleAnalysis(data)



#Yes the find the same variable - the method that best fits the data is the logistic analysis b/c I get 0.42 as opposed to .10 with Linear analysis. So Logistic analysis fits the line closer  





#####################################################
#Monday & Friday 
#Question 3 
######################################################

#linear regression equation - y = mx + B 
#0.108706302017 - linear analysis number I got from running 
    
    
#simple logistic regression equation -   y = b0 + b1x   



#multiple regression equation - y = b0 + b1x1 + b2x2 + … 




    
#######################################################
#Friday Work - worked with hannah and marissa 
#######################################################



#(a) #independent - candy | dependent - sugar content | null hypothese - both contain same amount of sugar 
#Candy - categorical | Sugar - continuous 


#(b) states - independent | voters - dependent | null hypothese - both states contain the same amount of voters 
#states - categorical  | voters - continuous 

#(c) phones with battery life - independent | selling rates - dependent | all rates would be the same 
#selling rates - continous | phones w/ battery life - categorical 




