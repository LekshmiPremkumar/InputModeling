#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:47:44 2022

@author: lek
"""
import random
import matplotlib.pyplot as plt
from scipy.stats import skew,chisquare
#from scipy.stats import chisquare
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pylab as py
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import poisson
from scipy.stats import norm,triang
import scipy.stats as stats
import statistics


random.seed(42)

class modelmatch:
    def __init__(self,data,typeofdata,custom_pdf):
        self.data=data
        self.typeofdata=typeofdata
        self.custom_pdf=custom_pdf
        
        self.mean=np.mean(self.data)
        self.sd=np.std(self.data)
        self.lamda=1/self.mean
        
     
        
    #calculating pdf of normal distribution
    def normal_pdf(self):
        self.n_pdf=[]
        if self.custom_pdf:
            
            for i in self.data:
                prob_density = 1/(np.sqrt(2*np.pi*self.sd)) * np.exp(-0.5*((i-self.mean)/self.sd)**2)
                self.n_pdf.append(prob_density)
            return np.array(self.n_pdf)
        else:
            self.n_pdf=stats.norm.pdf(self.data,self.mean,self.sd)
            return np.array(self.n_pdf)
        
    #calculating pdf of poisson distribution
    def poisson_pdf(self):
        self.p_pdf=[]
        if self.custom_pdf:
            
            for i in self.data:
                if i <0:
                    self.p_pdf.append(0)
                else:
                    prob=self.mean**int(i) / np.math.factorial(int(i)) * np.math.exp(-1*self.mean)
                    self.p_pdf.append(prob)
            return np.array(self.p_pdf)
        else:
            self.p_pdf=stats.poisson.pmf( self.data,self.mean)
            return np.array(self.p_pdf)

    #calculating pdf of exponential distribution
    def expon_pdf(self):
        self.e_pdf=[]
        if self.custom_pdf:
            
            for i in self.data:
                #i=int(np.abs(i))
                if i <0:
                    self.e_pdf.append(0)
                else:
                    prob=self.lamda* np.math.exp(-1*self.lamda*i)
                    self.e_pdf.append(prob)
            return np.array(self.e_pdf)
        else:
            self.e_pdf=stats.expon.pdf(self.data,self.lamda)
            return np.array(self.e_pdf)
        
    def triangular_pdf(self):
        self.t_pdf=[]
        self.c=statistics.mode(self.data)
        self.a=min(self.data)
        self.b=max(self.data)
        
        if self.custom_pdf:
            for i in self.data:
                if i >=self.a and i<=self.c:
                    prob=2*(i-self.a)/((self.b-self.a)*(self.c-self.a))
                    self.t_pdf.append(prob)
                elif i>=self.c and i<=self.b:
                    prob=2*(self.b-i)/((self.b-self.a)*(self.b-self.c))
                    self.t_pdf.append(prob)
                else:
                    self.t_pdf.append(0)
            return np.array(self.t_pdf)
                    
        else:
            self.t_pdf= stats.triang.pdf(self.data, self.c, )
            return np.array(self.t_pdf)
        
        
    def generate_data(self):
        #self.custom_pdf=custom_pdf
        #calculating pdfs
        self.npdf=self.normal_pdf()
        self.ppdf=self.poisson_pdf()
        self.epdf=self.expon_pdf()
        self.tpdf=self.triangular_pdf()

    
    def KL(self,a,b):
        self.a=a
        self.b=b
        s=0
        for i in range(len(self.a)):
            if self.a[i]!=0 and self.b[i]!=0:
                t=self.a[i]*np.log(self.a[i]/self.b[i])
                s=s+t
        return s
    
    
    def sorting(self,xvals,yvals):
        myx=[]
        myy=[]
        ind=np.argsort(xvals)#argument sorting to sort the x vals and corresp pdf vals
        #print(ind)
        for i in ind:
            myx.append(xvals[i])
            myy.append(yvals[i])
        return myx,myy
 
    
    def bestfit(self):
        self.pdf, self.bin_edges = np.histogram(self.data, bins=100,density=True )
        plt.hist(self.data,bins=100,density = 1,alpha=0.4)
        
        d={}
        d['normal']=self.KL(self.pdf,self.npdf)
        d['exponential']=self.KL(self.pdf,self.epdf)
        d['triang']=self.KL(self.pdf,self.tpdf)
        dis="None of the above"
        min=d['exponential']
        x=self.data
        for i in d:
            if d[i]<=min:
                min=d[i]
                dis=i
        print(f"The closest match to the sampled data is %s distribution" %(dis))
            
        
        self.data1,self.pdf1=self.sorting(self.bin_edges[1:],self.pdf)
        plt.plot(self.data1,self.pdf1,c='r')
        plt.legend(['fitting pdf'])
        
        self.data1,self.npdf1=self.sorting(self.data,self.npdf)
        plt.plot(self.data1,self.npdf1,c='b')
        plt.legend(['normal'])
        
        self.data1,self.epdf1=self.sorting(self.data,self.epdf)
        plt.plot(self.data1,self.epdf1,c='g')
        plt.legend(['exponential'])
       

        self.data1,self.tpdf1=self.sorting(self.data,self.tpdf)
        plt.plot(self.data1,self.tpdf1,c='black')
        plt.legend(['triangular'])
        
        
        plt.legend(['fitted pdf','normal','exponential','triangular'])
        plt.xlabel('x-values')
        plt.ylabel('pdf')
    


    
    def goodnessoffit(self):
        statistic, pval=chisquare(self.obs,self.data)
        #print(statistic,pval)
        if pval<0.5:
            #fail to reject
            self.val= self.normal()
            print("The data is distributed with mean ",self.val[0] , " and variance ",self.val[1])
        else:
            print("Data is not normally distributed")
    
    def qqplot(self):
        sm.qqplot(self.data, line ='45')
        py.show()

    
            

#y=modelmatch([random.normalvariate(0.5,1) for i in range(10000)],'continuous',False)
#y=modelmatch(np.random.normal(5,1,100000) ,'continuous',True)
#y=modelmatch(np.random.triangular(-3, 0, 8, 100000),'continuous',True)
y=modelmatch(stats.expon.rvs(0.3, size=100000),'continuous',False)

y.generate_data()
y.bestfit()



