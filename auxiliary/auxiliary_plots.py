# -*- coding: utf-8 -*-
"""
This module contains auxiliary functions for the creation of plot for table using in the main notebook

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

from auxiliary.auxiliary_subdatasets import *
from auxiliary.auxiliary_functions import *


def plot_hrswork(df,x,order):
    xdict={"edulv":'Education level',"percentile":'Wage percentile'}
    
    g=df.groupby([x,"year","married"])
    married=g.apply(lambda z: np.average(z['ahrswork'],weights=z["perwt"]))
    married=married.reset_index()
    married.columns=[x,'year',"married",'ahrswork']
    
    g1=df.groupby([x,"year","child5"])
    child5=g1.apply(lambda z: np.average(z['ahrswork'],weights=z["perwt"]))
    child5=child5.reset_index()
    child5.columns=[x,'year',"child5",'ahrswork']
    
    #Plots 
    ax1=sns.catplot(x=x,y='ahrswork', col="married", hue="year",kind="bar",order=order,data=married)
    ax1.set(xlabel=xdict[x], ylabel='Annual work hours')
    
    ax2=sns.catplot(x=x,y='ahrswork', col="child5", hue="year",kind="bar",order=order,data=child5)
    ax2.set(xlabel=xdict[x], ylabel='Annual work hours')
  
    return