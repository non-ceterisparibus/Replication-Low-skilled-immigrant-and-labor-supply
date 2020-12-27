
"""This module contains auxiliary functions for the creation of customized function in the main notebook."""
import pandas as pd
import numpy as np

from statsmodels.formula.api import wls
from statsmodels.stats.outliers_influence import variance_inflation_factor 

def aweighted_std(values,weights):
    """
    Analytic weights
    https://www.stata.com/support/faqs/statistics/weights-and-summary-statistics/
    """
    values=values[values>0]
    weights=weights[weights>0]
    average = np.average(values, weights=weights)
    aweights = values.shape[0]*weights/ np.sum(weights)
    variance = np.sum(aweights*(values-average)**2)/(values.shape[0]-1)
    std = np.sqrt(variance)
    return "({:.1f})".format(std)

def wls_cluster(formula,df,wt,clt):
    """
    wt      : Weight
    clt     : Cluster
    """
    model = wls(formula=formula,data=df,weights=df[wt])
    reg   =model.fit(cov_type='cluster',cov_kwds={'groups': df[clt]})
    
    return reg   