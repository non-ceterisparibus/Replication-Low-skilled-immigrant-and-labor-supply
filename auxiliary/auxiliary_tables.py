# -*- coding: utf-8 -*-
"""This module contains auxiliary functions for the creation of tables in the main notebook."""

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols,wls

from auxiliary.auxiliary_datasets import *
from auxiliary.auxiliary_functions import *
from auxiliary.auxiliary_subdatasets import *

def create_table1():
    """
      Creates Table 1.

    """
    data=pd.read_stata('data/census_30cities.dta')
    df=data[['age','citizen','metaread','year','educ','perwt']]
    
    edu_filter = ['N/A or no schooling','Nursery school to grade 4',
                  'Grade 5, 6, 7, or 8','Grade 9','Grade 10','Grade 11']
    citizen_filter = ['Not a citizen', 'Naturalized citizen']
    
    #Filter total-workers and lowskilled-worker
    df=df[df['age'].astype(float).between(16,64)]
    df1=df[(df['citizen'].isin(citizen_filter))&(df['educ'].isin(edu_filter))]
    
    #Create statistic table
    total=pd.crosstab(df['metaread'],df['year'],df['perwt'],colnames=" ",aggfunc='sum')
    lowskilled=pd.crosstab(df1['metaread'],df1['year'],values=df1['perwt'],colnames=" ",aggfunc='sum')
    
    new_index=(['Atlanta', 'Baltimore', 'Boston','Buffalo', 'Chicago','Cincinnati', 
                'Cleveland', 'Columbus','Dallas-Fort Worth', 'Denver-Boulder','Detroit',
                'Honolulu','Houston','Kansas City', 'Los Angeles','Miami','Milwaukee',
                'Minneapolis', 'New Orleans','New York', 'Philadelphia','Phoenix',
                'Pittsburgh','Portland', 'San Diego','San Francisco', 'Seattle',
                'St. Louis', 'Tampa','Washington'])
    
    #Edit table1
    table1 = pd.DataFrame()
    table1=lowskilled.div(total, axis = 0)
    table1.index=new_index
    table1 = table1.style.format("{:.2%}")
    table1.index.rename('City', inplace=True)
    
    return table1

def create_table2(dt):
 
    tbls = ['p025','p2550', 'p5075', 'p75100','p90100']
    
    """ Looping over dataframe percentile"""
    index=["Usual hrs.per week|H>0","Work at leat 50hrs.(percent)",
           "Work at least 60hrs.(percent)","Age","Married","Children under 18","Child under 6"]
    for i, tbl in enumerate(tbls):
        df = dt[dt[tbl]==1]
        g=df.groupby(["year"])
        table=g.apply(lambda x: pd.Series(np.average(x[["chrswork","work50","work60","age",'married','children',"child5"]],
                                                     weights=x["perwt"], axis=0),index))
        table=table.transpose()
        tbls[i]=table
        
    """ Merge sub-table """
    table2 = pd.concat([tbls[0].iloc[:,0:3],tbls[1].iloc[:,0:3],tbls[2].iloc[:,0:3],
                        tbls[3].iloc[:,0:3],tbls[4].iloc[:,0:3]], axis=1)
    table2.columns = pd.MultiIndex.from_product([['0 -25 percentile','25 -50 percentile','50 -75 percentile',
                                                  '75 -100 percentile','90 -100 percentile'],['1980','1990','2000']])
    table2=table2.style.format("{:.2f}")
    
    return table2

def create_table3(dt):
    
    #Create education filter
    dt=educ_group(dt)
    #Datalist
    edus= ['Highschool','SomeCollege', 'College', 'Master','Advanced']
    tbl = ['tb1','tb2', 'tb3', 'tb4','tb5']
    
    """Looping over dataframe percentile"""
    index=["Usual hrs.per week|H>0","Labor force participation","Work at leat 50hrs.(percent)",
           "Work at least 60hrs.(percent)","Married","Children under 18","Child under 6"]
    for i, edu in enumerate(edus):
        df=dt[dt['edulv']==edu]
        g=df.groupby(["year"])
        table0=g.apply(lambda x: pd.Series(np.average(x[["lflw","work50","work60",'married','children',"child5"]],
                                                      weights=x["perwt"], axis=0)))
        idx = ~np.isnan(df["chrswork"])
        table1=g.apply(lambda x: pd.Series(np.average(x.chrswork[idx],weights=x.perwt[idx],axis=0)))
        table =pd.concat([table1,table0],axis=1)
        tbl[i] =table.transpose()
        tbl[i].index=index
        
    #merge sub-table
    table3 = pd.concat([tbl[0].iloc[:,0:3],tbl[1].iloc[:,0:3],
                        tbl[2].iloc[:,0:3],tbl[3].iloc[:,0:3],tbl[4].iloc[:,0:3]], axis=1)
    table3.columns = pd.MultiIndex.from_product([['At most high school grad','Some College','College grad',
                                              "Master's degree",'Professional degree or PhD'],
                                             ['1980','1990','2000']])
    table3=table3.style.format("{:.2f}")
    return table3

def create_table4(dt):
    #Create wage_percentile variables 
    cols=['p025','p2550','p5075','p75100','p90100']
    for col in cols:
        dt.loc[:,col]=0
    dt.loc[dt['hwage']>dt['hwagep90r'],'p90100']=1
    dt.loc[dt['hwage']>dt['hwagep75r'],'p75100']=1
    dt.loc[(dt['hwage']<=dt['hwagep75r'])&(dt['hwage']>dt['hwagep50r']),'p5075']=1
    dt.loc[(dt['hwage']<=dt['hwagep50r'])&(dt['hwage']>dt['hwagep25r']),'p2550']=1
    dt.loc[dt['hwage']<=dt['hwagep25r'],'p025']=1
    
    dt["sex"]='Female'
    dt.loc[dt['male']==1,"sex"]='Male'
    #Datalist
    plist =['p025', 'p2550','p5075','p75100','p90100']
    tbl = ['tbl25','tbl50', 'tbl75', 'tbl75100','tbl90100']
    
    #Split data for female only
    df=dt[(dt['male']==0)]
    
    "weekhswork",
    """Looping over dataframe percentile"""
    index=["Hrs./week market work (condional on reporting wage)",
           "Age","Married","Children under 18","Child under 6"]
    for i, p in enumerate(plist):
        f=df[df[p]==1]
        fm=dt[dt[p]==1]
        gf=f.groupby(["year"])
        gfm=fm.groupby(["year","sex"])
        tf=gf.apply(lambda x: pd.Series(np.nanmean(x[["uhrsworkcon","age",'married','children',"child5"]],axis=0),index))
        tfm=gfm.apply(lambda x: pd.Series(np.nanmean(x['weekhswork']))).unstack()
        tfm.columns=['Hrs./week on household chores','Hrs./week on household chores (by men in same wage bracket)']
        table=pd.concat([tfm,tf],axis=1)
        tbl[i]=table.transpose()
    
    #merge sub-table
    table4 = pd.concat([tbl[0].iloc[:,0:3],
                        tbl[1].iloc[:,0:3],
                        tbl[2].iloc[:,0:3],
                        tbl[3].iloc[:,0:3],
                        tbl[4].iloc[:,0:3]],axis=1)
    table4.columns = pd.MultiIndex.from_product([["0 - 25th","25 - 50th","50 - 75th","75 - 100th",
                                                  "90 - 100th"],['1980','2000']])
    table4=table4.style.format("{:.2f}")
    
    return table4

def create_table5(df):
    
    #Create group
    p025 = df.query('hwage<=hwagep25r')
    p2550 = df.query('hwage>hwagep25r and hwage<= hwagep50r')
    p5075 = df.query('hwage>hwagep50r and hwage<=hwagep75r')
    p75100 = df.query('hwage>hwagep75r')
    p90100 = df.query('hwage>hwagep90r')
    
    #Datalist
    dflist =[p025, p2550,p5075,p75100,p90100]
    tbl = ['tbl25','tbl50', 'tbl75', 'tbl75100','tbl90100']
    
    """Looping over dataframe percentile"""
    for i, dt in enumerate(dflist):
        """ Create female table %percentage"""
        key=["Dummy for positive exp. in housekeeping","Housekeeping exp. |E>0 (1990 dollars)","(std)"]
        g=dt.groupby(["year"])
        table = pd.concat([g.apply(lambda x: np.average(x['dum340310c1'],weights=x['finlwt21'])).round(3),
                           g.apply(lambda x: np.average(x['avcost340310'],weights=x['finlwt'])).round(1),
                           g.apply(lambda x: aweighted_std(x['avcost340310'],x['finlwt']))],
                          axis=0, keys=key).unstack()
        tbl[i]=table
    
    #merge sub-table
    table5 = pd.concat([tbl[0].iloc[:,0:3],
                        tbl[1].iloc[:,0:3],
                        tbl[2].iloc[:,0:3],
                        tbl[3].iloc[:,0:3],
                        tbl[4].iloc[:,0:3]], axis=1)
    table5.columns = pd.MultiIndex.from_product([["0 - 25th percentile","25 - 50th percentile","50 - 75th percentile",
                                              "75 - 100th percentile","90 - 100th percentile"],['1980','1990','2000']])
    
    return table5

def create_table6(wls1,wls2,wls3,wls4):
    
    table6 =pd.DataFrame(np.full((6, 4), np.nan))
        
    table6.iloc[0,:]=["{:.3f}".format(wls1.params['ins']),"{:.3f}".format(wls2.params['ins']),
                      "{:.3f}".format(wls3.params['ins']),"{:.3f}".format(wls4.params['ins'])]    #coeff
    table6.iloc[1,:]=["({:.3f})".format(wls1.bse['ins']),"({:.3f})".format(wls2.bse['ins']),
                      "({:.3f})".format(wls3.bse['ins']),"({:.3f})".format(wls4.bse['ins'])]      #std
    table6.iloc[2,:]=["Basic","Basic","Basic","Basic"]
    table6.iloc[3,:]=["No","Yes","No","No"]
    table6.iloc[4,:]=["No","No","Yes","No"]
    table6.iloc[5,:]=[wls1.nobs/3,wls2.nobs/3,wls3.nobs/3,wls4.nobs/3]                                          #observation
    
    table6.columns= pd.MultiIndex.from_product([["ln[(LS Imm + LS Nat)/Labor Force]"],["(1)","(2)","(3)","(4)"]])
    table6.index=['Log(∑share i,j,1970 × LS Imm jt)',"",'Controls','Excludes California',
                  'Excludes Miami, New York City, and Los Angeles ','Number of cities ']
    
    return table6