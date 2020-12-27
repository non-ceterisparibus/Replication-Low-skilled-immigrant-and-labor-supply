"""
This module contains auxiliary functions for the creation of dataset for table using in the main notebook
"""
import pandas as pd
import numpy as np
from auxiliary.auxiliary_datasets import *
from auxiliary.auxiliary_subdatasets import *
from auxiliary.auxiliary_functions import *

def basis_census():
    """ 
    basis_census, instrument and wage_percentile 
    """
    
    wage_percent=pd.read_stata('data/wage_percentiles_by_region.dta')
    instrument=pd.read_stata('data/instrument.dta')
    basic_census = pd.read_stata('data/basic_census.dta')

    #merge with instrument city
    df=pd.merge(basic_census,instrument,on=['metaread','year'],how='inner')
    
    #drop if max==1990|min==1990
    df=drop_year(df)
    
    #merge outer wage_percentage
    wage_percent['year']=wage_percent['year'].astype('int32')
    df=pd.merge(df,wage_percent,on=['region','year'],how='outer')
    
    # Individual controls/Group percentile
    df = ind_ctrl(df)
    
    # conditional usual hours worked uhrswork=0|99
    df=uhrworks(df)
    
    return df
def fixed_effect(df):
    """
    With dataset 3 years 1980 1990 2000
    """
    # Create fixed effect
    y=pd.get_dummies(df['year'],dummy_na=False,drop_first=True)
    y.sort_index(axis=1, inplace=True)
    for i, col in enumerate(y.columns):
        y.rename(columns={col:'y{:}'.format(i+1)}, inplace=True)

    rg=pd.get_dummies(df['region'])
    rg_y1=rg.mul(y['y1'],axis=0)
    for i, col in enumerate(rg_y1.columns):
        rg_y1.rename(columns={col:'y1rg{:}'.format(i+1)}, inplace=True)
    rg_y2=rg.mul(y['y2'],axis=0)
    for i, col in enumerate(rg_y2.columns):
        rg_y2.rename(columns={col:'y2rg{:}'.format(i+1)}, inplace=True)
    
    city=pd.get_dummies(df['metaread'],dummy_na=False,drop_first=True)
    city.sort_index(axis=1, inplace=True)
    for i, col in enumerate(city.columns):
        city.rename(columns={col:'City{:}'.format(i+1)}, inplace=True)
    
    return [city,rg_y1,rg_y2]
def fx_effect(df):
    """
    With dataset 2 years 1980 2000
    """
    # Create fixed effect
    y=pd.get_dummies(df['year'],dummy_na=False,drop_first=True)
    y.columns=['y1']

    rg=pd.get_dummies(df['region'],dummy_na=False,drop_first=True)
    rg.sort_index(axis=1, inplace=True)
    rg_y1=rg.mul(y['y1'],axis=0)
    for i, col in enumerate(rg_y1.columns):
        rg_y1.rename(columns={col:'y1rg{:}'.format(i+1)}, inplace=True)
    
    city=pd.get_dummies(df['metaread'],dummy_na=False,drop_first=True)
    city.sort_index(axis=1, inplace=True)
    for i, col in enumerate(city.columns):
        city.rename(columns={col:'City{:}'.format(i+1)}, inplace=True)
    
    return [city,y,rg_y1]

def ind_ctrl(df):
    # Individual controls/Group percentile
    marst_f=['Married, spouse present','Married, spouse absent']
    child_f=['Less than 1 year old','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
    
    df['lflw']      =np.where(df['labforce']=='Yes, in the labor force',1,0)
    df['child5']    =np.where(df['nchlt5']=='No children under age 5',0,1)
    df['children']  =np.where(df['yngch'].isin(child_f),1,0)
    df['work60']    =np.where(df['uhrswork']>59,1,0)
    df['work50']    =np.where(df['uhrswork']>49,1,0)
    df['married']   =np.where(df['marst'].isin(marst_f),1,0)
    df['black']     =np.where(df['race']=="Black/Negro",1,0)
    df['p90100']    =np.where(df['hwage']>df['hwagep90r'],1,0)
    df['p75100']    =np.where(df['hwage']>df['hwagep75r'],1,0)
    df['p5075']     =np.where((df['hwage']<=df['hwagep75r'])&(df['hwage']>df['hwagep50r']),1,0)
    df['p2550']     =np.where((df['hwage']<=df['hwagep50r'])&(df['hwage']>df['hwagep25r']),1,0)
    df['p025']      =np.where(df['hwage']<=df['hwagep25r'],1,0)
    df['age']       =df['age'].astype('int32')
    df['agesq']     =np.square(df['age'])
    
    return df

def uhrworks(df):
    # conditional usual hours worked - excludes zeros
    df['chrswork']=df['uhrswork']
    df.loc[df['uhrswork'].isin([0,99]),'chrswork']=np.nan
    df.loc[df['uhrswork']==99,'uhrswork']=np.nan
    
    return df

def merge_ctrl(df,controls):

    # merge controls
    controls['year'] = controls['year'].astype('int32')
    controls.metaread.replace({"Riverside-San Bernadino, CA":"Riverside-San Bernardino,CA"},inplace=True)
    df=pd.merge(df,controls,on=['metaread','year'],how='inner')
    
    return df
def drop_year(df):
    #drop if max==1990|min==1990
    df['year']=df['year'].astype('int32')
    df['metayear_max'] = df.groupby(['metaread'])['year'].transform(max)
    df['metayear_min'] = df.groupby(['metaread'])['year'].transform(min)
    df=df[(df['metayear_max']==2000)&(df['metayear_min']==1980)]
    
    return df
def educ_group(df):
    #Create education filter
    highschl=['N/A or no schooling','N/A','No schooling completed','Nursery school to grade 4',
              'Nursery school, preschool','Kindergarten','Grade 1, 2, 3, or 4','Grade 1','Grade 2',
              'Grade 3','Grade 4','Grade 5, 6, 7, or 8','Grade 5 or 6','Grade 5','Grade 6','Grade 7 or 8',
              'Grade 7','Grade 8','Grade 9','Grade 10','Grade 11','Grade 12','12th grade, no diploma',
              'High school graduate or GED']
    smcollege=['GED or alternative credential','Some college, but less than 1 year','1 year of college',
               '1 or more years of college credit, no degree','2 years of college',"Associate's degree, type not specified",
               "Associate's degree, occupational program","Associate's degree, academic program","3 years of college"]
    college= ['4 years of college',"Bachelor's degree"]
    grad=['5+ years of college',"6 years of college (6+ in 1960-1970)",
            "7 years of college","8+ years of college","Master's degree",
            "Professional degree beyond a bachelor's degree","Doctoral degree"]
    advedu=["Professional degree beyond a bachelor's degree","Doctoral degree"]
    advhigrd=['Attending 7th year of college','7th year of college',"Did not finish 8th year of college",
              "Attending 8th year of college","8th year or more of college"]
    #Create group
    df.loc[df['educd'].isin(highschl),'edulv']="Highschool"
    df.loc[df['educd'].isin(smcollege),'edulv']='SomeCollege'
    df.loc[df['educd'].isin(college),'edulv']='College'
    df.loc[((df['educd'].isin(advedu))&(df['year']!=1980))|((df['year']==1980)&(df['higraded'].isin(advhigrd))),'edulv']='Advanced'
    df.loc[((df['educd'].isin(grad))&~(df['edulv']=="Advanced")),'edulv']='Master'
    df['graduate']=np.where(df['educd'].isin(grad),1,0)
    
    return df

def data_plot6(data3):
    df= educ_group(data3)
    
    df['wkswork']=np.where(df['wkswork1'].isin([0,99]),np.nan,df['wkswork1'])
    df['ahrswork']=df['chrswork'].mul(df['wkswork'])
    df= df[df['ahrswork'].notnull()]
    
    return df
def data_plot7(df2):
    
    df2['wkswork']=np.where(df2['wkswork1'].isin([0,99]),np.nan,df2['wkswork1'])
    df2['ahrswork']=df2['chrswork'].mul(df2['wkswork'])
    df2= df2[df2['ahrswork'].notnull()]
    
    plist=['p025','p2550','p5075','p75100']
    for p in plist:
        df2.loc[df2[p]==1,'percentile']=p
    
    return df2
