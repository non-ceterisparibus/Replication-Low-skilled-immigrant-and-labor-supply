# -*- coding: utf-8 -*-
"""
This module contains auxiliary functions for the creation of dataset for table using in the main notebook
"""
import pandas as pd
import numpy as np
from auxiliary.auxiliary_datasets import *
from auxiliary.auxiliary_subdatasets import *
from auxiliary.auxiliary_functions import *

def data_table2():
    
    df=basis_census()
    #keep women only
    df=df[df['sex']=="Female"]

    return df

def data_table3():
    
    census_edu_women=pd.read_stata('data/census_edu_women.dta')
    instrument=pd.read_stata('data/instrument.dta')
    
    #merge with instrument city
    df=pd.merge(census_edu_women,instrument,on=['metaread','year'],how='inner')
    df['year']=df['year'].astype('int32')
    
    #drop if max==1990|min==1990
    df=drop_year(df)
        
    # Individual controls/Group edulv
    marst_f=['Married, spouse present','Married, spouse absent']
    child_f=['Less than 1 year old','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']

    df['lflw']      =np.where(df['labforce']=='Yes, in the labor force',1,0)
    df['child5']    =np.where(df['nchlt5']=='No children under age 5',0,1)
    df['children']  =np.where(df['yngch'].isin(child_f),1,0)
    df['work60']    =np.where(df['uhrswork']>59,1,0)
    df['work50']    =np.where(df['uhrswork']>49,1,0)
    df['married']   =np.where(df['marst'].isin(marst_f),1,0)
    df['black']     =np.where(df['race']=="Black/Negro",1,0)
    df['age']=df['age'].astype('int32')
    df['agesq']=np.square(df['age'])
    
    # conditional usual hours worked uhrswork=0|99
    df=uhrworks(df)
    
    return df

def data_table4():
    
    psid_atus=pd.read_stata('data/psid_atus.dta')
    instrument=pd.read_stata('data/instrument.dta')
    wage_percent=pd.read_stata('data/wage_percentiles_by_region.dta')
    
    #Edit psid_atus
    psid_atus['metarea']=psid_atus['metaread'].astype('int32')
    psid_atus['year']=psid_atus['year'].astype('int32')
    psid_atus['regioncd']=psid_atus['region'].astype('int32')
    
    #Edit instrument/wage_percent
    instrument['year']=instrument['year'].astype('int32')
    instrument['metarea']=instrument['metarea'].astype('int32')
    wage_percent['year']=wage_percent['year'].astype('int32')
    wage_percent['regioncd']=wage_percent['regioncd'].astype('int32')
    
    psid_atus=psid_atus[['year','weekhswork','uhrsworkcon','age','hwage',
                         'college',"grad",'married','children','child5','male',
                         'regioncd','metarea','stweight']]
    
    #merge instrument
    df=pd.merge(psid_atus,instrument,on=['metarea','year'],how='inner')
    # merge with percentiles from female wage distribution by region
    df=pd.merge(df,wage_percent,on=['regioncd','year'],how='outer')
    
    #Edit df
    df=df[(df['weekhswork'].notnull())&(df['region'].notnull())]
    
    return df

def data_table5():
    
    cex=pd.read_stata('data/cex.dta')
    instrument=pd.read_stata('data/instrument.dta')
    wage_percent=pd.read_stata('data/wage_percentiles_by_region.dta')
    
    # Edit cex
    cex['hwage']=cex['salaryx2'].div(cex['inc_hrsq2'].mul(cex['incweekq2']))
    cex=cex[cex['age2'].between(18,64)&~((cex['origin2'].between(10,17))&(cex['hsdropwoman']==1))]
    cex['metarea']=cex['metaread'].astype('int32')
    cex['year']=cex['year'].astype('int32')
    cex['regioncd']=cex['region'].astype('int32')
    cex=cex[['year','dum340310c1','avcost340310',"childless6",'hwage',
             'collegemorewoman','morecollegewoman','age2','finlwt21',
             'regioncd','metarea','marital2','children']]
    
    # Edit instrument/wage_percent
    instrument['year']=instrument['year'].astype('int32')
    instrument['metarea']=instrument['metarea'].astype('int32')
    wage_percent['year']=wage_percent['year'].astype('int32')
    wage_percent['regioncd']=wage_percent['regioncd'].astype('int32')  
    
    # merge instrument
    df=pd.merge(cex,instrument,on=['metarea','year'],how='inner')
    # merge with percentiles from female wage distribution by region
    df=pd.merge(df,wage_percent,on=['regioncd','year'],how='outer')
    
    # deflate expenses
    df.loc[df['year']==2000,'avcost340310']*=0.71
    df.loc[df['year']==1980,'avcost340310']*=1.72
    
    # Edit cost and weights if cost = 0 weight=0
    df['finlwt']=np.where(df['avcost340310']>0,df['finlwt21'],0)
    
    return df

def data_table6():
    """
     Due to difference between how stata work and how python read data value
     A part of metaread value in region9, alllf, controls have to be edited as belows.
    """
    
    instrument=pd.read_stata('data/instrument.dta')
    region9=pd.read_stata('data/region9.dta')
    alllf=pd.read_stata('data/alllf.dta')
    
    instrument['year'] = instrument['year'].astype('int32')
    alllf['year'] = alllf['year'].astype('int32')
    
    # Edit metaread value
    region9.metaread.replace({"Riverside-San Bernadino, CA":"Riverside-San Bernardino,CA"},inplace=True)
    alllf.metaread.replace({"Riverside-San Bernadino, CA":"Riverside-San Bernardino,CA"},inplace=True)
    
    # merge instrument/region9/alllf
    df=pd.merge(region9,instrument,on=['metaread'],how='inner')
    df=pd.merge(df,alllf,on=['metaread','year'],how='inner')
    
    # drop if max==1990|min==1990
    df=drop_year(df)
    
    return df
def data_table7(data2):
    """
    Using dataset from table2
    """
    
    # merge controls
    controls=pd.read_stata('data/controls.dta')
    df=merge_ctrl(data2,controls)
    
    # Create fixed effect
    city,rg_y1,rg_y2 = fixed_effect(df)
    
    # Individual controls/Group percentile
    df=ind_ctrl(df)
    
    # Variables for later regression
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    var_add=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()+controls.columns[2:14].tolist()
    
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    
    return [df,var_basic,var_add]

def data_table8AC():
    census_women_occ = pd.read_stata('data/census_women_occ.dta')
    instrument=pd.read_stata('data/instrument.dta')
    occ_rankings = pd.read_stata('data/occ_rankings.dta')
    controls=pd.read_stata('data/controls.dta')
    
    # merge with instrument
    df=pd.merge(census_women_occ,instrument,on=['metaread','year'],how='inner')
    # merge occ_rankings
    df=pd.merge(df,occ_rankings,on=["occ1990"],how='inner')
    
    # merge controls
    df['year']=df['year'].astype('int32')
    df=merge_ctrl(df,controls)
    
    # Drop if max==1990|min==1990
    df=drop_year(df)
    # Create fixed effects
    city,rg_y1,rg_y2 = fixed_effect(df)
    
    # Individual controls
    marst_f=['Married, spouse present','Married, spouse absent']
    child_f=['Less than 1 year old','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
    
    df['age']=df['age'].astype('int32')
    df['agesq']=np.square(df['age'])
    df['lflw']      =np.where(df['labforce']=='Yes, in the labor force',1,0)
    df['child5']    =np.where(df['nchlt5']=='No children under age 5',0,1)
    df['children']  =np.where(df['yngch'].isin(child_f),1,0)
    df['work60']    =np.where(df['uhrswork']>59,1,0)
    df['work50']    =np.where(df['uhrswork']>49,1,0)
    df['married']   =np.where(df['marst'].isin(marst_f),1,0)
    df['black']     =np.where(df['race']=="Black/Negro",1,0)
    df["top10_med_hrwage"]=np.where(df['rmedianwagehourocc']<41,1,0)
    df["top25_med_hrwage"]=np.where(df['rmedianwagehourocc']<83,1,0)
    df["top10_avg_hrweek"]=np.where(df['rmeanhoursocc']<15,1,0)
    df["top25_avg_hrweek"]=np.where(df['rmeanhoursocc']<66,1,0)
    df["top10_share50"]=np.where(df['rshare50']<19,1,0)
    df["top25_share50"]=np.where(df['rshare50']<48,1,0)
    
    # Drop if uhrswork=0|99
    df=uhrworks(df)
    
    # Variables for regression
    var_add=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()+controls.columns[2:14].tolist()
    
    # create variable to cluster
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    
    return [df,var_add]
def data_table8D(data3):
    
    # Fixed effects
    city,rg_y1,rg_y2=fixed_effect(data3)
    df=pd.concat([data3,city,rg_y1,rg_y2],axis=1)
    
    #Create education filter
    df=educ_group(df)
    
    # merge controls
    controls=pd.read_stata('data/controls.dta')
    df=merge_ctrl(df,controls)
    
     # create variable to cluster
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    # Variables for regression
    var_add=['age','agesq','married','black','children',
             'child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()+controls.columns[2:14].tolist()
    return [df,var_add]

def data_table10():
    
    df=basis_census()
    # merge controls at city level
    controls=pd.read_stata('data/controls.dta')
    df=merge_ctrl(df,controls)
    
    # Create fixed effect
    city,rg_y1,rg_y2=fixed_effect(df)
    
    #log of hourly wages/hours worked 
    df['lwage']=np.log(df['hwage'])
    df['lhrswork']=np.log(df['chrswork'])
    
    #Interactions with female dummy
    df['Female']=0
    df.loc[df['sex']=="Female","Female"]=1
    df['indepFemale']=df['indep']*df['Female']
    df['insFemale']=df['ins']*df['Female']
    df['child5Female']=df['child5']*df['Female']
    
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    var_add=['age','agesq','married','black','children','child5','Female',
             'child5Female'] + city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()+ controls.columns[2:14].tolist()
    
    #merge all required variables
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)

    return [df,var_add]

def table11_panelA1(df):
    """Using dataset from table2
    basis_census, instrument and wage_percentile
    Drop if max==1990|min==1990
    Drop uhrswork=0|99
    Individual controls/Group percentile
    Keep women only
    """
    # Create fixed effect
    city,rg_y1,rg_y2 = fixed_effect(df)
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    
    # Variables for regression
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    # if hwage~=.
    df =df[df['hwage'].notnull()]
    df['insP75100']  =df['ins']*df['p75100']
    df['insP90100']  =df['ins']*df['p90100']
    df['indepP75100']=df['indep']*df['p75100']
    df['indepP90100']=df['indep']*df['p90100']
    
    df=df[(df['chrswork'].notnull())&(df['perwt']!=0)]
    
    return [df,var_basic]

def table11_panelA2(df):
    """Using dataset from table4
    psid_atus, instrument and wage_percentile
    keep if region!=NA & weekhswork!= NA
    year = 1980|2000 only
    """
        #keep woman only
    df=df[df['male']==0]
        
    # Create fixed effect
    city,y,rg_y1=fx_effect(df)
    
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    df=pd.concat([df,city,rg_y1,y],axis=1)
    
    # Individual controls/Group percentile
    df =df[df['hwage'].notnull()]
    df['p90100']=np.where(df['hwage']>df['hwagep90r'],1,0)
    df['p75100']=np.where(df['hwage']>df['hwagep75r'],1,0)
    df['insP75100']  =df['ins'].mul(df['p75100'])
    df['insP90100']  =df['ins'].mul(df['p90100'])
    df['indepP75100']=df['indep'].mul(df['p75100'])
    df['indepP90100']=df['indep'].mul(df['p90100'])
    
    # Variables for later regression
    var_basic=['age',"y1",'married','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()
    #+rg_y2.columns.tolist() colinear
    return [df,var_basic]
def table11_panelB(df):
    """
    Using dataset from table5
    """
        # Construction of controls
    df =df[df['hwage'].notnull()]
    df['child5']=df['childless6']
    df['p90100']=np.where(df['hwage']>df['hwagep90r'],1,0)
    df['p75100']=np.where(df['hwage']>df['hwagep75r'],1,0)
    df['married']=np.where(df['marital2']==1,1,0)
    
        # Fixed Effects
    y=pd.get_dummies(df['year'])
    y.sort_index(axis=1, inplace=True)
    for i, col in enumerate(y.columns):
        y.rename(columns={col:'y{:}'.format(i+1)}, inplace=True)

    rg=pd.get_dummies(df['region'],dummy_na=False,drop_first=True)
    rg_y1=rg.mul(y['y1'],axis=0)
    for i, col in enumerate(rg_y1.columns):
        rg_y1.rename(columns={col:'y1rg{:}'.format(i+1)}, inplace=True)
    rg_y2=rg.mul(y['y2'],axis=0)
    for i, col in enumerate(rg_y2.columns):
        rg_y2.rename(columns={col:'y2rg{:}'.format(i+1)}, inplace=True)
        
    city=pd.get_dummies(df['metaread'],dummy_na=False,drop_first=True)
    for i, col in enumerate(city.columns):
        city.rename(columns={col:'City{:}'.format(i+1)}, inplace=True)
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    df=pd.concat([df,city,rg_y1,rg_y2,y],axis=1)
    
    #create interactions
    df = df[df['hwage'].notnull()]
    df['insP75100']=df['ins']*df['p75100']
    df['insP90100']=df['ins']*df['p90100']
    df['indepP75100']=df['indep']*df['p75100']
    df['indepP90100']=df['indep']*df['p90100']
    # Variables for later regression
    var_basic=['age2',"y1",'y2','married','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    
    return [df,var_basic]
def extA1_table(data2):
    # Using top quartile
    df=data2[data2['p75100']==1]
    
    # Create fixed effect
    city,rg_y1,rg_y2 = fixed_effect(df)
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    # Interaction variable
    df['insmarried']=df['ins']*df['married']
    df['indepmarried']=df['indep']*df['married']
    # Variables for later regression
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    df=df[(df['chrswork'].notnull())&(df['perwt']!=0)]
    return [df,var_basic]

def extA2_table(data4):
    """Using dataset from table4
    psid_atus, instrument and wage_percentile
    keep if region!=NA & weekhswork!= NA
    year = 1980|2000 only
    """
    #keep woman only - top quartile
    df= data4.query('hwage > hwagep75r')
    df=df[df['male']==0]
    
    # Create fixed effect
    city,y,rg_y1=fx_effect(df)
    
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    # Interaction variable
    df['insmarried']=df['ins']*df['married']
    df['indepmarried']=df['indep']*df['married']
    
    # Variables for later regression
    var_basic=['age',"y1",'married','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()
    #+rg_y2.columns.tolist() colinear
    
    df=pd.concat([df,city,rg_y1,y],axis=1)
    return [df,var_basic]

def extB_table(data5):
    
    df = data5.query('hwage > hwagep75r')
    #construction of controls
    df['child5'] =df['childless6']
    df['married']=np.where(df['marital2']==1,1,0)

    # Fixed Effects
    y=pd.get_dummies(df['year'])
    y.sort_index(axis=1, inplace=True)
        #Rename columns
    for i, col in enumerate(y.columns):
        y.rename(columns={col:'y{:}'.format(i+1)}, inplace=True)

    rg=pd.get_dummies(df['region'],dummy_na=False,drop_first=True)
    rg_y1=rg.mul(y['y1'],axis=0)
    for i, col in enumerate(rg_y1.columns):
        rg_y1.rename(columns={col:'y1rg{:}'.format(i+1)}, inplace=True)
    rg_y2=rg.mul(y['y2'],axis=0)
    for i, col in enumerate(rg_y2.columns):
        rg_y2.rename(columns={col:'y2rg{:}'.format(i+1)}, inplace=True)
        
    city=pd.get_dummies(df['metaread'],dummy_na=False,drop_first=True)
        #Rename columns
    for i, col in enumerate(city.columns):
        city.rename(columns={col:'City{:}'.format(i+1)}, inplace=True)
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    #create interactions
    df['insmarried']=df['ins']*df['married']
    df['indepmarried']=df['indep']*df['married']
    # Variables for later regression
    var_basic=['age2',"y1",'y2','married','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    
    df=pd.concat([df,city,rg_y1,rg_y2,y],axis=1)
    df =df[df['hwage'].notnull()]
    return [df,var_basic]

def data_tableA1(data3):
    wage_percent = pd.read_stata('data/wage_percentiles_by_region.dta')
    wage_percent['year'] = wage_percent['year'].astype('int32')
    
    # merge with percentiles from female wage distribution by region
    df=pd.merge(data3,wage_percent,on=['region','year'],how='outer')
    
    #Generate hwage - table A1
    df['inc_wage']=np.where(df['incwage']>0,df['incwage'],np.nan)
    df['wkswork']=np.where(df['wkswork1'].isin([0,99]),np.nan,df['wkswork1'])
    
    #generate hwage if uhrswork>0&uhrswork<99 (chrswork) & wkswork1>0&wkswork1<99
    df['hwage']=df['inc_wage'].div(df['chrswork'].mul(df['wkswork']))
    
    #use data3-top quartile and controls
    df = df.query('hwage > hwagep75r')
    
    #California and nonmover
    cali=[680,2840,4480,6780,6920,7120,7400,7470,7320,7360,8730,8120]
    bigcities=[5000,4480,5600]
    df['ncali']=1           #not cali
    df.loc[df['metarea'].isin(cali),'ncali']=0
    df['nonmover']=0
    df.loc[df['migrate5d'].isin(['Same state/county, different house','Same house']),'nonmover']=1
    df['nbigcities']=1       # not bigcities
    df.loc[df['metarea'].isin(bigcities),'nbigcities']=0
    df['base']=1
    
    return df

def fx_dfA1(df):
    """ 
    create fixed effects additional table A1
    """

    city,rg_y1,rg_y2 =fixed_effect(df)
    # create variable to cluster
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    #merge dataframe
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    # Variables for regression
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    return [df,var_basic]

def data_tableA2(data3):
    
    #Create education filter
    df=educ_group(data3)
    
    # create fixed effects
    city,rg_y1,rg_y2 =fixed_effect(df)
    
    # create variable to cluster
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    #merge dataframe
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()

    return [df,var_basic]

def tableA3_panelA1(df):
    "using additional table A2"
    df['collegeplus']=np.where((df['edulv']=="College")|(df['graduate']==1),1,0)
    #interations
    df['insCollegeplus']=df['ins']*df['collegeplus']
    df['insGraduate']=df['ins']*df['graduate']
    df['indepCollegeplus']=df['indep']*df['collegeplus']
    df['indepGraduate']=df['indep']*df['graduate']
    
    return df
def tableA3_panelA2(df):
    """Using dataset from table4
    psid_atus, instrument and wage_percentile
    keep if region!=NA & weekhswork!= NA
    year = 1980|2000 only
    """
        #keep woman only
    df=df[df['male']==0]
        
    # Create fixed effect
    city,y,rg_y1=fx_effect(df)
    
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    df=pd.concat([df,city,rg_y1,y],axis=1)
    
    df.rename(columns={'grad': "graduate"},inplace=True)
    df['collegeplus']=np.where((df['college']==1)|(df['graduate']==1),1,0)
    #interations
    df['insCollegeplus']=df['ins']*df['collegeplus']
    df['insGraduate']=df['ins']*df['graduate']
    df['indepCollegeplus']=df['indep']*df['collegeplus']
    df['indepGraduate']=df['indep']*df['graduate']
    # Variables for later regression
    var_basic=['age',"y1",'married','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()
    #+rg_y2.columns.tolist() colinear
    
    return [df,var_basic]
def tableA3_panelB(df):
    """
    Using dataset from table5
    """
    df['child5']=df['childless6']
    df['married']=np.where(df['marital2']==1,1,0)
    df.rename(columns={'collegemorewoman':'collegeplus',
                       'morecollegewoman':'graduate'},inplace=True)
        # Fixed Effects
    y=pd.get_dummies(df['year'])
    y.sort_index(axis=1, inplace=True)
    for i, col in enumerate(y.columns):
        y.rename(columns={col:'y{:}'.format(i+1)}, inplace=True)

    rg=pd.get_dummies(df['region'],dummy_na=False,drop_first=True)
    rg_y1=rg.mul(y['y1'],axis=0)
    for i, col in enumerate(rg_y1.columns):
        rg_y1.rename(columns={col:'y1rg{:}'.format(i+1)}, inplace=True)
    rg_y2=rg.mul(y['y2'],axis=0)
    for i, col in enumerate(rg_y2.columns):
        rg_y2.rename(columns={col:'y2rg{:}'.format(i+1)}, inplace=True)
        
    city=pd.get_dummies(df['metaread'],dummy_na=False,drop_first=True)
    for i, col in enumerate(city.columns):
        city.rename(columns={col:'City{:}'.format(i+1)}, inplace=True)
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    df=pd.concat([df,city,rg_y1,rg_y2,y],axis=1)
    
    #create interactions
    df['insCollegeplus']=df['ins']*df['collegeplus']
    df['insGraduate']=df['ins']*df['graduate']
    df['indepCollegeplus']=df['indep']*df['collegeplus']
    df['indepGraduate']=df['indep']*df['graduate']
    
    # Variables for later regression
    var_basic=['age2',"y1",'y2','married','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    
    return [df,var_basic]
           
            