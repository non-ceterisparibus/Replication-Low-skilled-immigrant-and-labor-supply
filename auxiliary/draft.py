# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:18:31 2020

@author: msduo

"""
import pandas as pd
import numpy as np
import econtools.metrics as emt

from statsmodels.formula.api import ols,wls
from auxiliary.auxiliary_datasets import *
from auxiliary.auxiliary_datasets import *

def data_table7p():
    """Due to difference between how stata(read base on code of data) and python(read lable of data if available)
    I make a edit for control data set before merging controls"""
    
    controls=pd.read_stata('data/controls.dta')
    wage_percent=pd.read_stata('data/wage_percentiles_by_region.dta')
    instrument=pd.read_stata('data/instrument.dta')
    basic_census = pd.read_stata('data/basic_census.dta')
    
    # keep women only
    basic_census=basic_census[basic_census['sex']=="Female"]
    # merge with instrument
    df=pd.merge(basic_census,instrument,on=['metaread','year'],how='inner')
    
    # merge controls
    df['year'] = df['year'].astype('int32')
    controls['year'] = controls['year'].astype('int32')
    controls.metaread.replace({"Riverside-San Bernadino, CA":"Riverside-San Bernardino,CA"},inplace=True)
    df=pd.merge(df,controls,on=['metaread','year'],how='inner')
    
    # merge outer wage_percentage
    wage_percent['year']=wage_percent['year'].astype('int32')
    df=pd.merge(df,wage_percent,on=['region','year'],how='outer')
    
    # Drop if max==1990|min==1990
    df['metayear_max'] = df.groupby(['metaread'])['year'].transform(max)
    df['metayear_min'] = df.groupby(['metaread'])['year'].transform(min)
    df=df[(df['metayear_max']==2000)&(df['metayear_min']==1980)]
    
    # Drop if uhrswork=0|99
    df.loc[df['uhrswork'].isin([0,99]),'uhrswork']=np.nan
    
    # Individual controls/Group percentile
    marst_f=['Married, spouse present','Married, spouse absent']
    child_f=['Less than 1 year old','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
    cols=['lflw','child5','children','married','work50','work60',
          'black','p025','p2550','p5075','p75100','p90100']
    for col in cols:
        df.loc[:,col]=0
    
    df.loc[df['labforce']=='Yes, in the labor force',"lflw"]=1
    df.loc[~(df['nchlt5']=='No children under age 5'),'child5']=1
    df.loc[df['yngch'].isin(child_f),'children']=1
    df.loc[(df['uhrswork']>59),'work60']=1
    df.loc[(df['uhrswork']>49),'work50']=1
    df.loc[df['marst'].isin(marst_f),'married']=1
    df.loc[df['race']=="Black/Negro",'black']=1
    df.loc[df['hwage']>df['hwagep90r'],'p90100']=1
    df.loc[df['hwage']>df['hwagep75r'],'p75100']=1
    df.loc[(df['hwage']<=df['hwagep75r'])&(df['hwage']>df['hwagep50r']),'p5075']=1
    df.loc[(df['hwage']<=df['hwagep50r'])&(df['hwage']>df['hwagep25r']),'p2550']=1
    df.loc[df['hwage']<=df['hwagep25r'],'p025']=1
    df['age']=df['age'].astype('int32')
    df['agesq']=np.square(df['age'])
    
    # Create fixed effect
    y=pd.get_dummies(df['year'])
    y.columns=['y1','y2','y3']
    
    rg=pd.get_dummies(df['region'])
    rg.columns=['rg1','rg2','rg3','rg4','rg5','rg6','rg7','rg8','rg9']
    rg_y1=rg.mul(y['y1'],axis=0)
    rg_y1.columns=['rg1y1','rg2y1','rg3y1','rg4y1','rg5y1','rg6y1','rg7y1','rg8y1','rg9y1']
    rg_y2=rg.mul(y['y2'],axis=0)
    rg_y2.columns=['rg1y2','rg2y2','rg3y2','rg4y2','rg5y2','rg6y2','rg7y2','rg8y2','rg9y2']
    
    city=pd.get_dummies(df['metaread'])
        #Rename columns
    for i, col in enumerate(city.columns):
        city.rename(columns={col:'City{:}'.format(i+1)}, inplace=True)
    
    # Variables for later regression
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    var_add=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()+controls.columns[2:14].tolist()
    
    # Assigning numerical values and storing in another column
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    
    return [df,var_basic,var_add]


def reg_table7(df,col_basic,col_add):
    col_list =['p90100','p75100','p5075','p2550','p025']
    Y=['uhrswork','work50','work60']
    
    ivtable7=['t025', 't2550','t5075','t75100','t90100']
    FS_basic = ['FSb25',"FSb50","FSb75","FS7b5100","FSb90100"]
    FS_add = ['FSa25',"FSa50","FSa75","FSa75100","FSa90100"]
    
    OLS = ['OLSuh',"OLSw50","OLSw60"]
    IVb = ['IVbuh',"IVbw50","IVbw60"]
    IVa = ['IVauh',"IVaw50","IVaw60"]
    
    "Looping through percentile group"
    for i, col in enumerate(col_list):
        dt=df[(df[col]==1)&(df['perwt']>0)]
        w=dt['perwt']
        #First stage OLS
        formula_fsbasic = 'indep ~  ins +' + ' + '.join(col_basic[0:])
        formula_fsadd = 'indep ~  ins +' + ' + '.join(col_add[0:])
        #First stage
        FS_basic[i]= wls(formula=formula_fsbasic,data=dt,weights=w).fit(cov_type='cluster',cov_kwds={'groups': dt['metaready']})
        FS_add[i]= wls(formula=formula_fsadd,data=dt,weights=w).fit(cov_type='cluster',cov_kwds={'groups': dt['metaready']})
        dt.loc[:,'indep_predb'] = FS_basic[i].predict()
        dt.loc[:,'indep_preda'] = FS_add[i].predict()
        
        "Looping through Y for OLS/IV basic/IV additional regression"
        for i, y in enumerate(Y):
            formulaOLS = y+' ~ indep +' + ' + '.join(col_basic[0:])
            formulaIV_basic= y+' ~ indep_predb +' + ' + '.join(col_basic[0:])
            formulaIV_add= y+' ~ indep_predb +' + ' + '.join(col_add[0:])
            
            # OLS
            OLS[i]= wls(formula=formulaOLS,data=dt,weights=w).fit(cov_type='cluster',cov_kwds={'groups': dt['metaready']})
            
            # Intrumental variables basic
            IVb[i]= wls(formula=formulaIV_basic,data=dt,weights=w).fit(cov_type='cluster',cov_kwds={'groups': dt['metaready']})
            
            #Intrumental variables additonal
            IVa[i]= wls(formula=formulaIV_add,data=dt,weights=w).fit(cov_type='cluster',cov_kwds={'groups': dt['metaready']})
        
        """
        End of looping through Y
        Extract OLS/IV regression results
        """
        ivtable7[i] =pd.DataFrame(np.full((2, 9), np.nan))
        ivtable7[i].iloc[0,:]=[OLS[0].params['indep'],IVb[0].params['indep_predb'],IVa[0].params['indep_preda'],
                               OLS[1].params['indep'],IVb[1].params['indep_predb'],IVa[1].params['indep_preda'],
                               OLS[2].params['indep'],IVb[2].params['indep_predb'],IVa[2].params['indep_preda']]              #coefficients
        ivtable7[i].iloc[1,:]=[OLS[0].bse['indep'],IVb[0].bse['indep_predb'],IVa[0].params['indep_preda'],
                               OLS[1].bse['indep'],IVb[1].bse['indep_predb'],IVa[1].bse['indep_preda'],
                               OLS[2].bse['indep'],IVb[2].bse['indep_predb'],IVa[2].bse['indep_preda']] #std"
    
    """
    End of looping through percentile group
    
    """
    table7=pd.concat([ivtable7[0],ivtable7[1],ivtable7[2],ivtable7[3],ivtable7[4]],axis=0)
    table7.columns= pd.MultiIndex.from_product([["Usual hours| H>0","p(Hours >= 50)","p(Hours >= 60)"],["OLS (basic)","IV (basic) ","IV (additional"]])
    table7.index=['90 - 100',"", '75 - 100',"",'50 - 75',"",'25 - 50',"",'0 - 25',""]
    
    #Extract First-stage regression results
    FS_basic0 = FS_basic[0].summary2().tables[1].round(4)
    FS_basic1 = FS_basic[1].summary2().tables[1].round(4)
    FS_basic2 = FS_basic[2].summary2().tables[1].round(4)
    FS_basic3 = FS_basic[3].summary2().tables[1].round(4)
    FS_basic4 = FS_basic[4].summary2().tables[1].round(4)
        
    FS_add0 = FS_add[0].summary2().tables[1].round(4)
    FS_add1 = FS_add[1].summary2().tables[1].round(4)
    FS_add2 = FS_add[2].summary2().tables[1].round(4)
    FS_add3 = FS_add[3].summary2().tables[1].round(4)
    FS_add4 = FS_add[4].summary2().tables[1].round(4)
    
    fs_table7 = pd.DataFrame(np.full((10, 2), np.nan))
    fs_table7.iloc[:,0]=[FS_basic4.iloc[1,0],FS_basic4.iloc[1,1],FS_basic3.iloc[1,0],FS_basic3.iloc[1,1],FS_basic2.iloc[1,0],FS_basic2.iloc[1,1],FS_basic1.iloc[1,0],FS_basic1.iloc[1,1],FS_basic0.iloc[1,0],FS_basic0.iloc[1,1]]
    fs_table7.iloc[:,1]=[FS_add4.iloc[1,0],FS_add4.iloc[1,1],FS_add3.iloc[1,0],FS_add3.iloc[1,1],FS_add2.iloc[1,0],FS_add2.iloc[1,1],FS_add1.iloc[1,0],FS_add1.iloc[1,1],FS_add0.iloc[1,0],FS_add0.iloc[1,1]]
    fs_table7.columns= pd.MultiIndex.from_product([["First stage"],[" OLS (basic)"," OLS (additional)"]])
    
    fntable7=pd.concat([table7,fs_table7 ],axis=1)

    return fntable7

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    https://kodify.net/python/math/truncate-decimals/
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor
def data_table3ex(data3):
    df= edu_group(data3)
    df= df[df['chrswork'].notnull()]
    
    df['wkswork']=np.where(df['wkswork1'].isin([0,99]),np.nan,df['wkswork1'])
    df['ahrswork']=df['chrswork'].mul(df['wkswork'])
    
    return df
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
    
    # create fixed effects
    city,rg_y1,rg_y2 =fixed_effect(df)
    
    # create variable to cluster
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    #California and nonmover
    cali=[680,2840,4480,6780,6920,7120,7400,7470,7320,7360,8730,8120]
    bigcities=[5000,4480,5600]
    df['ncali']=1           #not cali
    df.loc[df['metarea'].isin(cali),'ncali']=0
    df['nonmover']=0
    df.loc[df['migrate5d'].isin(['Same state/county, different house','Same house']),'nonmover']=1
    df['nbicities']=1       # not bigcities
    df.loc[df['metarea'].isin(bigcities),'nbigcities']=0
    df['base']=1
    
    #merge dataframe
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    
    # Variables for regression
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    
    return [df,var_basic]

def data_tableA2(data3):
    #Create education filter
    df=edu_group(data3)
    
    #keep native
    df=df[~df['citizen'].isin(["Naturalized citizen","Not a citizen"])]
    
    # create fixed effects
    city,rg_y1,rg_y2 =fixed_effect(df)
    
    # create variable to cluster
    df["metaready"] = df["metaread"].astype(str) + df["year"].astype(str)
    
    #merge dataframe
    df=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    
    #adding controls
    controls=pd.read_stata('data/controls.dta')
    df=merge_ctrl(df,controls)
    
    var_basic=['age','agesq','married','black','children','child5']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    #var_add=['age','agesq','married','black','children','child5']+city.columns.tolist()
    #+rg_y1.columns.tolist()+rg_y2.columns.tolist()+controls.columns[2:14].tolist()
    return [df,var_basic]
def reg_tableA2(df,var):
    clt=df['metaready']
    w = df['perwt']
    #First stage OLS
    formulafs= 'indep ~ ins + ' + ' + '.join(var[0:])
    FS=wls_cluster(formulafs,df,w,clt)
    df['indep_pred'] = FS.predict()
    
    #Intrumental variables
    formulaIV=' ~ indep_pred +' + ' + '.join(var[0:])
    
    IV_uh = wls_cluster("uhrswork"+formulaIV,df,w,clt)
    IV_lf = wls_cluster("lflw" + formulaIV,df,w,clt)
    IV_ch = wls_cluster("chrswork"+formulaIV,df,w,clt)
    IV_w50= wls_cluster("work50"+formulaIV,df,w,clt)
    IV_w60= wls_cluster("work60"+formulaIV,df,w,clt)
    
    tableA2= pd.DataFrame(np.full((2,5),""))
    
    return tableA2
def reg_tableA1(df,clt,var_basic):
    x  =['indep']           # endogenous regressor(s)
    ins=['ins']             # excluded instrument(s)
    wt ='perwt'             # Weight
 
    #Intrumental variables
    IV_ch = emt.ivreg(df, y_name="chrswork", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    IV_w50 = emt.ivreg(df, y_name="work50", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    IV_w60 = emt.ivreg(df, y_name="work60", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    
    tableA1 = pd.DataFrame(np.full((2,3),""))
    tableA1.iloc[0,:]=["{:.3f}".format(IV_ch.beta['indep']),"{:.3f}".format(IV_w50.beta['indep']),
                       "{:.3f}".format(IV_w60.beta['indep'])]
    tableA1.iloc[1,:]=["({:.3f})".format(IV_ch.se['indep']),"({:.3f})".format(IV_w50.se['indep']),
                       "({:.3f})".format(IV_w60.se['indep'])]
    tableA1.columns=["Usual hours|H > 0", "P (Hours >= 50)","P (Hours >= 60)"]
    return tableA1