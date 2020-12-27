# -*- coding: utf-8 -*-
"""This module contains auxiliary functions for the creation of regression in the main notebook."""


import pandas as pd
import numpy as np
import econtools.metrics as emt

from statsmodels.formula.api import wls
from auxiliary.auxiliary_datasets import *
from auxiliary.auxiliary_subdatasets import *
from auxiliary.auxiliary_functions import *

def get_reg_first_stage():
    
    df=data_table6()
    
    #Adding Controls
    controls=pd.read_stata('data/controls.dta')
    controls['year'] = controls['year'].astype('int32')
    controls.metaread.replace({"Riverside-San Bernadino, CA":"Riverside-San Bernardino,CA"},inplace=True)
    
    cali=[680,2840,4480,6780,6920,7120,7400,7470,7320,7360,8730,8120]
    df['California']=0
    df.loc[df['metarea'].isin(cali),'California']=1
    
    # Create fixed effect
    y=pd.get_dummies(df['year'])
    y.sort_index(axis=1, inplace=True)
    for i, col in enumerate(y.columns):
        y.rename(columns={col:'y{:}'.format(i+1)}, inplace=True)

    rg=pd.get_dummies(df['region9'])
    rg.columns=['rg1','rg2','rg3','rg4','rg5','rg6','rg7','rg8','rg9']
    rg_y1=rg.mul(y['y1'],axis=0)
    rg_y1.columns=['rg1y1','rg2y1','rg3y1','rg4y1','rg5y1','rg6y1','rg7y1','rg8y1','rg9y1']
    rg_y2=rg.mul(y['y2'],axis=0)
    rg_y2.columns=['rg1y2','rg2y2','rg3y2','rg4y2','rg5y2','rg6y2','rg7y2','rg8y2','rg9y2']
    
    city=pd.get_dummies(df['metaread'])
    city.sort_index(axis=1, inplace=True)
        # Rename columns
    for i, col in enumerate(city.columns):
        city.rename(columns={col:'City{:}'.format(i+1)}, inplace=True)
        
    # Concate dataframe
    reg1=pd.concat([df,city,rg_y1,rg_y2],axis=1)
    w1=reg1['all']
    reg2=reg1[reg1['California']==0]
    w2=reg2['all']
    reg3=reg1[~reg1['metarea'].isin([4480,5600,5000])]
    w3=reg3['all']
    reg4=pd.merge(reg1,controls,on=['metaread','year'],how='inner')
    w4=reg4['all']
    
    columns=['ins']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()
    columns4=['ins']+city.columns.tolist()+rg_y1.columns.tolist()+rg_y2.columns.tolist()+controls.columns[2:14].tolist()
    
    formula = 'indep ~ ' + ' + '.join(columns[0:])
    formula4 = 'indep ~ ' + ' + '.join(columns4[0:])
    wls1=wls(formula=formula, data=reg1, weights=w1).fit(cov_type='HC1')
    wls2=wls(formula=formula, data=reg2, weights=w2).fit(cov_type='HC1')
    wls3=wls(formula=formula, data=reg3, weights=w3).fit(cov_type='HC1')
    wls4=wls(formula=formula4, data=reg4, weights=w4).fit(cov_type='HC1')
    
    return [wls1,wls2,wls3,wls4]
def reg_fstable7(df,var_basic,var_add):
    """ 
    First-stage OLS regression for each percentile group
    """
    
    #First stage OLS
    formula_fsbasic = 'indep ~  ins +' + ' + '.join(var_basic[0:])
    formula_fsadd   = 'indep ~  ins +' + ' + '.join(var_add[0:])

    FS_basic= wls_cluster(formula_fsbasic,df,'perwt','metaready')
    FS_add= wls_cluster(formula_fsadd,df,'perwt','metaready')
    
    #Extract First-stage regression results
    fs_table7 = pd.DataFrame(np.full((3, 2),""))
    fs_table7.iloc[0,:]=["{:.3f}".format(FS_basic.params['ins']),"{:.3f}".format(FS_add.params['ins'])]
    fs_table7.iloc[1,:]=["({:.3f})".format(FS_basic.bse['ins']),"({:.3f})".format(FS_add.bse['ins'])]
    fs_table7.iloc[2,:]=["Basic","Additional"]
    fs_table7.columns= pd.MultiIndex.from_product([["First stage"],[" OLS"," OLS"]])

    return fs_table7

def reg_ivtable7(df,y,var_basic,var_add):
    """
    Parameters:
        df (DataFrame) – Data with any relevant variables.
        y_name (str) – Column name in df of the dependent variable.
        x_name (str or list) – Column name(s) in df of the endogenous regressor(s).
        z_name (str or list) – Column name(s) in df of the excluded instrument(s)
        w_name (str or list) – Column name(s) in df of the included instruments/exogenous regressors
        awt_name (str) – Column name in df to use for analytic weights in regression.
        cluster (str) – Column name in df used to cluster standard errors.
    """
    ydict = {"chrswork": "Usual hours | H > 0 ",
             "work50": "P(Hours >= 50)",
             "work60":"P(Hours >= 60)"}
    x  =['indep']       # endogenous regressor
    ins=['ins']         # excluded instrument(s)
    wt ='perwt'         # Weight
    clt='metaready'     # Cluster
    
    # OLS regression for outcome Y
    formulaOLS = y +' ~ indep +' + ' + '.join(var_basic[0:])
    OLS= wls_cluster(formulaOLS,df,wt,clt)
        
        # Intrumental variables basic
    IVb= emt.ivreg(df, y_name=y, x_name=x, z_name=ins, w_name=var_basic,
                   awt_name=wt, cluster=clt, addcons=True)
        
        #Intrumental variables additonal
    IVa= emt.ivreg(df, y_name=y, x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    # Extract regression results
    ivtable7 = pd.DataFrame(np.full((3, 3),""))
    ivtable7.iloc[0,:]=["{:.3f}".format(OLS.params['indep']),"{:.3f}".format(IVb.beta['indep']),
                        "{:.3f}".format(IVa.beta['indep'])]   
    ivtable7.iloc[1,:]=["({:.3f})".format(OLS.bse['indep']),"({:.3f})".format(IVb.se['indep']),
                        "({:.3f})".format(IVa.se['indep'])]             
    ivtable7.iloc[2,:]=["Basic","Basic","Additional"]
    ivtable7.columns= pd.MultiIndex.from_product([[ydict[y]],["OLS","IV","IV"]])
    
    return ivtable7
def reg_table8AC(df,var_add,panel):
    ydict={"top10_med_hrwage":"Top 10",'top25_med_hrwage':"Top 25",
           'top10_avg_hrweek':"Top 10",'top25_avg_hrweek':"Top 25",
           'top10_share50'   :"Top 10",'top25_share50'   :"Top 25"}
    
    table=reg8(df,var_add,panel,ydict)
    
    return table
def reg_table8D(df,var_add,edu):
    #Datalist
    ydict={'Advanced':"Professionals and PhDs",
           'Master':"Master’s degree",
           'College':"College grads "}
    
    table=reg8(df,var_add,edu,ydict)
    return table
    
def reg8(df,var_add,panel,ydict):
    x  =['indep']       # endogenous regressor
    ins=['ins']         # excluded instrument(s)
    wt ='perwt'         # Weight
    clt='metaready'     # Cluster
    
            #Intrumental variables additonal
    #uhrswork: Usual hours|week
    IVuhr= emt.ivreg(df, y_name='uhrswork', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    #lflw:Labor Force Participation
    IVlfp= emt.ivreg(df, y_name='lflw', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    #work50: P (Hours>= 50) 
    IVw50= emt.ivreg(df, y_name='work50', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    #work60: P (Hours>= 60) 
    IVw60= emt.ivreg(df, y_name='work60', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)

    #chrswork:Usual hours|H >0
    IVchr= emt.ivreg(df, y_name='chrswork', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    ivtable8 = pd.DataFrame(np.full((2,5),""))
    ivtable8.iloc[:,0]=["{:.3f}".format(IVuhr.beta['indep']),"({:.3f})".format(IVuhr.se['indep'])]
    ivtable8.iloc[:,1]=["{:.3f}".format(IVlfp.beta['indep']),"({:.3f})".format(IVlfp.se['indep'])]
    ivtable8.iloc[:,2]=["{:.3f}".format(IVchr.beta['indep']),"({:.3f})".format(IVchr.se['indep'])]
    ivtable8.iloc[:,3]=["{:.3f}".format(IVw50.beta['indep']),"({:.3f})".format(IVw50.se['indep'])]
    ivtable8.iloc[:,4]=["{:.3f}".format(IVw60.beta['indep']),"({:.3f})".format(IVw60.se['indep'])]
    ivtable8.index=[ydict[panel],""]
    ivtable8.columns =["Usual hours per week ","LFP","Usual hours|H>0 ","P (Hours>=50)","P (Hours>=60)"]
    
    return ivtable8
    
def reg_table9(df,panel,var_add):
    pdict = {"p90100":"90 - 100","p75100":"75 - 100",
             "p5075":" 50 - 75","p2550":" 25 - 50","p025":"0 - 25"}
    x  =['indep','indepchild5']     # endogenous regressor(s)
    ins=['ins','inschild5']         # excluded instrument(s)
    wt ='perwt'                     # Weight
    clt='metaready'                 # Cluster
            
        #Intrumental variables additonal
    
    #work50: P (Hours >= 50)
    IVw50= emt.ivreg(df, y_name='work50', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    #work60: P (Hours >= 60)
    IVw60= emt.ivreg(df, y_name='work60', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    #chrswork: Usual hours|H > 0 
    IVchr= emt.ivreg(df, y_name='chrswork', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    # Extract regression results
    ivtable9 = pd.DataFrame(np.full((2,6),""))
    ivtable9.iloc[:,0]=["{:.3f}".format(IVchr.beta['indep']),      "({:.3f})".format(IVchr.se['indep'])]
    ivtable9.iloc[:,1]=["{:.3f}".format(IVchr.beta['indepchild5']),"({:.3f})".format(IVchr.se['indepchild5'])]
    ivtable9.iloc[:,2]=["{:.3f}".format(IVw50.beta['indep']),      "({:.3f})".format(IVw50.se['indep'])]
    ivtable9.iloc[:,3]=["{:.3f}".format(IVw50.beta['indepchild5']),"({:.3f})".format(IVw50.se['indepchild5'])]
    ivtable9.iloc[:,4]=["{:.3f}".format(IVw60.beta['indep']),      "({:.3f})".format(IVw60.se['indep'])]
    ivtable9.iloc[:,5]=["{:.3f}".format(IVw60.beta['indepchild5']),"({:.3f})".format(IVw60.se['indepchild5'])]
    
    ivtable9.columns= pd.MultiIndex.from_product([["Usual hours|H>0 ", "P(Hours>= 50)","P(Hours>= 60)"],["Ln(LS Skilled)","Ln(LS Skilled)x child 0-5"]])
    ivtable9.index=[pdict[panel],""]
    
    return ivtable9

def reg_ivtable10(df,panel,var_add):
    x  =['indep','indepFemale']     # endogenous regressor(s)
    ins=['ins','insFemale']         # excluded instrument(s)
    wt ='perwt'                     # Weight
    clt='metaready'                 # Cluster

    pdict = {"p90100":"90 - 100","p75100":"75 - 100","p5075":" 50 - 75","p2550":" 25 - 50","p025":"0 - 25"}
    
            #Intrumental variables additional
    # Usual hours|H>0 
    IVch= emt.ivreg(df, y_name='chrswork', x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    #log(Usual hours|H>0)
    IVlh= emt.ivreg(df, y_name="lhrswork", x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    #log(wage)
    IVlw= emt.ivreg(df, y_name="lwage", x_name=x, z_name=ins, w_name=var_add,
                   awt_name=wt, cluster=clt, addcons=True)
    
    # Extract regression results
    ivtable10 = pd.DataFrame(np.full((2,6),""))
    ivtable10.iloc[:,0]=["{:.3f}".format(IVch.beta['indep']),      "({:.3f})".format(IVch.se['indep'])]
    ivtable10.iloc[:,1]=["{:.3f}".format(IVch.beta['indepFemale']),"({:.3f})".format(IVch.se['indepFemale'])]
    ivtable10.iloc[:,2]=["{:.3f}".format(IVlh.beta['indep']),      "({:.3f})".format(IVlh.se['indep'])]
    ivtable10.iloc[:,3]=["{:.3f}".format(IVlh.beta['indepFemale']),"({:.3f})".format(IVlh.se['indepFemale'])]
    ivtable10.iloc[:,4]=["{:.3f}".format(IVlw.beta['indep']),      "({:.3f})".format(IVlw.se['indep'])]
    ivtable10.iloc[:,5]=["{:.3f}".format(IVlw.beta['indepFemale']),"({:.3f})".format(IVlw.se['indepFemale'])]
    
    ivtable10.columns= pd.MultiIndex.from_product([["Usual hours|H>0 ","log(Usual hours|H>0)","log(Wage)"],["Ln(LS Skilled)","Ln(LS Skilled)xFemale"]])
    ivtable10.index=[pdict[panel],""]
    return ivtable10

def reg_table11(df,y,wt,var_basic):
    x_topq  =['indep','indepP75100']     # endogenous regressor(s)
    x_topd  =['indep','indepP90100']
    ins_topq=['ins','insP75100']         # excluded instrument(s)
    ins_topd=['ins','insP90100']
    var_topq=['p75100']+var_basic        # exogenous regressor(s)
    var_topd=['p90100']+var_basic
    clt='metaready'                      # Cluster
    
    pndict={'chrswork'      :"A1.Usual market hours worked/week(census)",
            'weekhswork'    :"A2. Hours per week spent doing household chores",
            'dum340310c1'   :"B1. Dummy for expenditures >0",
            'avcost340310'  :"B2. Level of expenditures(unconditional)"}

        # OLS
    formulaOLS = y+' ~ indep + indepP75100 + p75100 +' + ' + '.join(var_basic[0:])
    OLS= wls_cluster(formulaOLS,df,wt,clt)
    
        #Intrumental variables
    IV_topq= emt.ivreg(df, y_name=y, x_name=x_topq, z_name=ins_topq, w_name=var_topq,
                       awt_name=wt, cluster=clt, addcons=True)

    IV_topd= emt.ivreg(df, y_name=y, x_name=x_topd, z_name=ins_topd, w_name=var_topd,
                       awt_name=wt, cluster=clt, addcons=True)
        
    # Regression results
    table11 = pd.DataFrame(np.full((6,3),""))
    table11.iloc[0,:]=["{:.3f}".format(OLS.params['indep']),
                       "{:.3f}".format(IV_topq.beta['indep']),
                       "{:.3f}".format(IV_topd.beta['indep'])]
    table11.iloc[1,:]=["({:.3f})".format(OLS.bse['indep']),
                       "({:.3f})".format(IV_topq.se['indep']),
                       "({:.3f})".format(IV_topd.se['indep'])]
    table11.iloc[2,0:2]=["{:.3f}".format(OLS.params['indepP75100']),
                         "{:.3f}".format(IV_topq.beta['indepP75100'])]
    table11.iloc[3,0:2]=["({:.3f})".format(OLS.bse['indepP75100']),
                         "({:.3f})".format(IV_topq.se['indepP75100'])]
    table11.iloc[4,2]="{:.3f}".format(IV_topd.beta['indepP90100'])
    table11.iloc[5,2]="({:.3f})".format(IV_topd.se['indepP90100'])
    
    table11.index=['ln((LS Imm. + LS Nat.)/LF)',"",
                   'ln((LS Imm. + LS Nat.)/LF)xTop quartile',"",
                   'ln((LS Imm. + LS Nat.)/LF)xTop decile',""]
    table11.columns= pd.MultiIndex.from_product([[pndict[y]],["OLS"," IV ","IV"]])
    
    return table11
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
def reg_tableA2(df,var_basic,edu):
    idict={"grad":"Graduate education",
           'Advanced':"Professionals and PhDs",
           'Master':"Master’s degree ",
           'College':"College graduates",
           'SomeCollege':"Some college ",
           "Highschool":"HS grad less "}
    x  =['indep']           # endogenous regressor(s)
    ins=['ins']             # excluded instrument(s)
    wt ='perwt'             # Weight
    clt='metaready'
    
    IV_uh = emt.ivreg(df, y_name="uhrswork", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    IV_lf = emt.ivreg(df, y_name="lflw", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    IV_ch = emt.ivreg(df, y_name="chrswork", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    IV_w50= emt.ivreg(df, y_name="work50", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    IV_w60= emt.ivreg(df, y_name="work60", x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)
    
    tableA2= pd.DataFrame(np.full((2,5),""))
    tableA2.iloc[0,:]=["{:.3f}".format(IV_uh.beta['indep']),"{:.3f}".format(IV_lf.beta['indep']),
                       "{:.3f}".format(IV_ch.beta['indep']),"{:.3f}".format(IV_w50.beta['indep']),
                       "{:.3f}".format(IV_w60.beta['indep'])]
    tableA2.iloc[1,:]=["({:.3f})".format(IV_uh.se['indep']),"({:.3f})".format(IV_lf.se['indep']),
                       "({:.3f})".format(IV_ch.se['indep']),"({:.3f})".format(IV_w50.se['indep']),
                       "({:.3f})".format(IV_w60.se['indep'])]
    tableA2.columns=["Usual hours"," LFP","Usual hours|H > 0", "P (Hours >= 50)","P (Hours >= 60)"]
    tableA2.index=[idict[edu],""]
    
    return tableA2

def reg_tableA3(df,y,wt,var_basic):
    x_coll  =['indep','indepCollegeplus']     # endogenous regressor(s)
    x_grad  =['indep','indepGraduate']
    ins_coll=['ins','insCollegeplus']         # excluded instrument(s)
    ins_grad=['ins','insGraduate']
    var_coll=['collegeplus']+var_basic        # exogenous regressor(s)
    var_grad=['graduate']+var_basic
    clt='metaready'                      # Cluster
    
    pndict={'uhrswork'      :"A1.Usual market hours worked/week(census)",
            'weekhswork'    :"A2. Hours per week spent doing household chores",
            'dum340310c1'   :"B1. Dummy for expenditures >0",
            'avcost340310'  :"B2. Level of expenditures(unconditional)"}

        # OLS
    formulaOLS = y+' ~ indep + indepCollegeplus + collegeplus +' + ' + '.join(var_basic[0:])
    OLS= wls_cluster(formulaOLS,df,wt,clt)
    
        #Intrumental variables
    IV_coll= emt.ivreg(df, y_name=y, x_name=x_coll, z_name=ins_coll, w_name=var_coll,
                       awt_name=wt, cluster=clt, addcons=True)

    IV_grad= emt.ivreg(df, y_name=y, x_name=x_grad, z_name=ins_grad, w_name=var_grad,
                       awt_name=wt, cluster=clt, addcons=True)
        
    # Regression results
    tableA3 = pd.DataFrame(np.full((6,3),""))
    tableA3.iloc[0,:]=["{:.3f}".format(OLS.params['indep']),
                       "{:.3f}".format(IV_coll.beta['indep']),
                       "{:.3f}".format(IV_grad.beta['indep'])]
    tableA3.iloc[1,:]=["({:.3f})".format(OLS.bse['indep']),
                       "({:.3f})".format(IV_coll.se['indep']),
                       "({:.3f})".format(IV_grad.se['indep'])]
    tableA3.iloc[2,0:2]=["{:.3f}".format(OLS.params['indepCollegeplus']),
                         "{:.3f}".format(IV_coll.beta['indepCollegeplus'])]
    tableA3.iloc[3,0:2]=["({:.3f})".format(OLS.bse['indepCollegeplus']),
                         "({:.3f})".format(IV_coll.se['indepCollegeplus'])]
    tableA3.iloc[4,2]="{:.3f}".format(IV_grad.beta['indepGraduate'])
    tableA3.iloc[5,2]="({:.3f})".format(IV_grad.se['indepGraduate'])
    
    tableA3.index=['ln((LS Imm. + LS Nat.)/LF)',"",
                   'ln((LS Imm. + LS Nat.)/LF)xCollege or more',"",
                   'ln((LS Imm. + LS Nat.)/LF)xGraduate education ',""]
    tableA3.columns= pd.MultiIndex.from_product([[pndict[y]],["OLS"," IV ","IV"]])
    
    return tableA3

def reg_ext(df,y,wt,var_basic):
    x  =['indep','indepmarried']    # endogenous regressor(s)
    ins=['ins','insmarried']        # excluded instrument(s)
    clt='metaready'                 # Cluster
    
    pndict={'chrswork'      :"A1.Usual market hours worked/week(census)",
            'weekhswork'    :"A2. Hours per week spent doing household chores",
            'dum340310c1'   :"B1. Dummy for expenditures >0",
            'avcost340310'  :"B2. Level of expenditures(unconditional)"}
    
    # OLS
    formulaOLS = y+' ~ indep + indepmarried +' + ' + '.join(var_basic[0:])
    OLS= wls_cluster(formulaOLS,df,wt,clt)
    
        #Intrumental variables
    IV= emt.ivreg(df, y_name=y, x_name=x, z_name=ins, w_name=var_basic,
                  awt_name=wt, cluster=clt, addcons=True)

        # Regression results
    ext = pd.DataFrame(np.full((4,2),""))
    ext.iloc[0,:]=["{:.3f}".format(OLS.params['indep']),        "{:.3f}".format(IV.beta['indep'])]
    ext.iloc[1,:]=["({:.3f})".format(OLS.bse['indep']),         "({:.3f})".format(IV.se['indep'])]
    ext.iloc[2,:]=["{:.3f}".format(OLS.params['indepmarried']), "{:.3f}".format(IV.beta['indepmarried'])]
    ext.iloc[3,:]=["({:.3f})".format(OLS.bse['indepmarried']),  "({:.3f})".format(IV.se['indepmarried'])]
    
    ext.index=['ln((LS Imm. + LS Nat.)/LF)',"",'ln((LS Imm. + LS Nat.)/LF) x married',""]
    ext.columns= pd.MultiIndex.from_product([[pndict[y]],["OLS"," IV "]])
    return ext

    
    
        