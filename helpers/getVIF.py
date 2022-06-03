#!/home/govindas/venvs/aba/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed March 21 20:49:12 2021

@author: md
"""
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

matpath = sys.stdin.readline()
matpath = matpath[:-1]
sys.stdout.write('Calculating VIF for ' + matpath + '\n')

raw_cols = open(matpath,'r').readlines()[3].split('"')[1].split(';')

design = np.loadtxt(matpath)
tmp1_df = pd.DataFrame(design,columns=raw_cols)

cols = [y for y in [x for x in raw_cols if "Run" not in x] if "Motion" not in y]
tmptmp_df = tmp1_df[cols]

# Remove columns containing all zeros
badCols = (tmptmp_df == 0.0).all().items()
colsToRemove = [i for i, x in badCols if x]

badCols = (tmptmp_df == 0.0).all().items()
colsToKeep = [i for i, x in badCols if not x]
tmp_df = tmptmp_df[colsToKeep]

df_corr = tmp_df.corr()

df_for_VIF = tmp_df
df_for_VIF = sm.add_constant(df_for_VIF)
VIF_df = pd.DataFrame()
VIF_df['Feature'] = df_for_VIF.columns
VIF_df['VIF'] = [variance_inflation_factor(df_for_VIF.values, i) for i in range(df_for_VIF.shape[1])]

VIF_df_main = VIF_df[1:]
maxVIF_df = VIF_df_main[VIF_df_main['VIF']==VIF_df_main['VIF'].max()]
maxVIF = maxVIF_df.VIF.to_string(index=False)

sys.stdout.write('Max VIF is' + str(maxVIF) + '\n')

