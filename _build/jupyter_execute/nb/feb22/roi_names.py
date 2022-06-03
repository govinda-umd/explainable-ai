#!/usr/bin/env python
# coding: utf-8

# # Feb 23, 2022: roi names

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


roi_names_file = f"/home/govindas/parcellations/MAX_85_ROI_masks/README_MAX_ROIs_final_gm_85.txt"
roi_names_df = pd.read_csv(roi_names_file, delimiter='\t')
roi_names_df


# In[8]:


roi_names = []
for idx_roi in np.arange(len(roi_names_df)):
    roi_names.append(f"{' '.join(roi_names_df.iloc[idx_roi][['Hemi', 'ROI']].values)}")

# 
with open(f"/home/govindas/parcellations/MAX_85_ROI_masks/ROI_names.txt", 'w') as f:
    for roi_name in roi_names:
        f.write(f"{roi_name}\n")

