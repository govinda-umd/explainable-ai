#!/usr/bin/env python
# coding: utf-8

# # Feb 12, 22

# In[1]:


import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 9
plt.rcParams["errorbar.capsize"] = 0.5

from nilearn import datasets

# statistical image
motor_images = datasets.fetch_neurovault_motor_task()
stat_img = motor_images.images[0]
# In[2]:


# vector of activity for each roi
num_rois = 85
roi_data = np.random.rand(num_rois)


# In[3]:


# from nilearn.input_data import NiftiLabelsMasker
from nilearn import image
mask_file = f"/home/govindas/parcellations/templates/MAX_ROIs_final_gm_85.nii.gz"
print(mask_file)
mask = image.load_img(mask_file)
# masker = NiftiLabelsMasker(labels_img=mask_file, standardize=False)


# In[4]:


from nilearn import plotting
plotting.plot_stat_map(mask_file, colorbar=False)


# In[5]:


from nilearn import masking
from nilearn import image

stat_img_all_rois = image.new_img_like(ref_niimg=mask, 
                                       data=np.zeros_like(mask.get_fdata(), 
                                                          dtype=np.float32), 
                                       copy_header=True)
#image.math_img(f"img==0", img=mask_file)

for idx_roi in tqdm(np.arange(roi_data.shape[0])):
    mask_roi = image.math_img(f"img=={idx_roi+1}", img=mask)
    num_voxels = np.where(mask_roi.get_fdata())[0].shape[0]
    vox_data = roi_data[idx_roi] * np.ones(shape=(num_voxels,))
    stat_img = masking.unmask(vox_data, mask_img=mask_roi)
    stat_img_all_rois = image.math_img(f"img_all+img_roi", img_all=stat_img_all_rois, img_roi=stat_img)


# In[6]:


stat_img_all_rois.to_filename('./figures/stat_img_all_rois.nii.gz')


# In[7]:


plotting.plot_stat_map(stat_img_all_rois, colorbar=True)
# stat_img_all_rois.get_fdata().shape

from nilearn import datasets

# surface mesh
fsaverage = datasets.fetch_surf_fsaverage()
fsaveragefrom nilearn import surface

# map the image (in volume) to the surface
texture = surface.vol_to_surf(img=stat_img, surf_mesh=fsaverage.pial_right)from nilearn import plotting

# plot the (surface) texture of the mapped image over a surface.
plotting.plot_surf_stat_map(surf_mesh=fsaverage.infl_right, stat_map=texture, 
                            hemi='right', title='Surface right hemisphere', 
                            threshold=1.0, bg_map=fsaverage.sulc_right)plotting.plot_img_on_surf(stat_map=stat_img, 
                          hemispheres=['left', 'right'], 
                          views=['lateral', 'medial', 'dorsal', 'ventral'], 
                          threshold=1.0, colorbar=False)plotting.plot_glass_brain(stat_map_img=stat_img, display_mode='ortho', plot_abs=False, title='Glass brain', threshold=1.0, colorbar=True)from nilearn import image
parcellation_file = f"/home/govindas/parcellations/templates/MAX_ROIs_final_gm_85.nii.gz"
parcellation = image.load_img(parcellation_file)
# this gives open boundary roi surfaces, rois should have closed boundaries.
# parcellation_surf = surface.vol_to_surf(img=parcellation, surf_mesh=fsaverage.pial_right)# plotting.plot_surf_stat_map(surf_mesh=fsaverage.infl_right, stat_map=parcellation_surf, bg_map=fsaverage.sulc_right, threshold=20.0, colorbar=True)
plotting.plot_glass_brain(stat_map_img=parcellation, display_mode='ortho', plot_abs=False, threshold=2.0, colorbar=False)plotting.plot_stat_map(stat_map_img=parcellation, colorbar=False)plotting.plot_surf_contours(surf_mesh=fsaverage.infl_right, roi_map=parcellation_surf, colorbar=False)