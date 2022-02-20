#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import shap
import sklearn
import matplotlib.pyplot as plt


# ## linear model

# In[2]:


X, y = shap.datasets.boston()
X100 = shap.utils.sample(X, 100)

# linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)


# In[4]:


model.get_params()


# In[5]:


for i in range(X.shape[1]):
    print(X.columns[i], '=', model.coef_[i].round(4))


# In[6]:


X100


# In[9]:


shap.plots.partial_dependence(6, model.predict, X100, ice=False,
                              model_expected_value=True, feature_expected_value=True)


# In[47]:


explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X)


# In[27]:


sample_inds = [23]
shap.plots.partial_dependence(5, model.predict, X100, 
                              model_expected_value=True, feature_expected_value=True, 
                              ice=False, 
                              shap_values=shap_values[sample_inds, :])


# In[28]:


shap.plots.scatter(shap_values[:, 5])


# In[29]:


print(model.predict(X)[sample_inds])
print(shap_values.base_values[sample_inds])


# In[50]:


s = shap_values[sample_inds]
s.base_values = s.base_values[0]
s.values = s.values[0]
s.data = s.data[0]
shap.waterfall_plot(s, max_display=14)


# In[49]:


shap_values[sample_inds]


# ## generalized additive regression model

# In[54]:


# fit a GAM model to the data
import interpret.glassbox
model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
model_ebm.fit(X, y)

# explain the GAM model with SHAP
explainer_ebm = shap.Explainer(model_ebm.predict, X100)
shap_values_ebm = explainer_ebm(X)


# In[56]:


# make a standard partial dependence plot with a single SHAP value overlaid
shap.partial_dependence_plot(5, model_ebm.predict, X, 
                            model_expected_value=True, 
                            feature_expected_value=True, 
                            ice=False, 
                            shap_values=shap_values_ebm[sample_inds, :])


# In[71]:


shap.plots.scatter(shap_values_ebm[:, 5], color=shap_values_ebm)


# In[64]:


s = shap_values_ebm[sample_inds]
s.base_values = s.base_values[0]
s.values = s.values[0]
s.data = s.data[0]
shap.waterfall_plot(s, max_display=14)


# In[66]:


# ============================
# DID NOT UNDERSTAND THIS PLOT
# ============================
shap.plots.beeswarm(shap_values_ebm, max_display=14)


# ## non-additive boosted tree model

# In[68]:


# train XGBoost model
import xgboost
model_xgb = xgboost.XGBRegressor(n_estimators=100, max_depth=2).fit(X, y)

# explain the GAM model with SHAP
explainer_xgb = shap.Explainer(model_xgb, X100)
shap_values_xgb = explainer_xgb(X)


# In[70]:


# make a standard partial dependence plot with a single SHAP value overlaid
fig,ax = shap.partial_dependence_plot(
    "RM", model_xgb.predict, X, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False,
    shap_values=shap_values_ebm[sample_inds,:]
)


# In[73]:


shap.plots.scatter(shap_values_xgb[:, 5], color=shap_values_xgb)


# ## linear logistic regression model

# In[74]:


# a classic adult census dataset price dataset
X_adult,y_adult = shap.datasets.adult()

# a simple linear logistic model
model_adult = sklearn.linear_model.LogisticRegression(max_iter=10000)
model_adult.fit(X_adult, y_adult)


# In[75]:


def model_adult_proba(x):
    return model_adult.predict_proba(x)[:,1]
def model_adult_log_odds(x):
    p = model_adult.predict_log_proba(x)
    return p[:,1] - p[:,0]


# In[77]:


X_adult


# In[78]:


y_adult


# In[79]:


shap.partial_dependence_plot(8, model_adult_proba, X_adult, 
                             model_expected_value=True, 
                             feature_expected_value=True, ice=False)


# In[83]:


background_adult = shap.utils.sample(X_adult, 100)
background_adult

explainer = shap.Explainer(model_adult_proba, background_adult)
shap_values_adult = explainer(X_adult[:1000])


# In[85]:


background_adult = shap.maskers.Independent(X_adult, max_samples=100)
background_adult

explainer = shap.Explainer(model_adult_proba, background_adult)
shap_values_adult = explainer(X_adult[:1000])


# In[92]:


shap.plots.bar(shap_values_adult.abs.max(0), max_display=15)


# In[102]:


shap_values.max(0)


# ## non-additive boosted tree logistic regression model

# In[ ]:




