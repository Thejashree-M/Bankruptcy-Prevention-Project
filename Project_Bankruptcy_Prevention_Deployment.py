#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


# In[2]:


st.title('Model Deployment : Bankruptcy Prediction')
st.sidebar.header(' User Input Parameters')


# In[28]:


industrial_risk=st.sidebar.selectbox('industrial_risk',['0','0.5','1'])
management_risk=st.sidebar.selectbox(' management_risk',['0','0.5','1'])
financial_flexibility=st.sidebar.selectbox(' financial_flexibility',['0','0.5','1'])
credibility=st.sidebar.selectbox(' credibility',['0','0.5','1'])
competitiveness=st.sidebar.selectbox(' competitiveness',['0','0.5','1'])
operating_risk=st.sidebar.selectbox(' operating_risk',['0','0.5','1'])
data= {'industrial_risk' : industrial_risk,' management_risk':management_risk, ' financial_flexibility': financial_flexibility,
       ' credibility' : credibility,  ' competitiveness' : competitiveness, ' operating_risk' : operating_risk  }
features =pd.DataFrame(data,index= [0])


# In[29]:


st.subheader('User Input parameters')
st.write(features)


# In[30]:


data=pd.read_csv('bankruptcy-prevention.csv', delimiter=';')


# In[31]:


data.drop_duplicates(inplace=True)


# In[32]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
data[' class']=label.fit_transform(data[' class'])


# In[33]:


from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=42)
X=data.drop(' class',axis=1)
Y=data[' class']
X_resampled,Y_resampled=sm.fit_resample(X,Y)
data=pd.concat([X_resampled, Y_resampled],axis=1)


# In[34]:


X=data.drop(' class', axis=1)
Y=data[' class']


# In[35]:


modelRF=RandomForestClassifier(criterion='entropy', max_features=1,max_depth=3, n_estimators=100, oob_score=True, random_state=42)
modelRF.fit(X,Y)


# In[36]:


prediction=modelRF.predict(features)


# In[46]:


st.write('prediction :', prediction)
if prediction==0:
    st.write('Go Bankrupt')
else:
    st.write('Doesnot Go Bankrupt')


# In[47]:


prediction_proba = modelRF.predict_proba(features)
st.subheader('Prediction Probability for class bankrupt(0) and non bankrupt(1)')
st.write(prediction_proba)


# In[42]:




