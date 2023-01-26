#!/usr/bin/env python
# coding: utf-8

# ### Libs

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[3]:


palette = sns.color_palette("ch:2.5,-.1,dark=.1")


# ### Carrega datasets

# In[4]:


df_sus = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/mama/Mortalidade_Mama_SIM-SUS.csv',
    encoding='ISO-8859-1',
    index_col=0,
    low_memory=True
)


# In[5]:


print(df_sus.shape)
df_sus.head()


# ### Nulos

# In[7]:


null_count = df_sus.isna().sum()
null_percentage = round(100 * (null_count / len(df_sus)), 3)
null_tbl = pd.DataFrame(
    data=[null_count, null_percentage, df_sus.dtypes],
    index=['null_count', 'null_percentage', 'types']).T.sort_values(ascending=False, by='null_percentage'
)


# In[8]:


null_tbl

