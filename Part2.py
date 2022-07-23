#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pandas as pd
import numpy as np


# In[3]:


df = pd.read_json('transaction-data-adhoc-analysis.json')
df['transaction_date'] =  pd.to_datetime(df['transaction_date'], format="%Y/%m/%d")
df


# In[4]:


user_purchase = df.groupby(['name',df['transaction_date'].dt.month.rename('Month')])['transaction_value'].sum().reset_index()
user_purchase


# In[5]:


user_purchase.groupby('Month')['name'].count()


# In[6]:


retention = pd.crosstab(user_purchase['name'], user_purchase['Month'])
retention


# In[7]:


repeaters_jan = 0
repeaters_janTofeb = retention[(retention[2]>0) & (retention[1]>0)][2].count() 
repeaters_febTomar = retention[(retention[3]>0) & (retention[2]>0)][3].count()
repeaters_marToapr = retention[(retention[4]>0) & (retention[3]>0)][4].count()
repeaters_aprTomay = retention[(retention[5]>0) & (retention[4]>0)][5].count()
repeaters_mayTojun = retention[(retention[6]>0) & (retention[5]>0)][6].count()


# In[8]:


inactive_jan = 0
inactive_feb = retention[(retention[2]==0) & (retention[1]>0)][2].count()
inactive_mar = retention[(retention[3]==0) & ((retention[2]>0) | (retention[1]>0))][3].count()
inactive_apr = retention[(retention[4]==0) & ((retention[3]>0) | (retention[2]>0) | (retention[1]>0))][4].count()
inactive_may = retention[(retention[5]==0) & ((retention[4]>0) | (retention[3]>0) | (retention[2]>0) | (retention[1]>0))][5].count()
inactive_jun = retention[(retention[6]==0) & ((retention[5]>0) | (retention[4]>0) | (retention[3]>0) | (retention[2]>0) | (retention[1]>0))][6].count()


# In[9]:


engaged_jan = retention[(retention[1]>0)][1].count()
engaged_feb = retention[(retention[2]>0) & (retention[1]>0)][2].count()
engaged_mar = retention[(retention[3]>0) & (retention[2]>0) & (retention[1]>0)][3].count()
engaged_apr = retention[(retention[4]>0) & (retention[3]>0) & (retention[2]>0) & (retention[1]>0)][4].count()
engaged_may = retention[(retention[5]>0) & (retention[4]>0) & (retention[3]>0) & (retention[2]>0) & (retention[1]>0)][5].count()
engaged_jun = retention[(retention[6]>0) & (retention[5]>0) & (retention[4]>0) & (retention[3]>0) & (retention[2]>0) & (retention[1]>0)][6].count()


# In[10]:


array = np.array([[repeaters_jan, repeaters_janTofeb,repeaters_febTomar, repeaters_marToapr, repeaters_aprTomay, repeaters_mayTojun] 
                 , [inactive_jan, inactive_feb, inactive_mar, inactive_apr, inactive_may, inactive_jun]
                 , [engaged_jan, engaged_feb, engaged_mar, engaged_apr, engaged_may, engaged_jun]])


# In[11]:


Final_Table = pd.DataFrame(data = array, 
                        index = ['Repeaters', 'Inactive', 'Engaged'],
                        columns = ['January', 'February', 'March', 'April', 'May', 'June'])
Final_Table


# In[ ]:




