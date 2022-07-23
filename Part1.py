#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import numpy.linalg as la
import scipy.linalg as spla
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('transaction-data-adhoc-analysis.json')


# In[3]:


b = df['transaction_value']

pat = r'(?P<category>.*),(?P<item>.*),\(x(?P<quantity>\d+)\)'
df_new = (df.pop('transaction_items').str.split(';').explode().str.extract(pat).join(df)
        .astype({'quantity': 'int', 'transaction_date': 'datetime64'})
        .assign(month=lambda x: x['transaction_date'].dt.month_name()))
df_new.pop('transaction_value')

pivot_item = pd.pivot(df_new[['item', 'quantity']],
             columns = 'item',
             values = 'quantity').fillna(0)
df_new


# In[3]:


pivot_item


# In[4]:


a = pd.DataFrame(pivot_item.values)

b = b[:16608]
a = a[:16608]


Q, R = la.qr(a)
x = spla.solve_triangular(R, Q.T.dot(b), lower=False)
Q2, R2 = la.qr(a, mode="complete")
x2 = spla.solve_triangular(R[:7], Q.T[:7].dot(b), lower=False)
x3 = la.solve(a.T.dot(a), a.T.dot(b))

#source for this is: https://andreask.cs.illinois.edu/cs357-s15/public/demos/06-qr-applications/Solving%20Least-Squares%20Problems.html


item_dict = {}
for c, i in zip(pivot_item.columns, range(len(pivot_item.columns))):
    item_dict[c] = x[i]

df_new['unit_price'] = df_new['item'].map(item_dict)

df_new['total_sales_value'] = df_new['quantity'] * df_new['unit_price']


# In[8]:


df_pivot_jan = pd.pivot_table(df_new[df_new['month']=='January'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nJanuary Sale Statistics" + "\033[0m")
print(df_pivot_jan)


# In[5]:


df_pivot_feb = pd.pivot_table(df_new[df_new['month']=='February'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nFebruary Sale Statistics" + "\033[0m")
print(df_pivot_feb)


# In[6]:


df_pivot_mar = pd.pivot_table(df_new[df_new['month']=='March'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nMarch Sale Statistics" + "\033[0m")
print(df_pivot_mar)

df_pivot_apr = pd.pivot_table(df_new[df_new['month']=='April'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nApril Sale Statistics" + "\033[0m")
print(df_pivot_apr)


# In[5]:


df_pivot_jan = pd.pivot_table(df_new[df_new['month']=='January'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nJanuary Sale Statistics" + "\033[0m")
print(df_pivot_jan)

df_pivot_feb = pd.pivot_table(df_new[df_new['month']=='February'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nFebruary Sale Statistics" + "\033[0m")
print(df_pivot_feb)

df_pivot_mar = pd.pivot_table(df_new[df_new['month']=='March'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nMarch Sale Statistics" + "\033[0m")
print(df_pivot_mar)

df_pivot_apr = pd.pivot_table(df_new[df_new['month']=='April'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nApril Sale Statistics" + "\033[0m")
print(df_pivot_apr)

df_pivot_may = pd.pivot_table(df_new[df_new['month']=='May'], values=['quantity','unit_price','total_sales_value'], index='item', aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nMay Sale Statistics" + "\033[0m")
print(df_pivot_may)

df_pivot_jun = pd.pivot_table(df_new[df_new['month']=='June'], values=['quantity','unit_price','total_sales_value'], index='item',aggfunc={'quantity' : 'sum', 'unit_price' : 'mean', 'total_sales_value' : 'sum'})
print("\033[1m" + "\nJune Sale Statistics" + "\033[0m")
print(df_pivot_jun)


# In[8]:


df_total = pd.pivot_table(data=df_new,index='item',values='total_sales_value',aggfunc=np.sum)
df_total = df_total.rename(columns={'total_sales_value': 'Total Sales Value'}, level=0)
df_total = df_total.sort_values('Total Sales Value', ascending=False)
print("\033[1m" + "\nTotal Revenue Per Item (January to June)" + "\033[0m")
print(df_total)
print("\n")

plt.bar(df_total.index,df_total['Total Sales Value'])
pd.options.display.float_format = '{:.2f}'.format
plt.xticks(rotation=70) 
plt.xlabel('Food Item') 
plt.ylabel('Revenue') 
plt.ticklabel_format(style='plain', useOffset=False, useLocale=True, axis='y')
plt.title('Total Revenue Per Item (January to June)')
plt.show()



# In[ ]:




