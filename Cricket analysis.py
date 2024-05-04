#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('downloads/cricket.csv')
df.head()


# In[5]:


df.info()


# In[9]:


df.duplicated()
df.drop_duplicates(inplace= True)


# In[10]:


df.isnull().sum()


# In[12]:


df.rename(columns={'current_innings':'batting_team'}, inplace= True)
df['batting_team'].unique()


# In[14]:


country = df['batting_team'].unique()
print(country)


# In[16]:


print(df['match_name'].unique())


# In[17]:


# Split 'match_name' into two new columns: 'team1' and 'team2'
df[['team1', 'team2']] = df['match_name'].str.split(' v ', expand=True)

# Add a new column 'bowling_team' with the team not present in 'current_inning'
df['bowling_team'] = df.apply(lambda row: row['team2'] if row['batting_team'] == row['team1'] else row['team1'], axis=1)

df.drop(columns= ['team1', 'team2'], inplace= True )
df['bowling_team'].unique()


# In[18]:


def filtered_df(details, columns, top_teams):
    condition = df['batting_team'].isin(top_teams) & df['bowling_team'].isin(top_teams)
    return df.loc[condition, columns]


# In[21]:


batting_df1= ['batting_team','batsman1_name','batsman1_runs']
for i in country:
    filtered1 = filtered_df(df,batting_df1,country)
filtered1.rename(columns={'batsman1_name':'batsman_name'}, inplace=True)
filtered1=filtered1.groupby('batsman_name').max().sort_values(by='batsman1_runs',ascending=False)[:10]
filtered1


# In[22]:


batting_df2= ['batting_team','batsman2_name','batsman2_runs']
for i in country:
        filtered2 = filtered_df(df,batting_df2,country)
filtered2.rename(columns={'batsman2_name':'batsman_name'}, inplace=True)
filtered2=filtered2.groupby('batsman_name').max().sort_values(by='batsman2_runs',ascending=False)[:10]
filtered2


# In[23]:


# Merge the dataframes on the 'batsman1_name' and 'batsman2_name' columns

merged_df = pd.merge(filtered1, filtered2, on=('batsman_name','batting_team'))
merged_df


# In[24]:


#Add the corresponding columns and
merged_df['batsman_runs'] = merged_df['batsman1_runs'] + merged_df['batsman2_runs']
 #Dropping the extra columns
merged_df.drop(columns=['batsman1_runs','batsman2_runs'], inplace= True)
merged_df 

# Display the result
result = merged_df.groupby('batsman_name').max().sort_values(by='batsman_runs',ascending=False)[:10]
result


# In[34]:


#Top high scorer in batsman each team

#sorting by score
result_sorted = result.sort_values(by='batsman_runs', ascending=False)

#dropping lowers score
team_result = result_sorted.drop_duplicates(subset=['batting_team'])
team_result


# In[51]:


# Highest Score of Every Indian Batsman

df.loc[df['batting_team'] =='INDIA',['batting_team','batsman1_name','batsman1_runs']].groupby('batsman1_name').max().sort_values(by='batsman1_runs',ascending=False)[:5]


# In[52]:


top_runs = df[['batsman1_name','runs']].loc[df['batting_team']=='INDIA'].groupby('batsman1_name').sum().sort_values(by='runs',ascending=False)[0:5]
top_run_getters = top_runs.index
top_run_getters


# In[53]:


top5_ind = df[['match_name','batsman1_name','batsman1_runs']].loc[(df['batting_team']=='INDIA') & (df['batsman1_name'].isin(top_run_getters))].groupby(['match_name','batsman1_name']).max().sort_index().unstack()
top5_ind = top5_ind.cumsum()
top5_ind


# In[54]:


plt.figure(figsize=(10,6))
plt.plot(top5_ind[('batsman1_runs','Virat Kohli')], label='Virat Kohli', color='Green', linewidth=2)
plt.plot(top5_ind[('batsman1_runs','Suryakumar Yadav')], label='Suryakumar Yadav', color='#242F9B', linewidth=2)
plt.plot(top5_ind[('batsman1_runs','Hardik Pandya')], label='Hardik Pandya', color='#242F9B', linewidth=2, linestyle=':')
plt.plot(top5_ind[('batsman1_runs','KL Rahul')], label='KL Rahul', color='#242F9B', linewidth=2, linestyle='--')
plt.plot(top5_ind[('batsman1_runs','Rohit Sharma')], label='Rohit Sharma', color='#FBCB0A', linewidth=2)

plt.title(" India's top 5 run getters ")
plt.xticks(rotation=90)
plt.xlabel('Versus')
plt.ylabel('Total Runs')
plt.legend()
plt.grid()
plt.savefig('India top scorers', bbox_inches='tight')
plt.show()


# In[55]:


bowling_df= ['bowling_team','bowler1_name','bowler1_wkts']
for i in country:
     filtered = filtered_df(df,bowling_df,country)
filtered.groupby('bowler1_name').max().sort_values(by='bowler1_wkts',ascending=False)[:10]


# In[57]:


df.loc[df['bowling_team'] =='INDIA',['bowling_team','bowler1_name','bowler1_wkts']].groupby('bowler1_name').max().sort_values(by='bowler1_wkts',ascending=False)[:5]


# In[58]:


IND_results=df.loc[(df['home_team']=='INDIA') | (df['away_team']=='INDIA'), :].copy()
IND_results.drop(columns=['home_team','away_team'], inplace= True)
IND_results.columns


# In[59]:


ind_match_id=IND_results['match_id']

ind_match_id


# In[60]:


ind_df = df.loc[df['match_id'].isin(ind_match_id)]
ind_bowling = ind_df.loc[(ind_df['batting_team']!='INDIA') & (ind_df['wkt_text']!="")]

ind_top_bowlers = ind_bowling['bowler1_name'].value_counts()[:].index
ind_top_bowlers


# In[61]:


ind_bowling.sample(5)


# In[62]:


ind_bowling = ind_bowling.loc[ind_bowling['bowler1_name'].isin(ind_top_bowlers)]
ind_bowling = ind_bowling[['match_id','bowler1_name','bowler1_wkts']].groupby(['match_id','bowler1_name']).max().unstack()
ind_bowling=ind_bowling.cumsum()
ind_bowling


# In[63]:


plt.figure(figsize=(10,6))
# plt.style.use('seaborn')
plt.plot(ind_bowling['bowler1_wkts','Arshdeep Singh'],color='#06283D',linewidth=2, label='Arshdeep Singh')
plt.plot(ind_bowling['bowler1_wkts','Axar Patel'],color='#FFA500',linewidth=2, label='Axar Patel')
plt.plot(ind_bowling['bowler1_wkts','Bhuvneshwar Kumar'],color='#1363DF',linewidth=2, label='Bhuvneshwar Kumar')
plt.plot(ind_bowling['bowler1_wkts','Hardik Pandya'],color='#1363DF',linewidth=2, linestyle='--', label='Hardik Pandya')
plt.plot(ind_bowling['bowler1_wkts','Mohammed Shami'],color='#FFA500',linestyle='--',linewidth=2, label='Mohammed Shami')
plt.xlabel('versus')
plt.ylabel('Total wickets')
plt.xticks(rotation=90)
plt.title("How India's top 5 wicket takers")
plt.legend()
plt.grid(True)
# plt.savefig('top wicket takers.png',bbox_inches='tight')
plt.show()


# In[65]:


wkt_df= df[['match_id','match_name','batting_team','bowling_team','wicket_id', 'wkt_batsman_name', 'wkt_bowler_name', 'wkt_batsman_runs',
       'wkt_batsman_balls', 'wkt_text']]
wkt_df


# In[66]:


column_list = ['match_id', 'match_name', 'batting_team', 'bowling_team', 'wicket_id',
               'wkt_batsman_name', 'wkt_bowler_name', 'wkt_batsman_runs', 'wkt_batsman_balls', 'wkt_text']

wkt_df = filtered_df(df, column_list, country)
wkt_df


# In[67]:


wkt_df.isnull().sum()


# In[68]:


wkt_df.info()


# In[69]:


wkt_df1= wkt_df.dropna(subset=['wicket_id'],inplace= True)


# In[70]:


wkt_df


# In[ ]:




