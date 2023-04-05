#!/usr/bin/env python
# coding: utf-8

#  ## 1. Loading Packages

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
# %matplotlib inline


# ## 2. Loading Data & Basic Analysis

# In[4]:


df = pd.read_csv('API_19_DS2_en_csv_v2_5361599.csv', skiprows = 4)
data = df.copy()


# ### Basic Structure

# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.tail()


# ## Statistics 

# In[8]:


df.describe()


# In[10]:


df.describe(include='all')


# In[11]:


df.describe(include='object')


# ### Droping unnamed column

# ### NULL Values

# In[12]:


df.isnull().sum()


# ### Data Types

# In[13]:


df.dtypes


# In[14]:


df.describe(include = ["O"])


# In[15]:


df.describe(include="all")


# ### Ingest and manipulate the data using pandas dataframes.
# include a function which takes a filename as argument, reads a dataframe in World bank format and returns two dataframes: one with years as columns and one with
# countries as columns.

# In[16]:



def read_world_bank_data(filename):
    # read in the data and skip the first 4 rows
    df = pd.read_csv(filename, skiprows=4)
    
    # drop any columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)
    
    # rename the 'Country Name' column to 'Country'
    df = df.rename(columns={'Country Name': 'Country'})
    
    # remove any rows that don't have a country name
    df = df[df['Country'].notna()]
    
    # pivot the data to have years as columns and countries as rows
    df_years = df.pivot(index='Country', columns='Indicator Name')
    df_years.columns = df_years.columns.droplevel(0)
    
    # transpose the data to have countries as columns and years as rows
    df_countries = df_years.transpose()
    
    return df_years, df_countries


# In[17]:


df_years,df_countries = read_world_bank_data('API_19_DS2_en_csv_v2_5361599.csv')


# In[19]:


df_countries


# In[17]:


df_countries


# ### Explore the statistical properties of a few indicators, that are of interest to you, and
# cross-compare between individual countries and/or the whole world (you do not
# have to do all the countries, just a few will do) and produce appropriate summary
# statistics. You can also use aggregated data for regions and other categories. You
# are expected to use the .describe() method to explore your data and two other
# statistical methods

# In[18]:


# Filter the data to only include rows where the indicator name is "GDP per capita (constant 2010 US$)"
gdp_per_capita = df[df['Indicator Name'] == 'GDP per capita (constant 2010 US$)']

# Print the first 5 rows of the filtered data
print(gdp_per_capita.head())  


# In[19]:


# Select the indicators, countries, and years of interest
indicators = ['GDP per capita (current US$)', 'Life expectancy at birth, total (years)', 'CO2 emissions (metric tons per capita)']
countries = ['AFG', 'AFE', 'AFW']
years = [str(year) for year in range(2010, 2021)]

# Select the relevant columns and filter the data
df = df[['Country Name', 'Country Code', 'Indicator Name'] + years]
df = df.loc[(df['Indicator Name'].isin(indicators)) & (df['Country Code'].isin(countries))]

# Melt the DataFrame to long format
df = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name'], var_name='Year', value_name='Value')

# Convert Year to int
df['Year'] = df['Year'].astype(int)

# Compute summary statistics for each indicator and country
summary_stats = df.groupby(['Indicator Name', 'Country Code'])['Value'].describe()

# Print the summary statistics
print(summary_stats)


# In[20]:


df.columns


# In[21]:


data


# In[22]:


data.columns


# In[23]:


# Drop unnecessary columns
data = data.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])

# Melt the data
data = pd.melt(data, id_vars=['Country Name'], var_name='Year', value_name='Value')


# In[24]:


data


# In[25]:


data.isnull().sum()


# In[26]:


data = data.fillna(data.median())


# In[27]:


data.head(50)


# In[28]:


year_wise=data.groupby('Year')['Value'].sum().reset_index()


# In[29]:


year_wise


# In[52]:


fig = plt.subplots(figsize=(50, 40))
sns.lineplot(x='Value', y='Year', data=year_wise,marker='o').set(title='Year Wise Co2 emission', xlabel='Emission values', ylabel='Year')
sns.set_theme(style='white', font_scale=3)


# In[30]:


Year_country_wise=data.groupby(['Country Name','Year'])['Value'].sum().reset_index()


# In[31]:


Year_country_wise


# In[60]:


fig = plt.subplots(figsize=(50, 40))
sns.lineplot(x='Value', y='Year', data=Year_country_wise,marker='o',hue='Country Name').set(title='Year Wise Co2 emission', xlabel='Emission values', ylabel='Year')
sns.set_theme(style='white', font_scale=3)


# In[32]:


temp=data.groupby('Country Name')['Value'].sum().reset_index()


# In[34]:


temp=temp.sort_values('Value',ascending=False)


# In[35]:


temp


# In[36]:


plt.figure(figsize=(20,10))
ax=sns.barplot(x='Country Name',y='Value',data=temp.head(10));
plt.xlabel('Country Name')
plt.ylabel('Emission Value')
plt.title('Top 10 countries with high CO2 emission')


# In[37]:


temp=temp.sort_values('Value',ascending=True)


# In[38]:


temp


# In[39]:


plt.figure(figsize=(20,10))
ax=sns.barplot(x='Country Name',y='Value',data=temp.head(10));
plt.xlabel('Country Name')
plt.ylabel('emission Value')
plt.title('Top 10 countries with Lowest CO2 emission')


# In[40]:


data.columns


# In[41]:


corr_matrix = df[['Year', 'Value']].corr()


# In[42]:


corr_matrix


# In[ ]:


sns.set_context('poster')
sns.catplot(kind='box', x='Value', col='Country Name', row='Year', data= data);


# In[ ]:


# Compute correlation matrix
corr_matrix = data.corr()

# Display the correlation matrix
print(corr_matrix)


# In[ ]:


# Create a heatmap of the correlation matrix
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)


# In[ ]:


def explore_data(df):
    # get the summary statistics for the data
    summary_stats = df.describe()
    
    # calculate the mean and median for each indicator
    mean_df = df.mean()
    median_df = df.median()
    
    # calculate the standard deviation for each indicator
    std_df = df.std()
    
    # return the summary statistics and the mean, median, and standard deviation dataframes
    return summary_stats, mean_df, median_df, std_df


# In[ ]:


def explore_correlations(df):
    # calculate the correlation matrix for the data
    corr_matrix = df.corr()
    
    return corr_matrix


# In[ ]:


def plot_data(df, Country):
    # plot the data for a specific country
    plt.plot(df.loc[Country])
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(Country)
    plt.show()


# In[ ]:


plot_data(df, Country)


# In[ ]:


def plot_time_series(df, indicator, countries):
    # plot the time series for a specific indicator and list of countries
    df.loc[countries, indicator].transpose().plot()
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('Time Series Plot')
    plt.legend()
    plt.show()


# In[ ]:


explore_correlations(df)


# In[ ]:


df.columns


# In[ ]:




