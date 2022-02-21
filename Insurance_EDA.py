#!/usr/bin/env python
# coding: utf-8

# # Insurance Price - EDA
# 
# The aim of this work is to provide a series of considerations on the given population by going to analyses the attributes and the relatioship between them. The data used is Medical Costs Datasets (available on github from here: https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv) and contains individual medical costs billed by health insurance.
# 
# The population is characterised by the following features:
# 
# * AGE: age of primary beneficiary
# * SEX: insurance contractor gender
# * BMI: body mass index, providing an understanding of body using the ratio of height to weight
# * CHILDREN: number of children covered by health insurance
# * SMOKER: if the individual smoks or not
# * REGION: the beneficiary's residential area in the US
# 
# The target of our analisys is:
# * CHARGES: Indivudual medical costs billed by health insurance

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


url = 'https://raw.githubusercontent.com/Emanuele-Cali/MedicalCosts/main/insurance.csv'
df = pd.read_csv(url)


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# There are no missing values and for this reason no data cleanining is needed. 
# 
# We create a new categorical data using the bmi value. In particular we assign every benefiary to bmi_class according with the cdc guidelines to make some more analysis (https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/english_bmi_calculator/bmi_calculator.html)
# 
# The new column will be:
# * BMI_CLASSIFICATION: classification in Underweight, Healthy, Overweight, Obese

# In[7]:


df.loc[df.bmi<18.5, 'bmi_classification'] = 'Underweight'
df.loc[(df.bmi>= 18.5) & (df.bmi<=24.9), 'bmi_classification'] = 'Healthy'
df.loc[(df.bmi>24.9) & (df.bmi <=29.9), 'bmi_classification'] = 'Overweight'
df.loc[df.bmi>29.9, 'bmi_classification'] = 'Obese'


# In[8]:


df.head(3)


# #### Usufull table

# In[9]:


male = df['sex'] == 'male'
ages = df.sort_values('age').age.unique()
smoker = df['smoker']=='yes'
regions = df['region'].unique()


# ## Explorotary Data Analisys

# In this part of the survey we will use a lot of plots to understand the relationship between the variables. We will proceding taking a variable and than looking for relationship between this variable and the other.

# ### Feature : Age

# #### Age frequencies

# We start to see the age frequences for male e female, the age distribution and the proportion between male and female

# In[10]:


sex_age = pd.crosstab(df.age, df.sex)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,12))

ax1 = plt.subplot(212)
sex_age.plot(kind = 'bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Age frequence for male and female')

ax2 = plt.subplot(222)
ax2.margins()
df.groupby('sex').age.count().plot(kind='bar', ax=ax2)

ax2.set_title('Sex frequence')
ax3 = plt.subplot(221)
ax3.margins()
df.boxplot(by = 'sex', column='age', ax=ax3)
ax3.set_title('Age distribution')

plt.show()


# We can see from this plots that we have a peak on 18 and 19 years where we see ~67 insurances and than the frquence will drop on ~30 insurances. The sex distribution along the years is comparable between male and female. 
# We have a comparable proportion between male and female, we have 676 male and 662 female.
# From the boxplot we see that males are a little jounger than the females.

# #### Age frequencies on regions

# In[11]:


fig, ax=plt.subplots(3,4, figsize=(23,12))

l = 0
for i in regions:
    df_region_i = df[df['region'] == i]
    ax[0,l].bar(df_region_i.age.value_counts().index, df_region_i.age.value_counts(), label=i)
    ax[0,l].set_title('Age frequenz for '+ i)
    
    ax[1,l].boxplot((df_region_i[male].age, df_region_i[~male].age))
    ax[1,l].set_xticklabels(['male','female'], rotation=0)
    ax[1,l].set_title('Age boxpot for '+ i)
    ax[1,l].grid()
    
    rap = ax[2,l].bar(df_region_i.groupby('sex').age.count().index, df_region_i.groupby('sex').age.count())
    
    for i in rap:
        tot = df_region_i.age.count()
        x,y = i.get_xy()
        al = i.get_height()
        la = i.get_width()
        ax[2,l].annotate(str(round(i.get_height()/tot*100,2))+'%', (x+la/2.5,y+al*1.007))
 
    l += 1

plt.show()


# #### Age and bmi classification

# Now we analyze the bmi classification along the years

# In[12]:


tab = pd.crosstab(df.age, df.bmi_classification, normalize='index')

f, a = plt.subplots(figsize = (21,6))

fig = tab.plot.bar(y = ['Underweight','Healthy','Overweight','Obese'], stacked=True, ax =a)
plt.title('bmi proportion')
plt.legend(bbox_to_anchor=[0.2, - 0.1], loc= 'center',ncol = 4)
plt.show()


# We can´t see a clear correlation between age and bmi classification, but we see that the most population fall always in the Obese group.

# #### Age and smoker

# Now we analyze the relationship between the smoker attribut with age and sex attributes. We will see the frequence of smokers along the years and the proportion of smoker on sex attribute.

# In[13]:


smoker_sex = pd.crosstab(df.sex, df.smoker, normalize = 'index')
smoker_age = pd.crosstab(df.age, df.smoker)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,12))

ax1 = plt.subplot(212)
smoker_age.plot(kind = 'bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Age Distribution between smoker and no smoker')

ax2 = plt.subplot(222)
ax2.margins()
smoker_sex.plot(kind='bar', stacked=True, ax=ax2)
ax2.set_title('Smoker attributes on sex')

ax3 = plt.subplot(221)
ax3.margins()
df.boxplot(by=['smoker'], column='age', ax = ax3)
ax3.set_title('Age Distribution for smoker')
plt.show()


# In this plots we see that the population is almost no smoker both for male and female, but we have more males than females that are smoker.
# The age of smoker is lower than for no smoker. 
# The distribution of smoker on the age has an irregular pattern.

# From this plots we can see the age frequence, the sex distribution and the age distribution on the 4 regions.
# We see the biggest sex differences is in southeast and the biggest difference on the age is in northwest.
# 
# From this first part of the analysis we have no strong difference in the age and sex distribution.

# #### Age and charges

# Now we analyze the relationship between age and charges dividing the population on sex.

# In[14]:


fig, ax=plt.subplots(3,1, figsize = (8, 14))

ax[0].scatter(df[male]['age'], df[male]['charges'], label = 'male', alpha = 0.5)
ax[0].scatter(df[~male]['age'], df[~male]['charges'], label = 'female', alpha = 0.5)
ax[0].set_title('Relation Age-Charges for male and female')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Charges')
ax[0].legend()

#ax[1].hist(df[male]['charges'], alpha= 0.3 ,label = 'male')
#ax[1].hist(df[~male]['charges'], alpha= 0.3 ,label = 'female')
df.groupby('sex').charges.plot(kind='hist', ax = ax[1], alpha = 0.3)
ax[1].set_title('Charge distribution')

ax[1].grid()
ax[1].legend()


df.boxplot(column='charges', by = 'sex', ax = ax[2])
#ax[2].set_xticklabels(['male','female'])
ax[2].set_title('Charge distribution')

plt.show()


# In the first plot we see the relationship between age and charges and we see the first interessant thing.
# There is a positive relationship between the age and charges, both for male and female. We can see 3 different groups on the charges but this grouping don´t depend on the sex, as we can see from the plot in the middle. There we can see only that we have more male than famale paying more than 30.000 USD.

# Now we analyze the attribute smoker to see if exist correlation with other variables.

# ### Feature: Smoker

# #### Smoker and bmi

# In[15]:


smoker_bmi_class = pd.crosstab(df.smoker, df.bmi_classification, normalize='index')

f, a = plt.subplots(1,2, figsize= (21,4))

smoker_bmi_class.plot(y = ['Underweight','Healthy','Overweight','Obese'], kind='bar',stacked=True, ax = a[0])
a[1].scatter(df[~smoker].age, df[~smoker].bmi, label ='no smoker', alpha = 0.8)
a[1].scatter(df[smoker].age, df[smoker].bmi, label ='smoker', alpha = 0.8)
#df.groupby('smoker').plot(x = 'age', y = 'bmi' , kind='scatter')
a[1].legend(loc = 1)
plt.show()


# From this plots we can´t see any clear relationship between smoker and bmi

# #### Smoker and regions

# In[16]:



f, a =plt.subplots(1,2,figsize=(21,4))

pd.crosstab(df.region, df.smoker, normalize='index').plot(kind='bar', stacked=True, ax=a[0])
df.boxplot(column = 'age', by = ['region','smoker'], ax=a[1])


# #### Relationship between charges and year dividing the population by the smoker feature

# Now we see if smoker has an impact on the relationship between years and charges.

# In[17]:


fig, ax=plt.subplots(3,1, figsize = (8, 14))

ax[0].scatter(df[~smoker]['age'], df[~smoker]['charges'], label = 'no smoker', alpha = 0.5)
ax[0].scatter(df[smoker]['age'], df[smoker]['charges'], label = 'smoker', alpha = 0.5)

ax[0].set_xlabel('Age')
ax[0].set_ylabel('Charges')
ax[0].set_title('Relation Age-Charges for smoker and no smoker')
ax[0].legend()

df.groupby('smoker').charges.plot(kind='hist', ax=ax[1], alpha = 0.3)
ax[1].grid()
ax[1].legend()
ax[1].set_title('Charge distribution')

df.boxplot(column = 'charges', by='smoker', ax = ax[2])
ax[2].set_title('Charge distribution')

plt.show()


# As we can see from the first plot, the smoke attribute divides the relationship between charge and age into 2 groups, no smoker downstair and smokerupstair . The no smoker pay lass than the smoker.
# 
# We can see from the plt in the middle that we have 3 different charge distribution:
# * one for no smoker group
# * two for smoker group
# 
# We have to understand better the relation that are inside the smoker group to identify clearly the 2 different charge distribution but we can assert that smoker have a strong impact on the charges as we can see in the plot on the right

# ### Feature : Bmi

# #### Sex and bmi

# In[18]:


bmi_sex = pd.crosstab(df.sex, df.bmi_classification,normalize='index')
sex_bmi = pd.crosstab(df.bmi_classification, df.sex, normalize='index')

fig, ax = plt.subplots(2,2, figsize = (21,10))

bmi_sex.plot.bar(y=['Underweight','Healthy','Overweight','Obese'], stacked=True, ax = ax[0,0])
ax[0,0].set_title('Population for bmi class')

sex_bmi.plot.bar(stacked=True, ax = ax[0,1])
ax[0,1].set_title('XXXX')

df[male].boxplot(column = 'charges', by = 'bmi_classification', ax = ax[1,0])
ax[1,0].set_xticklabels(['Obese', 'Overweight', 'Healthy', 'Underweight'])
ax[1,0].set_title('Charge distribution for male')

df[~male].boxplot(column = 'charges', by = 'bmi_classification', ax = ax[1,1])
ax[1,1].set_xticklabels(['Obese', 'Overweight', 'Healthy', 'Underweight'])
ax[1,1].set_title('Charge distribution for female')

plt.margins()
plt.show() 


# From this plots we can see that male and female are equally distribuited on the bmi classification. From the boxplots we don´t find any relationship between charges distribution on bmi classification for male and female.
# We can conclude that the sex attributes has no particular correlation with charges.
# 
# Now we make the same analysys on the group smoker and no smoker

# #### Bmi and smoker

# In[19]:


tab = pd.crosstab(df.bmi_classification,df.smoker,normalize = 'index')

fig, ax = plt.subplots(1,3, figsize = (23,4))

tab.plot.bar(stacked = True, ax = ax[0])
ax[0].set_title('Type of contractors for bmi class')

df[~smoker].boxplot(column = 'charges', by='bmi_classification', ax = ax[1])
ax[1].set_title('Charge distribution for no smoker')
ax[1].set(ylim=(1000 ,65000))

df[smoker].boxplot(column = 'charges', by='bmi_classification', ax = ax[2])
ax[2].set_title('Charge distribution for smoker')
ax[2].set(ylim=(1000 ,65000))

plt.show()      


# From this plots we see clearly that for the no smoker group there is no different charges based on the bmi classification.
# Contrarily we can see for the smoker group that we have a correlation with the bmi classification and that the combination smoker plu obese cause the hiher charges.
# 
# Important to see is the difference in cost distribution between smokers and non-smokers in the Obese group

# #### Relation between bmi and charges dividing the population by the smoker feature

# In[20]:


fig, ax=plt.subplots(3,1, sharey = True, figsize = (8,14))

ax[0].scatter(x=df[~smoker].bmi, y=df[~smoker].charges, label='no smoker', alpha = 0.5)
ax[0].scatter(x=df[smoker].bmi, y=df[smoker].charges, label='smoker', alpha = 0.5)

ax[0].legend()
ax[0].set_title('Relationship between bmi and charges')

for i in df['bmi_classification'].unique():
    t1 = df[~smoker][df['bmi_classification'] == i]
    t2 = df[smoker][df['bmi_classification'] == i]
    ax[1].scatter(t1.bmi, t1.charges, label = i, alpha = 0.5)
    ax[2].scatter(t2.bmi, t2.charges, label = i, alpha = 0.5)
    ax[1].set_title('Relationship between bmi and charges for no smoker')
    ax[2].set_title('Relationship between bmi and charges for smoker')
    ax[1].legend()
    ax[2].legend()
plt.show()


# In this plots we show the ralation between bmi and charges for two different groups, no smoker and smoker.
# Is clear that for no smoker we have no relation with the bmi contrary as for the smoker group. For the smoker group we see that for a bmi >= 30 we have a different charge distribution.

# ### Correlation map

# Now we can tranform the categorical data into numerical and make a correlation map to have evidence of our analysis.

# df_corr = df.drop(columns = ['bmi_classification'])
# df_corr = pd.get_dummies(df_corr)
# 
# df_corr = df_corr.corr()
# 
# fig, ax = plt.subplots(figsize=(15,10))
# sn.heatmap(df_corr, annot = True, vmin=0, vmax=1, linewidth = 0.1)
# 
# plt.show()

# As expected we have a strong positive correlation between charges and smoker.
# Other 2 important correlation are between:
# * charges and age
# * charges and bmi

# ## Hipothesis testing

# In this part we will make 3 different hypothesis on the data and we will test it with statistical considerations

# ### Hypothesis 1
# 
# * H0 = charges are normal distributed
# * H1 = charges are not normal distributed

# In[21]:


from scipy.stats import normaltest

normaltest(df.charges, axis=0, nan_policy='propagate')


# We have a pvalue < 0.5 and for this reason we have to reject the null hipothesys and say that this values are not normally distribuited.
# We can see here the skew and the kurtosis values.

# In[22]:


df.charges.skew() , df.charges.kurt()


# In[23]:


df.charges.plot(kind='density')
plt.show()


# ### Hypothesis 2
# 
# * H0 = charges for smoker and no smoker has the same distribution
# * H1 = charges for smoker and no smoker has different distribution

# In[24]:


from scipy.stats import mannwhitneyu

mannwhitneyu_result = mannwhitneyu(df[smoker].charges, df[~smoker].charges)
print(mannwhitneyu_result)


# In[25]:


df[smoker].charges.plot(kind='density', label = 'smoker')
df[~smoker].charges.plot(kind='density', label = 'no smoker')
plt.legend()
plt.show()


# ### Hypothesis 2
# 
# * H0 = Smoker with bmi>29.9 has the same distribution than smoker with bmi<=29.9
# * H1 = Smoker with bmi>29.9 has different distribution than smoker with bmi<=29.9

# In[26]:


from scipy.stats import mannwhitneyu

p1 = df[(df.smoker=='yes')& (df.bmi<=29.9)].charges
p2 = df[(df.smoker=='yes')& (df.bmi>29.9)].charges

mannwhitneyu(p1,p2)


# In[27]:


p1 = df[(df.smoker=='yes')& (df.bmi<=29.9)].charges
p2 = df[(df.smoker=='yes')& (df.bmi>29.9)].charges

mannwhitneyu_result = mannwhitneyu(p1.values,p2.values)
print(mannwhitneyu_result)

p1.plot(kind='density', label = 'smoker with bmi<=29.9')
p2.plot(kind='density', label = 'smoker with bmi>29.9')
plt.legend()
plt.show()

