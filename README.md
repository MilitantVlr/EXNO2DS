# EXNO2DS
# AIM:
To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT

## NAME:N.NAVYA SREE
## REG.NO: 212223040138
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
df=pd.read_csv("titanic_dataset.csv")
df
```
![Screenshot 2024-09-10 161359](https://github.com/user-attachments/assets/a19ac75f-2598-4a78-8028-f51d3d167448)
```
df.info()
```
![Screenshot 2024-09-10 161542](https://github.com/user-attachments/assets/a4cff9be-b822-44d9-b860-b93958c6f46d)
```
df.shape
```
![Screenshot 2024-09-10 161630](https://github.com/user-attachments/assets/1d5d6409-28ad-4170-8c4e-9ca136fbf975)
```
df.set_index("PassengerId",inplace=True)
df.describe()
```
![Screenshot 2024-09-10 161715](https://github.com/user-attachments/assets/7517f446-2232-4354-bacf-3588627c9681)
```
df.shape
```
![Screenshot 2024-09-10 161804](https://github.com/user-attachments/assets/099f3b28-ff0e-4562-bfa6-29ee683d390a)
## Categorical data analysis
```
df.nunique()
```
![Screenshot 2024-09-10 161843](https://github.com/user-attachments/assets/9161a28b-ab1e-44fe-8e83-7d86420102ae)
```
df["Survived"].value_counts()
```
![Screenshot 2024-09-10 161933](https://github.com/user-attachments/assets/079c8360-d6f2-4aae-8b90-fd4f5117d925)
```
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
![Screenshot 2024-09-10 162010](https://github.com/user-attachments/assets/4465d3e3-aa63-4aec-9811-5cdae6578329)
```
sns.countplot(data=df,x="Survived")
```
![Screenshot 2024-09-10 162047](https://github.com/user-attachments/assets/c0cf2836-37f2-4375-8c23-4683bb745434)
```
df
```
![Screenshot 2024-09-10 193049](https://github.com/user-attachments/assets/21973863-d4ef-4d5f-8b3e-b9a8bf6d6ac2)
```
df.Pclass.unique()
```
![Screenshot 2024-09-10 193155](https://github.com/user-attachments/assets/f885d8c6-8853-4b1a-8e78-7f6067cb479c)
```
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
![Screenshot 2024-09-10 193247](https://github.com/user-attachments/assets/3d3614e7-1832-4802-8fca-d70a86df9369)
## Bivariate Analysis
```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
![Screenshot 2024-09-10 193339](https://github.com/user-attachments/assets/c8b44162-3a88-431b-b21f-29b467e3ecbc)
```
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
![Screenshot 2024-09-10 193448](https://github.com/user-attachments/assets/3abf2cac-ad2d-416d-8fa4-b6323a2b08fd)
```
df.boxplot(column="Age",by="Survived")
```
![Screenshot 2024-09-10 193540](https://github.com/user-attachments/assets/42095543-8ea8-44cd-bb87-00896d5d65ae)
```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
![Screenshot 2024-09-10 193633](https://github.com/user-attachments/assets/be003743-cb0f-40df-b419-167c37da8b0e)
```
sns.jointplot(x="Age",y="Fare",data=df)
```
![Screenshot 2024-09-10 193736](https://github.com/user-attachments/assets/9bc1238d-accf-4719-b6bc-f0d22508491e)
## Multivariate Analysis
```
fig, ax1 = plt.subplots(figsize=(8,5))
plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```
![image](https://github.com/user-attachments/assets/e8ae7471-ddf7-43f6-a837-e2fe2a51f3f0)
```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")

```
![Screenshot 2024-09-10 194235](https://github.com/user-attachments/assets/3342116c-1702-434a-aadd-994dfc55974b)
## Co-relation

```
corr=df.corr()
sns.heatmap(corr,annot=True)
```
![Screenshot 2024-09-10 194336](https://github.com/user-attachments/assets/c2a9c49e-efb9-4232-900a-936c347ab084)
```
sns.pairplot(df)
```
![Screenshot 2024-09-10 194444](https://github.com/user-attachments/assets/c8a2e266-a4d2-4929-b52b-bf8109d5edd4)









# RESULT

Exploratory Data Analysis on the given data set successfully.
