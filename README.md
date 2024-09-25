# EXPERIMENT 2: EXPLORATORY DATA ANALYSIS

## AIM:
  To perform Exploratory Data Analysis on the given data set.
      
## EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
## ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT:
### PROGRAMS:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("titanic_dataset.csv")

df
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image.png>)

## INFORMATION ABOUT THE DATASET:
```python
df.info()
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image2.png>)

## DISPLAY NO OF ROWS AND COLUMNS:
```python
df.shape
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image3.png>)

## DROP THE NULL VALUES:
```PYTHON
df.dropna(inplace=True)
df
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image4.png>)

## AFTER DROPPPING THE NULL VALUES(GIVE THE NUMBER OF ROWS AND COLUMNS):
```PYTHON
df.shape
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image5.png>)

## SET PASSENGER ID AS INDEX COLUMN:
```PYTHON
df.set_index("PassengerId",inplace=True)
df
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image6.png>)

## STASTISTICS OF THE DATASET:
```PYTHON
df.describe()
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image7.png>)

## CATEGORICAL DATA ANALYSIS:
## USE VALUE COUNT FUNCTION AND PERFROM CATEGORICAL ANALYSIS:
```python
df.nunique()
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image8.png>)

## IT'S COUNTS THE NUMBERS OF UNSURVIVED AND SURVIVED:
```PYTHON
df["Survived"].value_counts()
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image9.png>)

## PERCENTAGE OF UNSURVIVED AND SURVIVED:
```PYTHON
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
### OUTPUT:
![alt text](<SCREENSHOT IMAGES/image10.png>)


## IT'S COUNTS THE NUMBERS OF MALE AND FEMALE IN SEX COLUMN:
```PYTHON
df['Sex'].value_counts()
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image11.png>)

## PERCENTAGE OF MALE AND FEMALE:
```PYTHON
per1=(df['Sex'].value_counts()/df.shape[0]*100).round(2)
per1
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image12.png>)

## COUNTS THE Pclass OF (1,2,3):
```PYTHON
df["Pclass"].value_counts()
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image13.png>)

## PERCENTAGE OF Pclass:
```PYTHON
per2=(df["Pclass"].value_counts()/df.shape[0]*100).round(2)
per2
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image14.png>)

## UNIVARIATE ANALYSIS:
## USE COUNTPLOT AND PERFORM UNIVARIATE ANALYSIS FOR THE "SURVIVED" COLUMN IN TITANIC DATASET:
```PYTHON
sns.countplot(data=df,x="Survived")
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image15.png>)

## HISTOGRAM FOR DISTRIBUTION:
```PYTHON
sns.histplot(df['Survived'], kde=True)
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image16.png>)

## KDE PLOT FOR SMOOTH DISTRIBUTION:
```PYTHON
sns.kdeplot(df['Survived'], shade=True)
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image17.png>)

## COUNT PLOT TO SEE THE FREQUENCY OF CATEGORIES:
```PYTHON
sns.countplot(x='Embarked', data=df)
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image18.png>)

## IDENTIFY UNIQUE VALUES IN "PASSENGER CLASS" COLUMN:
```PYTHON
df.Pclass.unique()
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image19.png>)

## RENAMING COLUMN:
```PYTHON
df.rename(columns = {'Sex':'Gender'}, inplace = True)
df
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image20.png>)

## BIVARIATE ANALYSIS:
## USE CATPLOT METHOD FOR BIVARIATE ANALYSIS:
```PYTHON
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image21.png>)


## KDE PLOT (JOINT DISTRIBUTION):
```PYTHON
sns.kdeplot(x='Survived', y='Age', data=df, cmap="Blues", shade=True)
plt.title('Bivariate KDE Plot')
plt.show()
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image23.png>)

## REGRESSION PLOT FOR LINEAR RELATIONSHIPS:
```PYTHON
sns.lmplot(x='Survived', y='Age', data=df)
plt.title('Regression Plot: Numerical Column 1 vs Numerical Column 2')
plt.show()
```

### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image24.png>)


## USE BOXPLOT METHOD TO ANALYZE AGE AND SURVIVED COLUMN:
```PYTHON
df.boxplot(column="Age",by="Survived")
```

### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image26.png>)

## USE SCATTERPLOT OF COLUMN AGE AND FARE:
```PYTHON
sns.scatterplot(x=df["Age"],y=df["Fare"])
```

### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image27.png>)

## USE JOINTPLOT COLUMN AGE AND FARE:
```PYTHON
sns.jointplot(x="Age",y="Fare",data=df)
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image28.png>)


## VIOLIN PLOT FOR COMBINED DISTRIBUTION AND RANGE:
```PYTHON
sns.violinplot(x='Gender', y='Survived', data=df)
plt.title('Violin Plot: Numerical vs Categorical')
plt.show()
```

### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image30.png>)

## MULTIVARIATE ANALYSIS:
## USE BOXPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,AGE,GENDER):
```PYTHON
fig, ax1=plt.subplots(figsize=(8,5))
plt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```
### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image31.png>)


## USE CATPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,SURVIVED,GENDER):
```PYTHON
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```

### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image32.png>)

## CO-RELATION:
```PYTHON
corr=df.corr()
sns.heatmap(corr,annot=True)
```

### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image33.png>)

## IMPLEMENT HEATMAP AND PAIRPLOT FOR THE DATASET:
```PYTHON
sns.pairplot(df)
```

### OUTPUT:

![alt text](<SCREENSHOT IMAGES/image34.png>)

## RESULT:
Hence performing Exploratory Data Analysis on the given data set is successfully executed.
