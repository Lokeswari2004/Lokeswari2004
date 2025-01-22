import numpy as np  # library used for working with arrays
import pandas as pd # library used for data manipulation and analysis

import seaborn as sns # library for visualization
import matplotlib.pyplot as plt # library for visualization
%matplotlib inline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv") 
df.head()

# 1. Let's find the number of columns

# finds the number of columns in the dataset
total_cols=len(df.axes[1])
print("Number of Columns: "+str(total_cols))

# 2. Let's show the first 10 records of the dataset.

df.head(10)

# 3. Finding the number of rows in the dataset

# finds the number of rows in the dataset
total_rows=len(df.axes[0])
print("Number of Rows: "+str(total_rows))

# 4.Now let us understand the dimensions of the dataset.

print('The dimension of the DataFrame is: ', df.ndim)

* Note: The Pandas dataframe.ndim property returns the dimension of a series or a DataFrame.

For all kinds of dataframes and series, it will return dimension 1 for series that only consists of rows and will return 2 in case of DataFrame or two-dimensional data.

# 5.The size of the dataset.

df.size

Note: In Python Pandas, the dataframe.size property is used to display the size of Pandas DataFrame.

* It returns the size of the DataFrame or a series which is equivalent to the total number of elements.


* If I want to calculate the size of the series, it will return the number of rows. In the case of a DataFrame, it will return the rows multiplied by the columns.

# 6. Let us now find out the **data types **of all variables in the dataset.

#The info() function is used to print a concise summary of a DataFrame. 
#This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

df.info()

* There are 768 entries

* There are 2 float data types and 67 integer data types

# 7.Now let us check for missing values.

#functions that return a boolean value indicating whether the passed in argument value is in fact missing data.
# this is an example of chaining methods 

df.isnull().values.any()

Pandas defines what most developers would know as null values as missing or missing data in pandas. Within pandas, a missing value is denoted by NaN.

#it can also output if there is any missing values each of the columns

df.isnull().any()

df.shape

# 8. First ten records

df.head(10)

# 9.Last ten records

df.tail(10)

df.describe()

# 10.Checking if there are any null values having records

df[df.isnull().any(axis=1)]

# 11.Check for the Null Values

df.isnull().sum()

* There are no null values in our dataset. Let us move on to visualizing the dataset to gather more insights about the data.

# 12.check for the duplicate data and drop them


df[df.duplicated()]

# observation:-

* there is no duplicated valuse in my dataset



# Data visualization

* we will start visualizing the features of the dataset one by one. Firstly, Univariate feature visualization will be done, then we will move onto Multivariate feature visualization.

# 1.Analysis of Pregnancies

* As observed, Pregnancies is a Quantitative feature. There are many plots to analyse these type of data. Histograms, Box plots and Violin plots, are useful to know how the data is distributed.

fig1, ax1 = plt.subplots(1, 2, figsize=(20, 7))
fig2, ax2 = plt.subplots(figsize=(20, 7))

sns.histplot(data=df, x="Pregnancies", kde=True, ax=ax1[0],color = 'magenta')
sns.boxplot(data=df, x="Pregnancies", ax=ax1[1],color = 'aqua')

sns.violinplot(data=df, x="Pregnancies", ax=ax2,color = 'lime')

plt.show()

print("Median of Pregnancies: ", df["Pregnancies"].median())
print("Maximum of Pregnancies: ", df["Pregnancies"].max())

df["Pregnancies"].value_counts()

# From the above analysis we observe that:

  *  Most patients had 0, 1 or 2 pregnancies.

  *  Median value of Pregnancies is 3.

  *  Also, patients had upto 17 pregnancies!

There are 3 outliers on the boxplot.

# 2.Analysis of Outcome (Target Variable)

 
* A Count plot and a Pie chart will be two useful plots to analyse the Outcome column as it is a categorical feature. Usefulness in the sense, both the plots will allow us to observe the distribution of each category in the feature.

fig, ax = plt.subplots(1, 2, figsize=(20, 7))

sns.countplot(data=df, x="Outcome", ax=ax[0])
df["Outcome"].value_counts().plot.pie(explode=[0.1, 0],autopct="%1.1f%%", labels=["No", "Yes"], shadow=True, ax=ax[1])

plt.show()

We observe from above Plot

  *  65.1% patients in the dataset do NOT have diabetes.
  *  34.9% patients in the dataset has diabetes.


# 3.Analysis of Glucose

* Glucose is a Quantitative feature. Histograms, Box plots and Violin plots, are useful to know how the data is distributed.

fig3, ax3 = plt.subplots(1, 2, figsize=(20, 7))
fig4, ax4 = plt.subplots(figsize=(20, 7))

sns.histplot(data=df, x="Glucose", kde=True, ax=ax3[0], color = 'magenta')
sns.boxplot(data=df, x="Glucose", ax=ax3[1], color = 'aqua')

sns.violinplot(data=df, x="Glucose", ax=ax4, color = 'lime')

plt.show()

print("Median of Glucose: ", df["Glucose"].median())
print("Maximum of Glucose: ", df["Glucose"].max())
print("Mean of Glucose: ", df["Glucose"].mean())

print("Rows with Glucose value of 0: ", df[df["Glucose"] == 0].shape[0])

We observe that:

 *   Median (117.0) and mean (120.8) of Glucose lie very close to each other i.e. the distribution is more or less symmetric and uniform.

 *   As seen from the box plot, an outlier lies on 0-value, which I talked about earlier.

 *   There are 5 rows with Glucose value as 0. This is not logical, so we need to keep this in mind.


# 4.Analysis of Blood Pressure

* BloodPressure is a Quantitative feature. Histograms, Box plots and Violin plots, are useful to know how the data is distributed.

fig5, ax5 = plt.subplots(1, 2, figsize=(20, 7))
fig6, ax6 = plt.subplots(figsize=(20, 7))

sns.histplot(data=df, x="BloodPressure",kde=True,  ax=ax5[0],color = 'magenta')
sns.boxplot(data=df, x="BloodPressure", ax=ax5[1], color = 'aqua')

sns.violinplot(data=df, x="BloodPressure", ax=ax6, color = 'lime')

plt.show()

print("Median of Blood Pressure: ", df["BloodPressure"].median())
print("Maximum of Blood Pressure: ", df["BloodPressure"].max())
print("Mean of Pressure: ", df["BloodPressure"].mean())

print("Rows with BloodPressure value of 0: ", df[df["BloodPressure"] == 0].shape[0])

We observe that:

 *   Median (72.0) and mean (69.1) of BloodPressure lie very close to each other i.e. the distribution is more or less symmetric and uniform.

  *  As seen from the box plot and violin plot, some outliers lie on 0-value, which I talked about earlier.

 *   There are 35 rows with BloodPressure value as 0. This is not logical

# 5.Analysis of Insulin

Plotting Histogram, Box plot and Violin plot for Insulin.

fig7, ax7 = plt.subplots(1, 2, figsize=(20, 7))
fig8, ax8 = plt.subplots(figsize=(20, 7))

sns.histplot(data=df, x="Insulin", kde=True, ax=ax7[0],color = 'magenta')
sns.boxplot(data=df, x="Insulin", ax=ax7[1],color = 'aqua')

sns.violinplot(data=df, x="Insulin", ax=ax8,color = 'lime')

plt.show()

print("Rows with Insulin value of 0: ", df[df["Insulin"] == 0].shape[0])

* The plots for Insulin are highly skewed. Also, the 0-value logical error is the most for this feature. 374 out of 768 instances have value of Insulin as 0.

# 6.Analysis of BMI

Plotting Histogram, Box plot and Violin plot for BMI.

fig9, ax9 = plt.subplots(1, 2, figsize=(20, 7))
fig10, ax10 = plt.subplots(figsize=(20, 7))

sns.histplot(data=df, x="BMI", kde=True, ax=ax9[0], color = 'magenta')
sns.boxplot(data=df, x="BMI", ax=ax9[1], color = 'aqua')

sns.violinplot(data=df, x="BMI", ax=ax10 , color = 'lime')

plt.show()

print("Median of BMI: ", df["BMI"].median())
print("Maximum of BMI: ", df["BMI"].max())
print("Mean of BMI: ", df["BMI"].mean())

print("Rows with BMI value of 0: ", df[df["BMI"] == 0].shape[0])

# We observe that:

 *   Median (32.0) and Mean (31.9) of BMI are very close to each other. Thus, the distribution is more or less symmetric and uniform

 *   Maximum BMI is 67.1

  *  There are 11 rows with BMI value as 0


# 7.Analysis of Diabetes Pedigree Function

* DiabetesPedigreeFunction is a continuous and quantitative variable.

fig11, ax11 = plt.subplots(1, 2, figsize=(20, 7))
fig12, ax12 = plt.subplots(figsize=(20, 7))

sns.histplot(data=df, x="DiabetesPedigreeFunction", kde=True, ax=ax11[0],color = 'magenta')
sns.boxplot(data=df, x="DiabetesPedigreeFunction", ax=ax11[1],color = 'aqua')

sns.violinplot(data=df, x="DiabetesPedigreeFunction", ax=ax12,color = 'lime')

plt.show()

print("Median of DiabetesPedigreeFunction: ", df["DiabetesPedigreeFunction"].median())
print("Maximum of DiabetesPedigreeFunction: ", df["DiabetesPedigreeFunction"].max())
print("Mean of DiabetesPedigreeFunction: ", df["DiabetesPedigreeFunction"].mean())

# We observe that:

 *   The histogram is higly skewed on the left side.

 *   There are many outliers in the Box plot.

 *   Violin plot distribution is dense in the interval 0.0 - 1.0.


# 8.Analysis of Age

Plotting Histogram, Box plot and Violin plots for Age.

fig13, ax13 = plt.subplots(1, 2, figsize=(20, 7))
fig14, ax14 = plt.subplots(figsize=(20, 7))

sns.histplot(data=df, x="Age", kde=True, ax=ax13[0],color = 'magenta')
sns.boxplot(data=df, x="Age", ax=ax13[1],color = 'aqua')

sns.violinplot(data=df, x="Age", ax=ax14,color = 'lime')

plt.show()

print("Median of Age: ", df["Age"].median())
print("Maximum of Age: ", df["Age"].max())
print("Mean of Age: ", df["Age"].mean())

# 9. Exploratory Data Analysis 
* In this section, we will be doing some basic Exploratory Data Analysis to get the "feel" of the data, we will be checking the distributions, the correlations etc of the different columns and try to remove the null values present.

fig, axes = plt.subplots(4, 2, figsize=(15, 10))
axes = axes.flatten()
ax_idx = 0
columns = df.drop('Outcome', axis = 1).columns
for col in columns:
    df[col].plot(kind = 'hist', ax = axes[ax_idx], title = col)
    ax_idx += 1

plt.suptitle('Sales Trend according to Departments')
plt.tight_layout()
plt.show()

# 10 it then saves this into a new dataframe df1

df1 = df[(df['BloodPressure']==df['BloodPressure'].median()) & (df['BMI']<df['BMI'].median())]
number_of_women=len(df.axes[0])
print("Number of women:" +str(number_of_women))

# 11.Getting a pairwise distribution between Glucose, Skin thickness and Diabetes pedigree function.

* The pair plot gives a pairwise distribution of variables in the dataset. pairplot() function creates a matrix such that each grid shows the relationship between a pair of variables. On the diagonal axes, a plot shows the univariate distribution of each variable.

sns.pairplot(data=df,vars=['Glucose', 'SkinThickness', 'DiabetesPedigreeFunction'], hue = 'Outcome')
plt.show()

# 12. Studying the correlation between glucose and insulin using a Scatter Plot.

* A scatter plot is a set of points plotted on horizontal and vertical axes. The scatter plot can be used to study the correlation between the two variables. One can also detect the extreme data points using a scatter plot.

sns.scatterplot(x='Glucose',y='Insulin',data=df)
plt.show()

# 13.Let us explore the possibility of outliers using the Box Plot.

* Boxplot is a way to visualize the five-number summary of the variable. Boxplot gives information about the outliers in the data.

plt.boxplot(df['Age'])

plt.title('Boxplot of Age')
plt.ylabel('Age')
plt.show()

* The box plot shows the presence of outliers above the horizontal line.

# 14.Understanding the number of women in different age groups with diabetes.

plt.hist(df[df['Outcome']==1]['Age'], bins = 5)
plt.title('Distribution of Age for Women who has Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

* Of all the women with diabetes most are from the age between 22 to 30.

* The frequency of women with diabetes decreases as age increases.

# 15.understanding the number of women in different age groups without diabetes.

plt.hist(df[df['Outcome']==0]['Age'], bins = 5)
plt.title('Distribution of Age for Women who do not have Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

* The highest number of Women without diabetes range between ages 22 to 33.

* Women between the age of 22 to 35 are at the highest risk of diabetes and also the is the highest number of those without diabetes.

# 16. And finally let us find and visualize the correlation between all variables.

corr_matrix = df.iloc[:,0:8].corr()

corr_matrix

# 17. Fill missing Values

* We will replace the 0-values of the following columns with their median values:

    * Glucose

   *  BloodPressure

   *  SkinThickness

   *  Insulin

   *  BMI


# first copy the original dataframe into a new pandas dataframe
df_cp = df.copy()
# now fill the missing values with the median value of the particular columns
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

df.describe().T

# separating the data and labels
X = df.drop(columns = 'Outcome', axis=1)
Y = df['Outcome']

print(X)
print(Y)

# 18.Analyzing the Data

#Scatter plot
sns.scatterplot(data=df,x="Insulin",y="Glucose")
plt.show()

* Age and pregnancies are positively correlated.
* Glucose and insulin are positively correlated.
* SkinThickness and BMI are positively correlated.

# Conclusion

* The glucose levels may not alone help us in diagnosing diabetes. If you continue to do this analysis on other features as well, the predictions can be more accurate. Not all plots work for all the data, we have to try and test to find which fits the best!


