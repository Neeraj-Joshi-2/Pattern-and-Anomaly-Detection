# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:25:06 2021

@author: NEERAJ
"""
 
#PANDAS

#%%
#Creating the dataframes

import pandas as pd

df = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
print(df)


df_2 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
print(df_2)

#Assigning index parameter

df_3 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],  'Sue': ['Pretty good.', 'Bland.']} , 
                    index=['Product A', 'Product B'])
print(df_3)

#%%

sr = pd.Series([1, 2, 3, 4, 5])
print(sr)

sr_2 = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
print(sr_2)

#%%
import pandas as pd
#Reading the file

wine_reviews = pd.read_csv("D:/DOCUMENTS/SEM/SEM 5/Pattern And Anomaly Detection Lab/winemag-data.csv")
#print(wine_reviews)
print(wine_reviews.shape)
print(wine_reviews.head())

#Removing unwanted index column

wine_reviews = pd.read_csv("D:/DOCUMENTS/SEM/SEM 5/Pattern And Anomaly Detection Lab/winemag-data.csv" ,
                           index_col = 0)

print(wine_reviews.head())

#%%

#Printing data of country column
print(wine_reviews.country)

#Indexing
print(wine_reviews['country'])
print(wine_reviews['country'][0])

#%%
#Indexing using iloc accessor operator
#To select the first row of data in a DataFrame

print(wine_reviews.iloc[0])

#getting a column with iloc
print(wine_reviews.iloc[ : , 0])
print(wine_reviews.iloc[: 3 , 0]) 
print(wine_reviews.iloc[1:3, 0])
print(wine_reviews.iloc[[0, 1, 2], 0])

#negative indexing
print(wine_reviews.iloc[-5:])


#%%

#Label-based selection

#print(wine_reviews.loc[0, 'country'])

print(wine_reviews.loc[: , ['country', 'points', 'price']])
 
#%%

# Manipulating the index
print(wine_reviews.set_index("country"))

#Conditional selection
print(wine_reviews.country == 'Italy')

print(wine_reviews.loc[wine_reviews.country == 'Italy'])

print(wine_reviews.loc[(wine_reviews.country == 'Italy') & (wine_reviews.points >= 90)])

print(wine_reviews.loc[(wine_reviews.country == 'Italy') | (wine_reviews.points >= 90)])

# usinf isin

print(wine_reviews.loc[wine_reviews.country.isin(['US', 'Spain'])])

#using notnull

print(wine_reviews.loc[wine_reviews.price.notnull()])

#%%

#print(wine_reviews.loc[(wine_reviews.country).str.len() > 5])

#%%
#Assigning data

wine_reviews['critic'] = 'everyone'
print(wine_reviews['critic'])


wine_reviews['index_backwards'] = range(len(wine_reviews), 0, -1)
print(wine_reviews['index_backwards'])



#%%
#Summary Functions

print(wine_reviews.price.describe())
print(wine_reviews.country.describe())

print(wine_reviews.price.mean())
print(wine_reviews.country.unique())

#unique values with counts
print(wine_reviews.country.value_counts())

#%%
#maps()
review_price_mean = wine_reviews.price.mean()
wine_reviews.price.map(lambda p : p - review_price_mean)

#apply()
def remean_points(row):
    row.price = row.price - review_price_mean
    return row

wine_reviews.apply(remean_points, axis='columns')

#map() and apply() return new, transformed Series and DataFrames
print(wine_reviews.head(1))

#%%
#Alternative method for above operation
review_price_mean = wine_reviews.price.mean()
wine_reviews.price - review_price_mean

#combining information
print(wine_reviews.country + " - " + wine_reviews.winery)

#%%
#grouping data by price

print(wine_reviews.groupby('price').price.count())
print(wine_reviews.groupby('price').points.min())

print(wine_reviews.groupby('winery').apply(lambda df: df.country.iloc[0]))
#pick out the best wine by country and province
print(wine_reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()]))

#generate a simple statistical summary of the dataset
print(wine_reviews.groupby(['country']).price.agg([len, min, max]))

#%%
#Multi-indexes
countries_reviewed = wine_reviews.groupby(['country', 'province']).description.agg([len])
print(countries_reviewed)

mi = countries_reviewed.index
print(type(mi))

#converting back to a regular index
print(countries_reviewed.reset_index())

#%%
#SORTING
countries_reviewed = countries_reviewed.reset_index()
print(countries_reviewed.sort_values(by='len'))

#sorting in descending order
print(countries_reviewed.sort_values(by='len', ascending=False))

#To sort by index values, use the companion method sort_index()
print(countries_reviewed.sort_index())
#sort by more than one column at a time
print(countries_reviewed.sort_values(by=['country', 'len']))

#%%
#datatype of price column
print(wine_reviews.price.dtype)
#datatype of all columns present in df
print(wine_reviews.dtypes)
# transform the points column from  int64 data type into a float64 data type
print(wine_reviews.points.astype('float64'))
#A DataFrame or Series index datatype
print(wine_reviews.index.dtype)

#%%
#Missing data

#methods specific to missing data
print(wine_reviews[pd.isnull(wine_reviews.country)])
#Replacing missing values is a common operation
print(wine_reviews.region_2.fillna("Unknown"))
#replacing the data
print(wine_reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino"))

#%%
#renaming the column
print(wine_reviews.rename(columns={'points': 'score'}))
#using python dictionary for renaming index
print(wine_reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'}))

#row index and the column index have their own name attribute
print(wine_reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns'))

#%%
#Combining

#combining two dataframes using concact
df_1 = pd.DataFrame({'Bob': ['I liked it.'],   'Sue': ['Pretty good.']})
df_2 = pd.DataFrame({'Bob': ['It was awful.'], 'Sue': ['Bland.']})

print(pd.concat([df_1 , df_2]))

#%%
#join() combine different DataFrame objects which have an index in common
left = df_1.set_index(['Sue'])
right = df_2.set_index(['Sue'])
print(left.join(right , lsuffix='_df1', rsuffix='_df2'))


#%%

#NUMPY

import numpy as np
#creating array
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

#Use a tuple to create a NumPy array
arr = np.array((1, 2, 3, 4, 5))
print(arr)
#%%
#0d array
arr1 = np.array(42)
print(arr1)
#1d array
arr2 = np.array([1, 2, 3, 4, 5])
print(arr2)
#2d array
arr3 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr3)
#3d array
arr4 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr4)

#dimensions of each array
print(arr1.ndim)
print(arr2.ndim)
print(arr3.ndim)
print(arr4.ndim)

#%%
#Higher Dimensional Arrays
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim)

#%%
#array indexing 
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('Last element from 2nd dim: ', arr[1, -1])

#Array Slicing
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5])
print(arr[4:])
print(arr[:4])

#Negative slicing
print(arr[-3:-1])
#step value to determine the step of the slicing
print(arr[1:5:2])
print(arr[::2])

#%%
#slicing 2d array
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
#From the second element, slice elements from index 1 to index 4 
print(arr[1, 1:4])
#From both elements, return index 2
print(arr[0:2, 2])
#From both elements, slice index 1 to index 4 
print(arr[0:2, 1:4])

#%%
#Numpy Datatypes
#data type of an array object
arr = np.array([1, 2, 3, 4])
print(arr.dtype)

#data type of an array containing strings:
arr = np.array(['apple', 'banana', 'cherry'])
print(arr.dtype)

#%%
#Creating Arrays With a Defined Data Type , string in this case
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
print(arr.dtype)

#Create an array with data type 4 bytes integer
arr = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(arr.dtype)

#%%
#astype() function creates a copy of the array
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print(newarr)
print(newarr.dtype)

#Change data type from integer to boolean
arr = np.array([1, 0, 3])
newarr = arr.astype(bool)
print(newarr)
print(newarr.dtype)

#%%
#Make a copy, change the original array, and display both arrays
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)

#Make a view, change the original array, and display both arrays
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)

#%%
#NumPy array has the attribute base that returns None if the array owns the data
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
y = arr.view()

print(x.base)
print(y.base)

#%%
#Shape of an Array
#The shape of an array is the number of elements in each dimension
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('shape of array :', arr.shape)

#%%
#Numpy array shaping 
# 1-D array with 12 elements into a 2-D array
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)

#1-D array with 12 elements into a 3-D array.
#The outermost dimension will have 2 arrays that contains 3 arrays, each with 2 elements
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(2, 3, 2)
print(newarr)

# returns the original array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(arr.reshape(2, 4).base)

#%%
#unknown dimension
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
newarr = arr.reshape(2, 2, -1)
print(newarr)

#Flattening the arrays
#Convert the array into a 1D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)
print(newarr)

#%%
#NumPy Array Iterating
arr = np.array([1, 2, 3])
for x in arr:
    print(x)

#Iterate on the elements of the following 2-D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
    print(x)
#Iterate on each scalar element of the 2-D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
  for y in x:
    print(y)

#%%
#iterating 3d array  
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
    print(x)
    
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
  for y in x:
    for z in y:
      print(z)
      
#%%
#Iterating Arrays Using nditer()
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
for x in np.nditer(arr):
  print(x)

#Iterate through the array as a string
arr = np.array([1, 2, 3])
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)

#%%
#Iterating With Different Step Size
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for x in np.nditer(arr[: , ::2]):
    print(x)

#%%
#Enumerated Iteration Using ndenumerate()
arr = np.array([1, 2, 3])
for idx, x in np.ndenumerate(arr):
  print(idx, x)

#for 2d array
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for idx, x in np.ndenumerate(arr):
  print(idx, x)

#%%
#NumPy Joining Array
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)

#Join two 2-D arrays along rows (axis=1)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
print(arr)

#%%
#Joining Arrays Using Stack Functions
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1)
print(arr)

#Stacking Along Rows
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.hstack((arr1, arr2))
print(arr)

#%%
#Stacking Along Columns
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.vstack((arr1, arr2))
print(arr)

#Stacking Along Height (depth)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.dstack((arr1, arr2))
print(arr)

#%%
#NumPy Splitting Array
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr)
#Split the array in 4 parts
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 4)
print(newarr)

#%%
#Split Into Arrays
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr)
print(newarr[0])
print(newarr[1])
print(newarr[2])

#Splitting 2-D Arrays
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)
print(newarr)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3)
print(newarr)

#Split the 2-D array into three 2-D arrays along rows.
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
print(newarr)

#%%
#using hsplit
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.hsplit(arr, 3)
print(newarr)

#%%
#NumPy Searching Arrays
#Find the indexes where the value is 4
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)

#Find the indexes where the values are even
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr%2 == 0)
print(x)

#Find the indexes where the values are odd:
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr%2 == 1)
print(x)

#%%
#Find the indexes where the value 5 should be inserted
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 5)
print(x)
#Find the indexes where the value 7 should be inserted, starting from the right
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7, side='right')
print(x)
#Find the indexes where the values 2, 4, and 6 should be inserted
arr = np.array([1, 3, 5, 7])
x = np.searchsorted(arr, [2, 4, 6])
print(x)

#%%
#sorting arrays
arr = np.array([3, 2, 0, 1])
print(np.sort(arr))
#Sort the array alphabetically
arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))
#Sort a boolean array
arr = np.array([True, False, True])
print(np.sort(arr))

#%%
#Sorting a 2-D Array
arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr))

#%%
#NumPy Filter Array
#Create an array from the elements on index 0 and 2:
arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print(newarr)

#%%
#Create a filter array that will return only values higher than 42
arr = np.array([41, 42, 43, 44])
# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

#%%
#Create a filter array that will return only even elements from the original array
arr = np.array([1, 2, 3, 4, 5, 6, 7])
# Create an empty list
filter_arr = []
# go through each element in arr
for element in arr:
  # if the element is completely divisble by 2, set the value to True, otherwise False
  if element % 2 == 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

#%%
#Creating Filter Directly From Array
#Create a filter array that will return only values higher than 42:

arr = np.array([41, 42, 43, 44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

#%%
#Create a filter array that will return only even elements from the original array
arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)





















