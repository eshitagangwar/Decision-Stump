
# coding: utf-8

# # Decision Stump
# 
# _Author: Nghia T. Le_  
# 
# **Total points = 100 points**
# 
# The decision stump learner is introduced in lecture video 12 of Module 1. In short, a decision stump is a one-node decision tree. The dataset is split on the values of a single provided attribute. The label at the leaf nodes is decided via the majority voting algorithm.
# 
# Below we show an example decision stump, the result of the dataset and the implementation that you will be working in this assignment. The attribute is "anaemia". The dataset is split on whether a person has anaemia (right) or not (left). Both the labels at the leaf nodes indicate "alive" prediction.
# 
# <img src="./img/stump_intro.png" width="400" height="400">
# 
# This assignment is designed to help you build a decision stump while also building your 
# familiarity with programming in Python. 
# **Remember to run your 
# code from each cell IN ORDER before submitting your assignment.** Running your code beforehand 
# will notify you of errors and give you a chance to fix your errors before submitting. 
# You should view your Vocareum submission as if you are delivering a final project to 
# your manager or client.
# 
# ### Learning Objectives 
# - Construct a decision stump with a split on one node and use the majority vote algorithm at leaf nodes
# - Become familiar with a machine learning algorithm and mindset
# - Gain further familiarity coding in Python and related modules (e.g. NumPy)

# <a name="index"></a>
# 
# # Table of Contents
# 
# #### [Part I: Data Inspection [20 pts]](#data)
# 
# - [Question 1 [5 pt]: Number of Attribute](#q1)
# - [Question 2 [5 pt]: Name of Attribute](#q2)
# - [Question 3 [10 pt]: Values of Attribute](#q3)
# - [Create Your Own  Data](#create-data)
# - [A Note on NumPy](#numpy)
# 
# #### [Part II: Implementation [80 pts]](#implementation)
# - [Question 4 [15 pts]: Majority Vote](#q4)
# - [Question 5 [20 pts]: Split](#q5)
# - [Question 6 [10 pts]: Train](#q6)
# - [Question 7 [15 pts]: Predict](#q7)
# - [Question 8 [10 pts]: Error Rate](#q8)
# - [Question 9 [10 pts]: Putting Everything Together](#q9)
# 
# #### [Summary](#summary)

# <a name="data"></a>
# 
# # Part I: Data Inspection
# 
# We begin by inspecting the dataset. For this assignment, we use a modified version of the *heartFailure dataset*, which contains heart failure clinical records. The original dataset consists of 299 instances and 13 attributes (*e.g.* age, diabetes, smoking, etc.). **In this assignment, we only ask you to handle attributes with binary values** (*e.g.* 0/1, yes/no). Note that you can extend your decision stump design (and decision tree in general) to handle non-binary values, but it is not required here. As such, we have removed attributes with non-binary values. The modified dataset consists of 299 instances and 5 binary attributes. More information on the heartFailure dataset can be found on this [link](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records).
# 
# We further divided the dataset into two datasets to help you with inspection and testing: `toy` and `full`. Each of these two datasets were then split into training and testing data:
# 
# * `toy_train` and `toy_test`: each contains 10 instances sampled randomly from the full dataset. There are 2 attributes, `high_blood_pressure` and `smoking`. We recommend you debug with these data.
# * `full_train` and `full_test`: the full (binary) heartFailure dataset, with 299 instances and all 5 attributes, split into 150 instances for training and 149 for testing.
# 
# Format-wise, each dataset is represented as a table, with each row representing each instance `X` and each column representing an attribute. The first row contains the name of each attribute. **The last column is always the label `Y`**.
# 
# First, run the following cell containing imports and constants.
# 

# In[2]:


from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import csv

from testing import TestHelper

TOY_TRAIN_FILE = './data/toy_train.csv'     # filepath of toy_train dataset
TOY_TEST_FILE = './data/toy_test.csv'       # filepath of toy_test dataset
FULL_TRAIN_FILE = './data/full_train.csv'   # filepath of full_train dataset
FULL_TEST_FILE = './data/full_test.csv'     # filepath of full_test dataset


# Run the cell below to visualize the `toy_train` dataset. The first column is a pandas DataFrame index column that can be ignored.

# In[3]:


toy_train = pd.read_csv(TOY_TRAIN_FILE) # load data into a panda DataFrame
toy_test = pd.read_csv(TOY_TEST_FILE)
print(toy_train)                        # show toy_train data


# To help you with data preprocessing, we provide you with the function `load_data` that 
# takes in a filepath of the train/test dataset and output a tuple of `(X, Y, attribute_names)`, where: 
# - `X` is a NumPy array with $N$ rows and $M$ columns (*i.e.* shape `(N, M)`), where $N, M$ denote the number of instances and attributes, respectively
# - `Y` is a NumPy array with shape `(N, )`, where $N$ is number of instances. **Note**: there is a crucial difference between shape `(N,)` vs. `(N, 1)` in NumPy: The former represents an 1-dimension array with $N$ elements, while the latter represents a 2-D array with $N$ rows and 1 column.
# - `attribute_names`: a list of string, with the index corresponds to the column index of `X` (*e.g.* `['high_blood_pressure', 'smoking']`).

# In[4]:


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """This function takes in the filepath of the data and outputs the tuple of 
    (X, Y, attribute_names). This reader assumes the label Y is always positioned 
    at the last column

    Parameters
    ----------
    filename: type `str`
        The filepath to the dataset

    Returns
    -------
    A tuple (X, Y, attributes_name) where
        X: type `np.ndarray`, shape (N, M)
            Numpy arrays with N rows, M columns containing the attribute values for N training instances
        Y: type `np.ndarray`, shape (N, )
            Numpy arrays with N rows containing the true labels for N training instances
        attribute_names: type `List[str]`
            Names of the attributes
    """
    X: List[str] = []
    Y: List[str] = []
    attribute_names: List[str] = []
    with open(filename, 'r') as f: 
        reader = csv.reader(f)

        # get attribute list, get rid of label header
        attribute_names = next(reader)[:-1] 

        # get X, Y values and convert them to numpy arrays
        for row in reader: 
            X.append(row[:-1])
            Y.append(row[-1])
        X = np.array(X)
        Y = np.array(Y)

    return (X, Y, attribute_names)


# Load the following cell to visualize `X`, `Y`, and `attribute_names` of the `toy_train` dataset. Don't forget to load the cell containing `load_data` function above. Questions 1 - 3 then check your understanding of the data

# In[5]:


toy_X_train, toy_Y_train, toy_attributes = load_data(TOY_TRAIN_FILE)
print('Shape of X =', toy_X_train.shape)
print('X = ', toy_X_train)
print('Shape of Y = ', toy_Y_train.shape)
print('Y = ', toy_Y_train)
print('attribute_names =', toy_attributes)


# [Back to top](#index)
# 
# <a name="q1"></a>
# ### Question 1: Number of Attribute [5 pt]
# What is the number of attributes in the `toy_train` dataset? Assign your answer as a Python integer to variable `ans1` below.

# In[6]:


### GRADED
### YOUR SOLUTION HERE 
ans1 = toy_X_train.shape[1]
#print(ans1)# replace with your answer

###
### YOUR CODE HERE
###


# In[7]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# [Back to top](#index)
# 
# <a name="q2"></a>
# ### Question 2: Name of Attribute [5 pt]
# 
# What attribute does column index 1 of `X` in `toy_train` represent? Note that in Python, indexing starts from 0 (*i.e.* index 0 denotes the first element). Assign your answer as a Python string to variable `ans2` below.

# In[8]:


### GRADED
### YOUR SOLUTION HERE
ans2 = toy_attributes[1] # replace with your answer
#print(ans2)
###
### YOUR CODE HERE
###


# In[9]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# [Back to top](#index)
# 
# <a name="q3"></a>
#  
# ### Question 3: Value of Attribute [10 pt]
# 
# What are all the possible values of the attribute in question 2, in the heartFailure dataset? Please list them in ascending/alphabetical order in a Python list and assign your answer to variable `ans3` (*e.g.* `ans3 = ['value1', 'value2']`). Each element in the list should have type string.

# In[10]:


### GRADED
### YOUR SOLUTION HERE 
ans3 = sorted(toy_X_train[1]) # replace with your answer
print(ans3)

###
### YOUR CODE HERE
###


# In[11]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# <a name="create-data"></a>
# 
# ### Create Your Own Data
# 
# In addition to the `toy` dataset, you are welcome to create your own data for debugging purposes. You can create the data directly in the code block, instead of loading it from a file. Below we show you an example of `X` array with 4 rows (instances) and 2 columns (attributes), and the associated label array `Y`.

# In[12]:


# An example of X array with shape (4, 2) and Y array with shape (4,).
# The values are all 0 as placeholder, but you can modify the value 
# and the arrays as you wish. 
X = np.array([
    ['0','0'],
    ['0','0'],
    ['0','0'],
    ['0','0']
])
Y = np.array([
    '0',
    '0',
    '0',
    '0'
])


# For each of the implementation questions below, we provide you with a code cell that you can use to debug and play around. Most of our test cases include several small, hardcoded datasets like the above cell, as well as the `toy` and `full` heart datasets. You can use the code cell to debug the test cases that your program fails to pass.

# <a name="numpy"></a>
# 
# ### A Note on NumPy
# We will be working with NumPy arrays to represent our data throughout this and subsequent assignments. As such, we recommend you familarize yourself with the NumPy library if you have not encountered it before. Here are some resources to help you: 
# - NumPy Quickstart Tutorial: https://numpy.org/devdocs/user/quickstart.html
# - NumPy Absolute Basics for Beginner: https://numpy.org/devdocs/user/absolute_beginners.html
# - NumPy Other Tutorials: https://numpy.org/learn/
# - NumPy Documentation: https://numpy.org/doc/stable/
# 
# We recommend you start with the NumPy Quickstart tutorial, followed by NumPy Absolute Basics for Beginner Tutorial. You are not expected to go through all of the tutorials. The NumPy Documentation is also a useful resource if you need to look up information needed for your implementation. For this assignment, however, please familarize yourself with at least:
# - Basic properties of NumPy arrays: type, shapes, slicing and indexing notations.
# - Conversion from Python lists to NumPy arrays
# 
# To check your readiness with using NumPy for this assignment, see if you can answer the following questions, given a 2D NumPy array `X` with shape `(N, M)`: 
# 1. How do we obtain the shape of `X`?
# 2. How do we obtain the value of element at row $i$, column $j$ of `X`?
# 3. Given a Python list `L`, how do we convert `L` to the corresponding NumPy array?
# 

# [Back to top](#index)
# 
# <a name="implementation"></a>
# 
# # Part II: Implementation
# 
# In this section, you will be implementing the decision stump. Questions 4 - 8 ask you to implement the components of the decision stump, including majority vote at each child node ([question 4](#q4)), splitting at the parent node ([question 5](#q5)), training ([question 6](#q6)), making predictions ([question 7](#q7)), and computing error rate ([question 8](#q8)). You will then put everything together in [question 9](#q9).
# 
# If you remains unclear on how the decision stump learner works, please re-visit lecture video 12

# <a name="q4"></a>
# ### Question 4: Majority Vote [15 pts]
# 
# To begin, implement a `majority_vote` algorithm that takes in `X, Y` NumPy arrays and outputs the desired label. If there are an equal number of labels, the tie-breaker goes to the label value that appears first in the dataset (*i.e.* if there is an equal number of `'0'` and `'1'`, and `'1'` appears first, then `majority_vote` outputs `'1'`). In addition, you are guaranteed that there is no empty dataset in our test cases.
# 
# **Hint**: if you encounter wrong type issues, make sure your answer is of type string. For any variable `x`, you can convert it to a string representation using `str(x)`

# In[13]:


### GRADED
### YOUR SOLUTION HERE 
def majority_vote(X: np.ndarray, Y: np.ndarray) -> str:
    """This function computes the output label of the given dataset, following the 
    majority vote algorithm

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances

    Returns the majority label
    """
    ### YOUR CODE HERE
    d = {}
    for i in Y:
        if i in d:
            d[i]+=1
        else:
            d[i] =1
    test_val = list(d.values())[0]
  
    for ele in d:
        if d[ele] != test_val:
            inverse = [(value, key) for key, value in d.items()]
            return (max(inverse)[1])
    return Y[0] # replace this line with your return statement

###
### YOUR CODE HERE
###


# For debugging purposes, you can inspect the output of your code with the example below. 

# In[14]:


# Create a toy dataset for you to debug. Replace X, Y with the test cases that
# you did not pass
X = np.array([
    ['0','0'],
    ['0','0'],
    ['0','0'],
    ['0','0']
])
Y = np.array([
    '1',
    '0',
    '0',
    '0'
])

# call your function on the toy dataset
label = majority_vote(X, Y)

# print the result
print('label=', label) # expected answer is '0', type 'str'


# In[15]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[16]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# You can also inspect the output from the `toy` and `full` datasets. Even if you passed all the tests, it is still instructional to see what your functions output.

# In[17]:


# Example output of your majority_vote function on toy train dataset
toy_X_train, toy_Y_train, toy_attributes = load_data(TOY_TRAIN_FILE)
label = majority_vote(toy_X_train, toy_Y_train)
print(label) # expected answer is '0', type 'str'


# [Back to top](#index)
# 
# <a name="q5"></a>
# 
# ### Question 5: Split [20 pts]
# 
# Next, implement a `split` function that takes in a dataset (`X,Y` arrays), 
# the attribute to split on (represented as an column index of `X`), and returns a tuple of the split datasets.
# Specifically: 
# - For input `X` and `Y`, you should output the tuple `(X_left, Y_left, X_right, Y_right)`, all of type NumPy arrays. 
# - The left and right values of the attribute should be in alphabetical order (e.g. `'0'` on the left, `'1'` on the right). The left sub-dataset (`X_left, Y_left`) should corresponds to the left attribute value, and similarly for the right sub-dataset (`X_right`, `Y_right`).
# - `X` is guaranteed to be split into 2 non-empty datasets. In other words, the values of the split attributes in `X` is always more than 1 (otherwise, we would already be in a leaf node, and there is no need to keep splitting). As a side note when implementing decision tree, you should always check for this condition before splitting.
# 
# **Hint**: If you find your output contains the same examples as the expected output, but in different order, *you should modify your code to loop through `X, Y` in order of appearance (e.g. something like `for X_instance, Y_instance in zip(X, Y)`)*. While one can argue that the splitted datasets are the same regardless of the order of the examples, having them in the same order makes it much easier for us to automatically grade your work. If this doesn't apply to you, please ignore this hint. 

# In[18]:


### GRADED
### YOUR SOLUTION HERE 
def split(X: np.ndarray, 
          Y: np.ndarray, 
          split_attribute: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function takes a dataset and splits it into two sub-datasets according 
    to the values of the split attribute. The left and right values of the split 
    attribute should be in alphabetical order. The left dataset should correspond 
    to the left attribute value, and similarly for the right dataset. 

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances
    split_attribute: type `int`
        The index of the attribute to split the dataset on

    Returns the tuple of two sub-datasets, i.e. (X_left, Y_left, X_right, Y_right)
    """
    ### YOUR CODE HERE 
    keys = set()
    for i in X:
        keys.add(i[split_attribute])
        
    keys =sorted(list(keys))
    X_left, Y_left, X_right, Y_right = [], [], [],[]
    for i in range(len(X)):
        if X[i][split_attribute] == keys[0]:
            X_left.append(X[i])
            Y_left.append(Y[i])
        else:
            X_right.append(X[i])
            Y_right.append(Y[i])
    return (np.array(X_left), np.array(Y_left), np.array(X_right), np.array(Y_right)) # replace this line with your return statement

###
### YOUR CODE HERE
###


# Similar to `majority_vote`, you can inspect your function output by calling `split` on either your own created dataset, or load dataset from file.

# In[19]:


# Create a toy dataset for you to debug. Replace X, Y with the test cases that
# you did not pass.  You can load dataset with something like
# X, Y, attribute = load_data(TOY_TRAIN_FILE)
X = np.array([
    ['1','0'],
    ['1','0'],
    ['0','0'],
    ['0','1']
])
Y = np.array([
    '1',
    '0',
    '0',
    '0'
])
split_attribute = 0  

# call your function on the toy dataset
X_left, Y_left, X_right, Y_right = split(X, Y, split_attribute=split_attribute)
 
# print the result. You can print X_left, Y_left, X_right, Y_right as well
print(X_left)
# expected result for X_left is 
# [['0' '0']
#  ['0' '1']]


# In[20]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[21]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# [Back to top](#index)
# 
# <a name="q6"></a>
# 
# ### Question 6: Train [10 pts]
# 
# Implement the `train` function for your decision stump. This function takes in train dataset `X_train, Y_train`, and the index of the split attribute `attribute_index`. The output of this function is a tuple of two strings, where the first element of the tuple is the label of the left child node, and the second element is the label of the right child node. 
# 
# We recommend you make use of the `majority_vote` and `split` functions that you implemented above. Also, please make sure the output type is correct -- a tuple of two strings.

# In[22]:


### GRADED
### YOUR SOLUTION HERE
def train(X_train: np.ndarray, Y_train: np.ndarray, attribute_index: int) -> Tuple[str, str]:
    """This function takes the training dataset and a split attribute and outputs the 
    tuple of (left_label, right_label), corresponding the label on the left and right 
    leaf nodes of the decision stump

    Parameters
    ----------
    X_train: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N training instances
    Y_train: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N training instances
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns the tuple of labels, i.e. (left_label, right_label)
    """
    ### YOUR CODE HERE 
    X_left, Y_left, X_right, Y_right = split(X_train, Y_train, split_attribute=split_attribute)
    left_label = majority_vote(X_left, Y_left)
    right_label = majority_vote(X_right, Y_right)
    return (left_label, right_label) # replace this line with your return statement

###
### YOUR CODE HERE
###


# Similar to question 4 and 5, you can debug your `train` function with the following code block.

# In[23]:


# Create a toy dataset for you to debug. Replace X, Y with the test cases that
# you did not pass.  You can load dataset with something like
# X, Y, attribute = load_data(TOY_TRAIN_FILE)
X = np.array([
    ['1','0'],
    ['1','0'],
    ['0','0'],
    ['0','1']
])
Y = np.array([
    '1',
    '0',
    '0',
    '0'
])
split_attribute = 0  

# call your function on the toy dataset
left_label, right_label = train(X, Y, attribute_index=split_attribute)

# print the result
print('left label =', left_label)   # expected result is '0'
print('right label =', right_label) # expected result is '1'


# In[24]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[25]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# [Back to top](#index)
# 
# <a name="q7"></a>
# 
# ### Question 7: Predict [15 pts]
# 
# Implement the `predict` function that takes in your trained stump (output of the `train` function), an `X` array, the split `attribute_index` and predicts the labels of `X`. The output should be a 1-D NumPy array with the same number of elements as instances in `X`.
# 

# In[26]:


### GRADED
### YOUR SOLUTION HERE
def predict(left_label: str, right_label: str, X: np.ndarray, attribute_index: int) -> np.ndarray:
    """This function takes in the output of the train function (left_label, right_label), 
    the dataset without label (i.e. X), and the split attribute index, and returns the 
    label predictions for X

    Parameters
    ----------
    left_label: type `str`
        The label corresponds to the left leaf node of the decision stump
    right_label: type `str`
        The label corresponds to the right leaf node of the decision stump
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns the numpy arrays with shape (N,) containing the label predictions for X
    """
    ### YOUR CODE HERE 
    ans  =  []
    keys = set()
    for i in X:
        keys.add(i[split_attribute])
        
    keys =sorted(list(keys))
    for i in range(len(X)):
        if X[i][split_attribute] == keys[0]:
            ans.append(left_label)
        else:
            ans.append(right_label)
        
    return np.array(ans) # replace this line with your return statement  

###
### YOUR CODE HERE
###


# Similar to the above questions, you can debug your implementation using the following code block.

# In[27]:


# Create a toy dataset for you to debug. Replace X, Y with the test cases that
# you did not pass. You can load dataset with something like
# X, Y, attribute = load_data(TOY_TRAIN_FILE)
X = np.array([
    ['1', '0'],
    ['1', '0'],
    ['0', '0'],
    ['0', '0']
])
(left_label, right_label) = ('1', '1')
split_attribute = 0  

# call your function on the debug dataset
Y_pred = predict(left_label, right_label, X, attribute_index=split_attribute)

# print the result
print(Y_pred) # expected result is ['1' '1' '1' '1']


# In[28]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[29]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# [Back to top](#index)
# 
# <a name="q8"></a>
# 
# ### Question 8: Error Rate [10 pts]
# 
# Implement the `error_rate` function that takes in the true `Y` values and the `Y_pred` predictions (output of `predict` function) and computes the error rate, which is the number of incorrect instances divided by the number of total instances.

# In[30]:


### GRADED
### YOUR SOLUTION HERE
def error_rate(Y: np.ndarray, Y_pred: np.ndarray) -> float:    
    """This function computes the error rate (i.e. number of incorrectly predicted
    instances divided by total number of instances)

    Parameters
    ----------
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances
    Y_pred: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the predicted labels for N instances

    Returns the error rate, which is a float value between 0 and 1 
    """
    ### YOUR CODE HERE 
    corr = len(Y)
    incorr = 0
    for i in range(len(Y)):
        if Y[i] != Y_pred[i]:
            incorr+=1
        
    return incorr/corr # replace this line with your return statement  

###
### YOUR CODE HERE
###


# Similar to the above questions, you can debug your implementation using the following code block.

# In[31]:


# Create a toy dataset for you to debug. Replace Y, Y_pred with the test cases that
# you did not pass.  You can load dataset with something like
# X, Y, attribute = load_data(TOY_TRAIN_FILE)
Y = np.array(['1','0','0','0'])
Y_pred = np.array(['1','1','0','0'])

# call your function
rate = error_rate(Y, Y_pred)

# print the result
print(rate) # expected result is 0.25


# In[32]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# [Back to top](#index)
# 
# <a name="q9"></a>
# 
# ### Question 9: Putting Everything Together [10 pts] 
# 
# In this last task, you should make use of the previous functions to implement `train_and_test` to obtain the results of your decision stump given a dataset. The function `train_and_test` takes in the filepath of the train dataset, the filepath of the test dataset, the split attribute index, and returns the dictionary containing the relevant outputs (*e.g.* train and test error rates). To help you out, we have implemented the skeleton code, and your task is to fill in the lines marked `# please complete` with the appropriate code.
# 
# **Hint** you should find yourself doing the following:
# * Use the function `load_data` to load in the training and testing datasets 
# * Use the function `train` to train your decision stump on the training data
# * Use the function `predict` to get the training and testing predictions, with your trained decision stump
# * Use the function `error_rate` to get the train and test error rates

# In[41]:


### GRADED
### YOUR SOLUTION
def train_and_test(train_filename: str, test_filename: str, attribute_index: int) -> Dict[str, Any]:
    """This function ties the above implemented functions together. The inputs are 
    filepaths of the train and test datasets as well as the split attribute index. The
    output is a dictionary of relevant information (e.g. train and test error rates)

    Parameters
    ----------
    train_filename: type `str`
        The filepath to the training file
    test_filename: type `str`
        The filepath to the testing file
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns an output dictionary
    """
    X_train, Y_train, attribute_names = load_data(train_filename) # please complete
    X_test, Y_test, _ = load_data(test_filename) # please complete

    left_label, right_label = train(X_train, Y_train, attribute_index) # please complete
    Y_pred_train = predict(left_label, right_label, X_train, attribute_index) # please complete
    Y_pred_test = predict(left_label, right_label, X_test, attribute_index) # please complete

    train_error_rate = error_rate(Y_train, Y_pred_train) # please complete
    test_error_rate = error_rate(Y_test, Y_pred_test) # please complete

    return {
        'attribute_names' : attribute_names,
        'stump'           : (left_label, right_label),
        'train_error_rate': train_error_rate,
        'test_error_rate' : test_error_rate
    }
    
###
### YOUR CODE HERE
###


# You can inspect your implementation with the code block below. Feel free to change the `train_file` and `test_file` to either toy or full datasets, as well as varying the `attribute_index`

# In[43]:


train_file = FULL_TRAIN_FILE 
test_file = FULL_TEST_FILE
attribute_index = 0

#train_file = TOY_TRAIN_FILE
#test_file = TOY_TEST_FILE
#attribute_index = 0
# call your function
output_dict = train_and_test(train_file, test_file, attribute_index=attribute_index)
print(train_file)
print(test_file)
print(attribute_index)
# print the result
print('attributes: ', output_dict['attribute_names'])
print('stump: ', output_dict['stump'])  
print('train error: ',output_dict['train_error_rate']) # expected result is 0.3
print('test error: ', output_dict['test_error_rate']) # expected result is 0.34


# In[35]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# [Back to top](#index)
# 
# <a name="summary"></a>
# 
# # Summary 
# 
# Congratulations on building your first machine learning model! Although the decision stump is a relatively simple learner, it is a building block for decision tree, a popular and more complex learner that you will be building in assignment 2. 
# 
# Below we show you the visualization of the decision stump from the previous code block, using the default value (full dataset, `attribute_index=0`). This is the same stump in the beginning image, with slightly different notations that you have accustomed to when working on this assignment
# 
# <img src="./img/stump_conclusion.png" width="400" height="400">
# 
# We also show you the result table of accuracy, which is $1-\text{error_rate}$:
# 
# | Model | Split attribute | Train accuracy (%) | Test accuracy (%) |
# | :--- | :--- | :--- | :--- |
# | Decision stump | 0 | 70% | 65.8%  |
# 
# Let's recap what you have learned in this assignment:
# - Construct a decision stump with a split on one node and use the majority vote algorithm at leaf nodes
# - Become familiar with a machine learning algorithm and mindset
# - Gain further familiarity coding in Python and related modules (e.g. NumPy)
