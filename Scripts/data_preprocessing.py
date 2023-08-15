#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# In[9]:


def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path : Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)


# In[ ]:


def handle_outliers(data, continuous_columns, method='capping', lower_bound_factor=1.5, upper_bound_factor=1.5):
    """
    Handle outliers in the continuous columns.
    
    Args:
        data : Input DataFrame.
        continuous_columns : List of continuous numerical columns.
        lower_bound_factor: Factor for lower bound capping.
        upper_bound_factor : Factor for upper bound capping.
        
    Returns:
        pd.DataFrame: DataFrame with outliers handled.
    """
    if method == 'capping':
        data = data.copy()  
        for var in continuous_columns:
            column = data[var]
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - lower_bound_factor * IQR
            upper_bound = Q3 + upper_bound_factor * IQR

            data.loc[:, var] = column.clip(lower_bound, upper_bound)
            
        return data
    elif method == 'other_method':
        # todo
        pass
    else:
        raise ValueError("Invalid outlier handling method specified.")
        

# In[16]:


def handle_null_values(data):
    # Check for null values and drop rows with null values
    if data.isnull().values.any():
        data = data.dropna()
    return data


# In[15]:


def handle_duplicates(data):
    # Check for duplicates and drop them
    if data.duplicated().any():
        data = data.drop_duplicates()
    return data


# In[17]:


def clean_data(data):
    cleaned_data = handle_duplicates(data)
    cleaned_data = handle_null_values(cleaned_data)
    return cleaned_data


# In[19]:


def extract_continuous_variables(data, threshold=8):
    """
    Extract continuous numerical columns from the data.
    
    Args:
        data : Input DataFrame.
        threshold : Threshold to distinguish continuous from discrete variables.
        
    Returns:
        list: list of continuous numerical columns.
    """
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    continuous_variables = []
    for column in numeric_columns:
        unique_values = data[column].nunique()
        if unique_values >= threshold:
            continuous_variables.append(column)
    
    return list(continuous_variables)


# In[18]:


def extract_categorical_variables(data, min_unique_values=3, max_unique_values=4):
    """
    Extract categorical variables based on the number of unique values.
    
    Args:
        data : Input DataFrame.
        min_unique_values : Minimum number of unique values.
        max_unique_values : Maximum number of unique values.
        
    Returns:
        list: List of categorical variables.
    """
    unique_value_counts =  data.nunique()
    categorical_variables = unique_value_counts[(unique_value_counts > 2) & (unique_value_counts < 5)].index.tolist()
    return categorical_variables


# In[10]:


def encode_categorical(data, categorical_columns):
    """
    Encode categorical variables using one-hot encoding with pd.get_dummies.
    
    Args:
        data: Input DataFrame.
        categorical_columns : List of categorical column names.
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical columns.
    """
    data = data.copy()
    encoded_data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    return encoded_data


# In[11]:


def standardize_numerical(data, numerical_columns):
    """
    Standardize numerical features using MinMaxScaler.
    
    Args:
        data : Input DataFrame.
        numerical_columns : List of numerical column names.
        
    Returns:
        pd.DataFrame: DataFrame with standardized numerical columns.
    """
    scaler = MinMaxScaler()
    standardized_data = data.copy()
    standardized_data[numerical_columns] = scaler.fit_transform(standardized_data[numerical_columns])
    return standardized_data


# In[12]:


def split_features_target(data, target_column):
    """
    Split data into features (X) and target (y).
    
    Args:
        data : Input DataFrame.
        target_column : Name of the target column.
        
    Returns:
        pd.DataFrame: Features (X).
        pd.Series: Target (y).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


# In[13]:


def split_train_test(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets.
    
    Args:
        X : Features.
        y : Target.
        test_size : Size of the test set.
        random_state : Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Train features.
        pd.Series: Train target.
        pd.DataFrame: Test features.
        pd.Series: Test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test


# In[14]:


def save_processed_data(X_train, y_train, X_test, y_test, output_path):
    """
    Save preprocessed data into separate files.
    
    Args:
        X_train : Preprocessed training features.
        y_train : Preprocessed training target.
        X_test : Preprocessed testing features.
        y_test : Preprocessed testing target.
        output_path : Path to save the data.
    """
    
    X_train.to_csv(output_path + '/X_train.csv', index=False)
    y_train.to_csv(output_path + '/y_train.csv', index=False)
    X_test.to_csv(output_path + '/X_test.csv', index=False)
    y_test.to_csv(output_path + '/y_test.csv', index=False)

