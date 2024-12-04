# -*- coding: utf-8 -*-
"""
Created on 2024-10-15
"""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
train_df = pd.read_csv("E:\\excelR\\DS-ASS\\Logistic Regression\\Titanic_train.csv")
test_df = pd.read_csv("E:\\excelR\\DS-ASS\\Logistic Regression\\Titanic_test.csv")

# Combine datasets for preprocessing
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Data preprocessing
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Fare'].fillna(df['Fare'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# Split the dataset
train = df.iloc[:len(train_df)]
X = train.drop(columns=['Survived'])
y = train['Survived']

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Streamlit app layout
st.title('Titanic Survival Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    sex = st.sidebar.selectbox('Gender', ['male', 'female'])
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
    age = st.sidebar.number_input("Insert the Age", min_value=0, value=30)
    fare = st.sidebar.number_input("Insert the Fare", min_value=0.0, value=7.25)
    embarked = st.sidebar.selectbox('Embarked Port', ['C', 'Q', 'S'])
    sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
    parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
    
    # Encode user inputs
    sex_encoded = 1 if sex == 'female' else 0
    embarked_encoded = label_encoder.transform([embarked])[0]
    
    data = {
        'Pclass': pclass,
        'Sex': sex_encoded,
        'Age': age,
        'Fare': fare,
        'Embarked': embarked_encoded,
        'SibSp': sibsp,
        'Parch': parch
    }
    features = pd.DataFrame(data, index=[0])
    
    # Ensure the columns are in the same order as the model training
    features = features[X.columns]  # Use the columns from the training set
    
    return features

df_user_input = user_input_features()

st.subheader('User Input Parameters')
st.write(df_user_input)

# Prediction
prediction = model.predict(df_user_input)
prediction_proba = model.predict_proba(df_user_input)

st.subheader('Predicted Result')
st.write('Survived' if prediction[0] == 1 else 'Did not survive')

st.subheader('Prediction Probability')
st.write(prediction_proba)
