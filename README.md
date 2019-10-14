# TCD ML Comp. 2019/20 - Income Prediction (Ind.)

## Goal
To predict the income of people


## Steps
1. Remove Features that do not add benifit to the model ('Instance', 'Hair Colour')
2. Converted non meaningful data to null or to the most related data e.g 0 = male
3. Impute null values in the categoric and numeric data using simpleImputer
4. Use get_dummies to convert categorical data to numeric columns
5. Do steps 1-4 for the test dataset
6. Fill in the missing columns with zero where the test data had extra items in its
    categoric features
7. Normalise all the axis by taking the log and using MinMaxScaler for all features
8. Perform MLP Regression on the dataset where max iterations is 1500 as it is a large
    dataset
9. Store back into file

## Final Top Score
60434.127  
