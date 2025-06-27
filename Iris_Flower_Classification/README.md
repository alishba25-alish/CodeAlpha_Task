==== Iris Flower Classification using Logistic Regression ====

## Description:
This project uses the popular *Iris dataset* to predict the species of an Iris flower based on its measurements (sepal & petal length and width).  
We trained a *Logistic Regression* model using *Scikit-learn*, and evaluated its accuracy and performance.

## Dataset Features:

- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species *(target)*

## Libraries Used:

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn

## Steps Performed:

1. Loaded the dataset using Pandas
2. Checked for missing values (.info())
3. Described the data (.describe())
4. Dropped Id column
5. Encoded target (Species) using LabelEncoder
6. Split data into 80% training and 20% testing using train_test_split
7. Trained Logistic Regression model
8. Made predictions on test data
9. Evaluated model using:
   - Accuracy Score  
   - Classification Report  
   - Confusion Matrix (plotted using Seaborn)

## Model Accuracy:

Accuracy: 1.0
> The model correctly predicted all flower species in the test set.

## Project Files:

- iris_project.py – Main python file containing all model code
- README.md – Documentation file
- iris.csv – Dataset file