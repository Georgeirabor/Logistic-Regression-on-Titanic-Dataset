# Logistic Regression Model for Titanic Survival Prediction

## Overview
This is my first machine learning project and I would be creating a Logistic Regression model on the Titanic dataset from Kaggle. The goal is to develop a logistic regression model that predicts survival outcomes on the Titanic, while gaining insights into the factors influencing survival. Logistic regression, being a simple yet powerful algorithm, is well-suited for binary classification problems like this one.

## Dataset Overview
The Titanic dataset is a classic in the machine learning community, offering a range of passenger data such as age, gender, class, ticket fare, and survival status. This dataset is available on Kaggle and is divided into training and test datasets. Key features in the dataset include:

- **PassengerId**: Unique identifier for each passenger
- **Survived**: Indicates whether a passenger survived (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger in years
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Fare paid by the passenger
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Data Processing and Analysis Steps

### Data Collection
The Titanic dataset was sourced from Kaggle in a CSV files: `complete.csv`. The training dataset, which is labeled, is used to train the logistic regression model, while the test set is reserved for evaluating model performance.

### Key Problem
The primary objective is to answer the question: **Can we predict a passenger's survival status based on the available features?**

Some additional points explored include:
- What impact do factors like gender, class, and age have on survival rates?
- Is it possible to achieve a model accuracy of at least 75%?

### Data Exploration and Preprocessing

1. **Initial Data Review**:
   - Using Python's Pandas library, the data was loaded to review its structure and identify missing or inconsistent values.

2. **Handling Missing Data**:
   - **Age**: Missing age values were filled in with the median age of the passengers:  
     `train_data['Age'].fillna(train_data['Age'].median(), inplace=True)`
   - **Embarked**: Rows with missing Embarked values were dropped, as there were only two such instances:
     `train_data.dropna(subset=['Embarked'], inplace=True)`
   - **Cabin**: This feature was excluded from analysis due to a significant number of missing values.

3. **Categorical Data Encoding**:
   - Categorical variables such as Sex and Embarked were converted to numerical values through categorical encoding to prepare the data for modeling.

### Preparing Data for the Model

The dataset was divided into features (X) and the target variable (y), and then split into training and validation subsets for model training and evaluation.

### Building and Evaluating the Model

1. **Model Training**:
   - A logistic regression model was trained using Scikit-learn:
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression()
     model.fit(X_train, y_train)
     ```

2. **Model Evaluation**:
   - The model’s performance was assessed using metrics such as accuracy, a confusion matrix, and a classification report to evaluate its predictive capabilities.

### Conclusion and Recommendations

- **Insights**: The analysis revealed that demographic and socio-economic factors (such as gender and class) played a significant role in determining the likelihood of survival.
- **Future Directions**: I recommend improving safety measures for vulnerable groups (like women and children) and enhancing conditions in third-class sections to boost survival rates among economically disadvantaged passengers.
- **Next Steps**: Moving forward, I plan to explore more advanced models like Random Forests and Gradient Boosting to improve prediction accuracy.

This project provided valuable hands-on experience in data cleaning, feature engineering, and model evaluation, marking a great starting point in my machine learning journey.
