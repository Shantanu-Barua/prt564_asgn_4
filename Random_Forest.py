#Random Forest
#Import Packages and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#load data and select the 'Number' sheet
full_df = pd.read_excel("AvianDataset.xlsx", sheet_name=['Number', 'Proportion'])
df = full_df['Number']


#Preprocessing Data
#-------------------
#drop columns that might leakage prone or highly correlated columns
df = df.drop(columns=['Adult', 'Immature'])

#create a 'seasons' feature based on month for seasonal information
df['Seasons'] = df['Month'] % 12 // 3 + 1
df['Winter'] = (df['Seasons'] == 1).astype(int)
df['Summer'] = (df['Seasons'] == 3).astype(int)

#'Temperature' column using min and max temp of each region
#average based on the min and max temperateues creates the 'Temperature' feature for each region
df['scotland_temp'] = df[['Max_temp\nScotland', 'Min_temp\nScotland']].mean(axis=1)
df['east_england_temp'] = df[['Max_temp\nEast of England', 'Min_temp\nEast of England']].mean(axis=1)
df['southwest_temp'] = df[['Max_temp\nSouthwest', 'Min_temp\nSouthwest']].mean(axis=1)
df['wales_temp'] = df[['Max_temp\nWales', 'Min_temp\nWales']].mean(axis=1)
df['westmidlands_temp'] = df[['Max_temp\nWest Midlands', 'Min_temp\nWest Midlands']].mean(axis=1)
df['southeast_temp'] = df[['Max_temp\nSouth East', 'Min_temp\nSouth East']].mean(axis=1)

#interaction for rainfall
#temp * rainfall for each region
df['Temp_x_Rain_scotland'] = df['scotland_temp'] * df['Hours of rainfall\nScotland']
df['Temp_x_Rain_east_england'] = df['east_england_temp'] * df['Hours of rainfall\nEast of England']
df['Temp_x_Rain_southwest'] = df['southwest_temp'] * df['Hours of rainfall\nSouthwest']
df['Temp_x_Rain_wales'] = df['wales_temp'] * df['Hours of rainfall\nWales']
df['Temp_x_Rain_westmidlands'] = df['westmidlands_temp'] * df['Hours of rainfall\nWest Midlands']
df['Temp_x_Rain_southeast'] = df['southeast_temp'] * df['Hours of rainfall\nSouth East']


#Target variables
#----------------
#convert 'Total death' into 3 categorical bins (0 = low, 1 = medium, 2 = high)
y = pd.qcut(df['Total death'], q=3, labels=[0,1,2]).astype(int)

#extract features and drop the target column
x = df.drop(columns=['Total death'])


#Feature types
#-------------
#identifying numerical and categorical features and seperating them
num_features = x.select_dtypes(include=np.number).columns.tolist()
cat_features = x.select_dtypes(include=['object', 'category']).columns.tolist()


#Preprocessing Pipline
#---------------------
#numerical pipeline: impute and scale missing values
num_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

#categorical pipeline: one-hot encode categorical features
cat_tranform = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

#combine the numerical and categocial pre-processing
preprocessor = ColumnTransformer([
    ('num', num_transform, num_features),
    ('cat', cat_tranform, cat_features)
])


#Radom forrest classfier adn Hyperparameter Grid
#----------------------------------------------
#pipeline with preprocessor classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=0))
])

#hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}


#Train model
#------------
#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

#gridsearch with 5-fold CV used for hyperparaketer tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

#best estimator and predictions
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)

#Evaluation
#-----------
#evaluate model
print("Best Parameters Found:", grid_search.best_params_)
print("\nRandom Forest Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#Visualisation
#--------------
#Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1', 'Pred 2'], yticklabels=['Actual 0', 'Actual 1', 'Actual 2'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

#Classification report as table
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df.round(2).to_string())

#precision, recall, F1 score plotted for each class
report_df[['precision', 'recall', 'f1-score']].drop('accuracy').iloc[:3].plot(kind='bar', figsize=(8, 6))
plt.title('Precision, Recall, F1-Score per Class')
plt.xlabel('Class')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.show()
