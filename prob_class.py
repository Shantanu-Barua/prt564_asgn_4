import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the master dataset
ms_df = pd.read_excel("AvianDataset.xlsx", sheet_name="Number")

# print(ms_df)
# nan_count = ms_df.isna().sum()
# print(nan_count)


# replace NaN with 0
ms_df = ms_df.replace(np.nan, 0)

# nan_count = ms_df.isna().sum()
# print(nan_count)


### 1. Best feature selection: SelectKBest ###

# Step 1: Drop non-feature columns
non_features = ['No.', 'Month', 'Year', 'Total death'] 
X = ms_df.drop(columns=non_features)
y = ms_df['Total death']

# Optional: Convert 'y' into categories if you want to use classification
# Example: Categorize into Low/Medium/High death
y = pd.cut(y, bins=[-1, 2, 5, float('inf')], labels=['Low', 'Medium', 'High'])

# Encode categorical labels
y_encoded = LabelEncoder().fit_transform(y)

# Step 2: Apply SelectKBest to choose top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y_encoded)

# Visualize
feature_names = X.columns[selector.get_support()]
feature_scores = selector.scores_[selector.get_support()]

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_scores, y=feature_names, palette='viridis')
plt.title("Top 10 Feature Scores (SelectKBest - f_classif)")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Step 3: Get selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected features (SelectKBest):", selected_features.tolist())

# Step 4: Use selected features for classification
X_selected = ms_df[selected_features]

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

# Step 6: Naive Bayes classification
model = GaussianNB()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate classification
y_pred = model.predict(X_test)
print("Classification report: \n")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

report_dict = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Plot as heatmap for classification report
plt.figure(figsize=(8, 5))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu') 
plt.title("Classification Report (Precision, Recall, F1-score)")
plt.tight_layout()
plt.show()