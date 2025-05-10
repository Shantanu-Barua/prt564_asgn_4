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

### Convert death counts to category: Low, Medium, High ###

# Compute mean and standard deviation
mean_death = ms_df['Total death'].mean()
std_death = ms_df['Total death'].std()

# print(mean_death, std_death)

# Define thresholds
# low_thresh = mean_death - std_death
high_thresh = mean_death + std_death

# print(low_thresh, high_thresh)

# Create a new column with categorical labels
def categorize_death(val):
    if val == 0:
        return 'Low'
    elif val > high_thresh:
        return 'High'
    else:
        return 'Medium'

ms_df['Death Category'] = ms_df['Total death'].apply(categorize_death)

# print(ms_df.columns)


# print(ms_df[['Total death', 'Death Category']])

### 2. Age groups, diseases ###

wanted = ['0-7 days', 'Immature','Adult', 'Egg Peritonitis & Salpingitis', 'Colisepticaemia', "Yolk sac infection/\nomphalitis", 'Broiler ascites', 'Red Mite', 'Neoplasm',	"Marek's\nDisease", 'Death Category']

age_ds_df = ms_df[wanted]
# print(age_ds_df)

age_cols = ['0-7 days', 'Immature', 'Adult']
df_age_melted = age_ds_df.melt(id_vars=['Egg Peritonitis & Salpingitis', 'Colisepticaemia', "Yolk sac infection/\nomphalitis", 'Broiler ascites', 'Red Mite', 'Neoplasm',	"Marek's\nDisease", 'Death Category'], value_vars=age_cols,
                    var_name='Age Group')

df_age_melted.drop(columns=['value'], inplace=True)


### one hot encoding for age group###

df_age_melted = pd.get_dummies(df_age_melted, columns=["Age Group"], drop_first=True, dtype=int)

# print(df_age_melted)

# Use selected features for classification
X_selected = df_age_melted.drop(columns=['Death Category'])
y = df_age_melted['Death Category']

# print(X_selected)
y_encoded = LabelEncoder().fit_transform(y)

# Train/test split (stratified to balance classes)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Naive Bayes classification
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification report
target_labels = ['Low', 'Medium', 'High']
report = classification_report(y_test, y_pred, target_names=target_labels)
print("Classification Report:\n", report)

# Convert report to DataFrame for heatmap
report_dict = classification_report(y_test, y_pred, target_names=target_labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Plot classification report heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu')
plt.title("Classification Report (Precision, Recall, F1-score)")
plt.tight_layout()
plt.show()
