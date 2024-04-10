import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Data
data = []
file_path = r'C:\Users\Brandon\Documents\Datasets\2023.enriched.jsonl'
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))
df = pd.json_normalize(data)

# Clean the Data
# Assuming no specific cleaning based on column types is needed here

# Feature Extraction
# Text column "description" to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
df['description'] = df['description'].fillna('')  # Replace NaN with empty strings
tfidf_features = tfidf_vectorizer.fit_transform(df['description'])

# One-hot encode categorical variables "job_type" and "industry"
df = pd.get_dummies(df, columns=['job_type', 'industry'])

# Assuming 'is_tech' is the target variable for a classification task
target_variable = df['is_tech']

# Split the Data
X = df.drop('is_tech', axis=1)  # Drop the target variable to isolate features
y = target_variable  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize or Standardize the Data
# It's common to normalize/standardize numerical features but not the one-hot encoded or tf-idf features
# Assuming there are numerical features that need scaling, identified as "numerical_column1", "numerical_column2"
scaler = StandardScaler()
numerical_features = ['numerical_column1', 'numerical_column2']  # Replace with your actual numerical columns
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Note: This code snippet assumes `df` has columns like 'description', 'job_type', 'industry', and 'is_tech'.
# It also assumes numerical columns need scaling. Adjust these placeholders to fit your dataset's actual structure.