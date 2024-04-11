import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Reading the .jsonl file path from the text file...")
text_file_with_path = r'C:\Users\Brandon\Documents\Repositorys\MPTJB\MPTJB\AI\jsonl_file_path.txt'
with open(text_file_with_path, 'r', encoding='utf-8') as f:
    file_path = f.readline().strip()
print(f".jsonl file path read successfully: {file_path}")

print("Loading data from .jsonl file...")
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))
print("Data loaded successfully.")

df = pd.json_normalize(data)
print("Data normalized into DataFrame.")

# Clean the Data
# Assuming no specific cleaning based on column types is needed here
print("Cleaning data...")
# Add any specific data cleaning steps here
print("Data cleaned.")

# Feature Extraction
print("Extracting features...")
# Text column "description" to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
df['description'] = df['description'].fillna('')  # Replace NaN with empty strings
tfidf_features = tfidf_vectorizer.fit_transform(df['description'])
print("TF-IDF features extracted.")

# One-hot encode categorical variables "occupation_field" and "industry"
df = pd.get_dummies(df, columns=['occupation_field', 'industry'])
print("Categorical variables one-hot encoded.")

# Assuming 'is_tech' is the target variable for a classification task
target_variable = df['is_tech']

# Split the Data
print("Splitting data into training and test sets...")
X = df.drop('is_tech', axis=1)  # Drop the target variable to isolate features
y = target_variable  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# Note: Normalization/Standardization section is commented out as it's optional and depends on the presence of numerical features
# print("Normalizing numerical features...")
# scaler = StandardScaler()
# numerical_features = ['numerical_column1', 'numerical_column2']  # Replace with your actual numerical columns
# X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
# X_test[numerical_features] = scaler.transform(X_test[numerical_features])
# print("Normalization complete.")

print("Preprocessing complete. The script is ready for machine learning modeling.")
