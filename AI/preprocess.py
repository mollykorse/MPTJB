
## BRANDON KOD V.1
# try:
#     import pandas as pd
#     import json
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler

#     # Reading JSONL file path from a text file for better project structure and security
#     print("Reading the .jsonl file path from the text file...")
#     text_file_with_path = r'c:/Systemvetenskap/Job Listings Dataset/2023.enriched.jsonl'
#     with open(text_file_with_path, 'r', encoding='utf-8') as f:
#         file_path = f.readline().strip()
#     print(f".jsonl file path read successfully: {file_path}")

#     # Load and process the .jsonl file line by line to handle large files efficiently
#     print("Loading data from .jsonl file...")
#     data = []
#     counter = 0
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#             counter += 1
#             if counter % 10000 == 0:
#                 print(f"Processed {counter} lines...")
#     print(f"Data loaded successfully. Total lines processed: {counter}")

#     # Normalizing JSON data into a flat DataFrame
#     df = pd.json_normalize(data)
#     print("Data normalized into DataFrame.")

#     # Clean the Data
#     print("Cleaning data...")
#     # Users can add specific data cleaning steps here depending on the dataset
#     print("Data cleaned.")

#     # Feature Extraction
#     print("Extracting features...")
#     # Convert 'description.text' to TF-IDF features; replace 'description.text' with your column of interest
#     if 'description.text' in df.columns:
#         # Joining lists into strings if necessary, adjust column name as needed
#         df['description.text'] = df['description.text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
#         tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#         tfidf_features = tfidf_vectorizer.fit_transform(df['description.text'])
#         print("TF-IDF features extracted.")
#     else:
#         print("'description.text' column not found, skipping TF-IDF feature extraction.")

#     # One-hot encode selected categorical variables; adjust column names as needed
#     for column in ['occupation_field', 'industry']:
#         if column in df.columns:
#             if df[column].apply(lambda x: isinstance(x, list)).any():
#                 print(f"Column '{column}' contains list-type data, considering a different handling approach.")
#             else:
#                 df = pd.get_dummies(df, columns=[column])
#                 print(f"Categorical variable '{column}' one-hot encoded.")
#         else:
#             print(f"'{column}' column not found. Skipping one-hot encoding for '{column}'.")

#     # Preparing data for machine learning
#     print("Splitting data into training and test sets...")
#     if 'is_tech' in df.columns:
#         X = df.drop('is_tech', axis=1)  # Drop the target variable to isolate features; adjust as necessary
#         y = target_variable  # Define your target variable here
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print("Data split complete.")
#     else:
#         print("Data split skipped due to missing 'is_tech' column.")

#     # Save the preprocessed dataset for future use
#     output_file_path = file_path.replace('.jsonl', '_preprocessed.csv')
#     print(f"Saving the preprocessed dataset to {output_file_path}...")
#     df.to_csv(output_file_path, index=False)
#     print("Preprocessed dataset saved successfully.")

#     print("Preprocessing complete. The script is ready for machine learning modeling.")
# except Exception as e:
#     print(f"An error occurred: {e}")



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## NY TEST KOD ENG STOPWORDS
# import pandas as pd
# import json
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from scipy.sparse import hstack

# try:
#     # Reading JSONL file path from a text file for better project structure and security
#     print("Reading the .jsonl file path from the text file...")
#     text_file_with_path = r'C:\Systemvetenskap\Python\Molkor First Fork\MPTJB\AI\Jsonl_file_path.txt'
#     with open(text_file_with_path, 'r', encoding='utf-8') as f:
#         file_path = f.readline().strip()
    
#     if not os.path.exists(file_path):
#         print(f"Error: No file found at {file_path}")
#         exit()  # Stop execution if the file is not found
    
#     print(f".jsonl file path read successfully: {file_path}")

#     # Load and process the .jsonl file line by line to handle large files efficiently
#     print("Loading data from .jsonl file...")
#     data = []
#     counter = 0
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#             counter += 1
#             if counter % 10000 == 0:
#                 print(f"Processed {counter} lines...")
#     print(f"Data loaded successfully. Total lines processed: {counter}")

#     # Normalizing JSON data into a flat DataFrame
#     df = pd.json_normalize(data)
#     print("Data normalized into DataFrame.")

#     # Clean the Data
#     print("Cleaning data...")
#     # Placeholder for data cleaning steps: removing duplicates, handling missing data, etc.
#     print("Data cleaned.")

#     # Feature Extraction
#     print("Extracting features...")
#     if 'description.text' in df.columns:
#         # Convert 'description.text' to TF-IDF features
#         tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#         tfidf_features = tfidf_vectorizer.fit_transform(df['description.text'])
#         # Joining original DataFrame with TF-IDF features
#         df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=["tfidf_" + str(i) for i in range(tfidf_features.shape[1])])
#         df = pd.concat([df.drop(['description.text'], axis=1), df_tfidf], axis=1)
#         print("TF-IDF features extracted and added.")
#     else:
#         print("'description.text' column not found, skipping TF-IDF feature extraction.")

#     # One-hot encode selected categorical variables
#     for column in ['occupation_field', 'industry']:
#         if column in df.columns:
#             df = pd.get_dummies(df, columns=[column])
#             print(f"Categorical variable '{column}' one-hot encoded.")
#         else:
#             print(f"'{column}' column not found. Skipping one-hot encoding for '{column}'.")

#     # Preparing data for machine learning
#     print("Splitting data into training and test sets...")
#     if 'is_tech' in df.columns:
#         X = df.drop('is_tech', axis=1)  # Isolate features
#         y = df['is_tech']  # Define target variable
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print("Data split complete.")
#     else:
#         print("Data split skipped due to missing 'is_tech' column.")

#     # Save the preprocessed dataset for future use
#     output_file_path = file_path.replace('.jsonl', '_preprocessed.csv')
#     df.to_csv(output_file_path, index=False)
#     print(f"Preprocessed dataset saved successfully at {output_file_path}")

#     print("Preprocessing complete. The script is ready for machine learning modeling.")
# except FileNotFoundError as e:
#     print(f"File not found error: {e}")
# except json.JSONDecodeError as e:
#     print(f"JSON decode error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## ny test kod swe stopwords FUNKAR EJ

# import pandas as pd
# import json
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# def flatten_data(item):
#     """ Flatten any lists and convert all data to strings. """
#     if isinstance(item, list):
#         return ' '.join(flatten_data(x) for x in item)
#     return str(item)

# try:
#     print("Reading the .jsonl file path from the text file...")
#     text_file_with_path = r'C:\Systemvetenskap\Python\Molkor First Fork\MPTJB\AI\Jsonl_file_path.txt'
#     with open(text_file_with_path, 'r', encoding='utf-8') as f:
#         file_path = f.readline().strip()

#     if not os.path.exists(file_path):
#         print(f"Error: No file found at {file_path}")
#         exit()

#     print(f".jsonl file path read successfully: {file_path}")

#     print("Loading data from .jsonl file...")
#     data = []
#     counter = 0
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#             counter += 1
#             if counter % 10000 == 0:
#                 print(f"Processed {counter} lines...")
#     print(f"Data loaded successfully. Total lines processed: {counter}")

#     df = pd.json_normalize(data)
#     df = df.applymap(flatten_data)  # Ensure all data is flattened and stringified

#     print("Data normalized into DataFrame.")

#     # Analyzing null values
#     print("Analyzing null values in columns...")
#     null_percentage = df.isnull().mean() * 100
#     print("Percentage of null values per column:")
#     print(null_percentage)

#     # Set a threshold for dropping columns with too many null values, e.g., 50%
#     threshold = 50
#     columns_to_drop = null_percentage[null_percentage > threshold].index
#     df.drop(columns=columns_to_drop, inplace=True)
#     print(f"Columns with more than {threshold}% null values dropped.")

#     print("Remaining columns:")
#     print(df.columns)

#     print("Cleaning data...")
#     df.drop_duplicates(inplace=True)
#     print("Data cleaned.")

#     print("Extracting features...")
#     swedish_stopwords = stopwords.words('swedish')

#     # Combine text columns into one for TF-IDF
#     text_columns = [
#         'description.text'
#     ]
#     # Ensure columns exist before attempting to concatenate their contents
#     existing_text_columns = [col for col in text_columns if col in df.columns]
#     df['combined_text'] = df[existing_text_columns].fillna('').agg(' '.join, axis=1)

#     tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words=swedish_stopwords)
#     tfidf_features = tfidf_vectorizer.fit_transform(df['combined_text'])
#     df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#     df = pd.concat([df.drop(existing_text_columns + ['combined_text'], axis=1), df_tfidf], axis=1)
#     print("TF-IDF features extracted and added.")

#     print("Splitting data into training and test sets...")
#     if 'is_tech' in df.columns:
#         X = df.drop('is_tech', axis=1)
#         y = df['is_tech']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print("Data split complete.")
#     else:
#         print("Data split skipped due to missing 'is_tech' column.")

#     output_file_path = file_path.replace('.jsonl', '_preprocessed.csv')
#     df.to_csv(output_file_path, index=False)
#     print(f"Preprocessed dataset saved successfully at {output_file_path}")

#     print("Preprocessing complete. The script is ready for machine learning modeling.")
# except FileNotFoundError as e:
#     print(f"File not found error: {e}")
# except json.JSONDecodeError as e:
#     print(f"JSON decode error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SUBSET KOD SOM SKAPAR EN FIL


# import pandas as pd
# import json
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')



# def flatten_json(y):
#     """ Flatten nested JSON data into a flat dictionary. """
#     out = {}

#     def flatten(x, name=''):
#         if type(x) is dict:
#             for a in x:
#                 flatten(x[a], name + a + '_')
#         elif type(x) is list:
#             i = 0
#             for a in x:
#                 flatten(a, name + str(i) + '_')
#                 i += 1
#         else:
#             out[name[:-1]] = x

#     flatten(y)
#     return out

# try:
#     print("Reading the .jsonl file path from the text file...")
#     text_file_with_path = r'C:\Systemvetenskap\Python\Molkor First Fork\MPTJB\AI\Jsonl_file_path.txt'
#     with open(text_file_with_path, 'r', encoding='utf-8') as f:
#         file_path = f.readline().strip()

#     if not os.path.exists(file_path):
#         print(f"Error: No file found at {file_path}")
#         exit()

#     print(f".jsonl file path read successfully: {file_path}")

#     print("Loading data from .jsonl file...")
#     data = []
#     counter = 0
#     subset_size = 1000  # Adjust this number to change the subset size
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             json_data = json.loads(line)
#             flat_data = flatten_json(json_data)
#             data.append(flat_data)
#             counter += 1
#             if counter % 1000 == 0:
#                 print(f"Processed {counter} lines...")
#             if counter >= subset_size:
#                 break
#     print(f"Data loaded successfully. Total lines processed: {counter}")

#     df = pd.DataFrame(data)

#     print("Data normalized into DataFrame.")

#     # Analyzing null values
#     print("Analyzing null values in columns...")
#     null_percentage = df.isnull().mean() * 100
#     print("Percentage of null values per column:")
#     print(null_percentage)

#     # Set a threshold for dropping columns with too many null values, e.g., 50%
#     threshold = 50
#     columns_to_drop = null_percentage[null_percentage > threshold].index
#     df.drop(columns=columns_to_drop, inplace=True)
#     print(f"Columns with more than {threshold}% null values dropped.")

#     print("Remaining columns:")
#     print(df.columns)

#     print("Cleaning data...")
#     df.drop_duplicates(inplace=True)
#     print("Data cleaned.")

#     print("Extracting features...")
#     swedish_stopwords = stopwords.words('swedish')
#     text_columns = ['description_text']  # Update as per your actual text column names
#     existing_text_columns = [col for col in text_columns if col in df.columns]
#     df['combined_text'] = df[existing_text_columns].fillna('').agg(' '.join, axis=1)

#     tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words=swedish_stopwords)
#     tfidf_features = tfidf_vectorizer.fit_transform(df['combined_text'])
#     tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
#     df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_feature_names)
#     df = pd.concat([df.drop(existing_text_columns + ['combined_text'], axis=1), df_tfidf], axis=1)
#     print("TF-IDF features extracted and added.")

#     print("Splitting data into training and test sets...")
#     if 'is_tech' in df.columns:
#         X = df.drop('is_tech', axis=1)
#         y = df['is_tech']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print("Data split complete.")
#     else:
#         print("Data split skipped due to missing 'is_tech' column.")

#     output_file_path = file_path.replace('.jsonl', '_preprocessed_subset.csv')
#     df.to_csv(output_file_path, index=False)
#     print(f"Preprocessed dataset saved successfully at {output_file_path}")

#     print("Preprocessing complete. The script is ready for machine learning modeling.")
# except FileNotFoundError as e:
#     print(f"File not found error: {e}")
# except json.JSONDecodeError as e:
#     print(f"JSON decode error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")

