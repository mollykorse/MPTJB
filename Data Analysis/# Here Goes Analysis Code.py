# Here Goes Analysis Code

# This code picks out 10 most common headlines related to 'sustainability' or 'hållbarhet'


# import pandas as pd
# import re  # For regular expression operations


# import os

# correct_path = 'C:\\Systemvetenskap\\Python\\Molkor First Fork\\MPTJB\\Data Analysis'

# # Change the current working directory to the script's folder
# os.chdir(correct_path)

# def load_and_filter_data(csv_path):
#     print("Loading data from CSV file...")
#     data = pd.read_csv(csv_path, low_memory=False)
#     print("Data loaded successfully.")
    
#     # Filter data to find mentions of 'sustainability' or 'hållbarhet' in the 'description.text' column
#     filtered_data = data[data['description.text'].str.contains('sustainability|hållbarhet', case=False, na=False)]
#     return filtered_data

# def find_common_headlines(filtered_data):
#     if not filtered_data.empty:
#         # Filter out headlines that are likely to be dates (pattern: YYYY-MM-DDTHH:MM:SS)
#         filtered_data = filtered_data[~filtered_data['headline'].str.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$')]
        
#         # Count occurrences of each headline
#         headline_counts = filtered_data['headline'].value_counts()
#         return headline_counts.head(10)  # Return top 10 most common headlines
#     else:
#         print("No entries mention 'sustainability' or 'hållbarhet'.")
#         return pd.Series()  # Return empty series if no data

# def main():
#     csv_path = '2023.csv'
#     filtered_data = load_and_filter_data(csv_path)
#     common_headlines = find_common_headlines(filtered_data)
#     print("Most common headlines for entries related to sustainability or hållbarhet (excluding dates):")
#     print(common_headlines)

# if __name__ == "__main__":
#     main()




## OUTPUT:

# Loading data from CSV file...
# Data loaded successfully.
# Most common headlines for entries related to sustainability or hållbarhet (excluding dates):
# Servicerådgivare sökes till Mobility Motors - Sätra         81
# Redovisningsekonom till Johanson Design i Markaryd          78
# 24500:- plus provision                                      69
# Kreativ och driven säljare till Bisevo                      66
# 🌟 Få Ditt Karriärlyft hos ABC SALES för Göta Energi! 🌟      60
# Systemutvecklare till innovativt IT-företag                 49
# Senior cyber security engineer till innovativt techbolag    49
# Elektronikkonstruktör till tekniskt konsultföretag          49
# E-mobility lead till kreativt techbolag                     49
# Embedded utvecklare till kreativt techbolag                 49
# Name: headline, dtype: int64









# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------



## How top 5 keywords for Testers different from UX designers?

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# import pandas as pd
# from nltk.corpus import stopwords as nltk_stopwords
# from nltk.tokenize import word_tokenize
# from nltk.probability import FreqDist
# from stop_words import get_stop_words

# import os

# # The correct path based on the screenshot you've shared
# correct_path = 'C:\\Systemvetenskap\\Python\\Molkor First Fork\\MPTJB\\Data Analysis'

# # Change the current working directory to the script's folder
# os.chdir(correct_path)

# # Your code to load the CSV should follow here


# def load_and_filter_data(csv_path, keywords):
#     df = pd.read_csv(csv_path)
#     pattern = '|'.join([f"\\b{k}\\b" for k in keywords])  # Regex for whole word match
#     return df[df['headline'].str.contains(pattern, case=False, na=False)]


# def extract_top_keywords(texts, top_n=5):
#     english_stopwords = set(nltk_stopwords.words('english'))
#     swedish_stopwords = set(get_stop_words('swedish'))
#     all_stopwords = english_stopwords.union(swedish_stopwords)

#     words = word_tokenize(texts.str.cat(sep=' ').lower())
#     filtered_words = [word for word in words if word.isalnum() and word not in all_stopwords]
#     freq_dist = FreqDist(filtered_words)
#     return [word for word, _ in freq_dist.most_common(top_n)]


# def main(csv_file):
#     tester_keywords = ['tester', 'QA', 'test engineer']
#     ux_keywords = ['UX designer', 'user experience', 'UI/UX']

#     tester_df = load_and_filter_data(csv_file, tester_keywords)
#     ux_df = load_and_filter_data(csv_file, ux_keywords)

#     tester_top_keywords = extract_top_keywords(tester_df['description.text'])
#     ux_top_keywords = extract_top_keywords(ux_df['description.text'])

#     print("Top 5 Keywords for Testers:", tester_top_keywords)
#     print("Top 5 Keywords for UX Designers:", ux_top_keywords)


# main('2023.csv')










# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## In which context do the keywords “hållbarhet” vs. “sustainability” appear? 

# import csv
# import re
# import os

# # Print the current working directory
# print("Current Working Directory:", os.getcwd())

# # Change the working directory
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')

# # Verify the change
# print("New Working Directory:", os.getcwd())

# # Path to the CSV file
# file_path = '2023.csv'

# # Compiled regular expressions for each phrase for efficiency
# pattern_hallbarhet = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?hållbarhet(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)
# pattern_sustainability = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?sustainability(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)

# # Function to process rows and capture context
# def process_rows(reader):
#     results_hallbarhet = []
#     results_sustainability = []
#     for row in reader:
#         # Process each regex and store results up to 10 instances
#         for pattern, result_list, label in [
#             (pattern_hallbarhet, results_hallbarhet, "Hållbarhet"),
#             (pattern_sustainability, results_sustainability, "Sustainability")
#         ]:
#             if len(result_list) < 10:
#                 for match in pattern.finditer(row['description.text']):
#                     before = match.group(1) or ""
#                     after = match.group(2) or ""
#                     result_list.append(f"Context ({label}): {before.strip()} {label.lower()} {after.strip()}")
#                     if len(result_list) == 10:
#                         break

#             if all(len(lst) == 10 for lst in [results_hallbarhet, results_sustainability]):
#                 break

#     return results_hallbarhet, results_sustainability

# # Main function to open the file and process the CSV
# def main():
#     with open(file_path, newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         results_hallbarhet, results_sustainability = process_rows(reader)

#     # Print results for each term
#     print("\nHållbarhet Contexts:")
#     for result in results_hallbarhet:
#         print(result)
#     print("\nSustainability Contexts:")
#     for result in results_sustainability:
#         print(result)

# if __name__ == "__main__":
#     main()









# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # How homogeneous are descriptions in a small town (Marstrand) and a larger town (Borås)?



import os
import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Change the working directory
os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')

# Verify the change
print("New Working Directory:", os.getcwd())

# Path to the CSV file
file_path = '2023.csv'

# Load the data
df = pd.read_csv('2023.csv')

# Filter rows based on municipalities
filtered_df = df[df['workplace_address.municipality'].isin(['Borås', 'Kungälv'])]

# Process each municipality separately
municipalities = ['Borås', 'Kungälv']
results = {}

for city in municipalities:
    # Subset data for the city
    subset = filtered_df[filtered_df['workplace_address.municipality'] == city]
    
    # Vectorize the text descriptions
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(subset['description.text'])
    
    # Calculate cosine similarity matrix
    cos_sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Calculate average cosine similarity for the city
    # We avoid self-comparison by subtracting 1 from the count and setting the diagonal to 0
    n = cos_sim_matrix.shape[0]
    if n > 1:
        np.fill_diagonal(cos_sim_matrix, 0)
        avg_cos_sim = cos_sim_matrix.sum() / (n * (n - 1))
    else:
        avg_cos_sim = None  # Not enough data to calculate similarity

    results[city] = avg_cos_sim

# Output the results
print("Average Cosine Similarity by Municipality:")
for city, sim in results.items():
    print(f"{city}: {sim if sim is not None else 'Insufficient data'}")




# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#########################



########################


######   ANALYSIS v.2


# import os
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [
#     'specific', 'general', 'terms', 'common', 'examples', 'including', 'etc', 'och', 'att', 'du', 'är', 'vi', 'med','som', 'en', 'för', 'av', 'ett', 'https', 'har'
#     'på', 'kommer'
#     # Add any additional domain-specific stopwords
# ]

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')  # Adjust language as necessary
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load data
# file_path = '2023.csv'
# df = pd.read_csv(file_path, low_memory=False)

# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = ['sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig']
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].copy()

# # Process text for analysis, integrating all stopwords
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sustainable_df['description.text'])

# # Apply LDA for topic modeling
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# topics = lda.fit_transform(tfidf_matrix)

# # Output topics
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for topic_idx, topic in enumerate(lda.components_):
#     print(f"Topic {topic_idx + 1}:")
#     print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# # Sentiment Analysis: use .loc to ensure modifications are reflected in the DataFrame copy
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# # Displaying results or further analysis here...

# # Analyze frequency of mentions and visualize
# sustainability_count_by_city = df.groupby('workplace_address.municipality')['mentions_sustainability'].sum()
# sns.barplot(x=sustainability_count_by_city.index, y=sustainability_count_by_city.values)
# plt.title('Frequency of Sustainability Mentions by Municipality')
# plt.ylabel('Count')
# plt.xlabel('Municipality')
# plt.xticks(rotation=45)
# plt.show()











# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#########################



########################


######   ANALYSIS v.3


# import os
# import re
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [
#     'specific', 'general', 'terms', 'common', 'examples', 'including', 'etc',
#     'och', 'att', 'du', 'är', 'vi', 'med', 'som', 'en', 'för', 'av', 'ett',
#     'https', 'har', 'på', 'kommer', 'dig', 'oss', 'till', 'vara', 'samt', 
#     'eftersom', 'vår', 'jobbet', 'vill', 'egen', 'jag', 'work', 'den', 'jobba', 
#     'jobbet', 'within', 'våra','inom', 'det', 'din', 'söker', 'kan', 'om', 'alla', 
#     'kunna', 'då', 'ser', 'din', 'får', 'kan', 'working', 'experience', 'us', 'utan', 
#     'role', 'både', 'arbetar', 'hos', 'eller', 'arbeta', 'tillsammans', 'ditt', 'efter',
# ]

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')  # Adjust language as necessary
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load data
# file_path = '2023.csv'
# df = pd.read_csv(file_path, low_memory=False)

# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = ['sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig']
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].copy()

# # Process text for analysis, integrating all stopwords
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sustainable_df['description.text'])

# # Apply LDA for topic modeling
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# topics = lda.fit_transform(tfidf_matrix)

# # Output topics
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for topic_idx, topic in enumerate(lda.components_):
#     print(f"Topic {topic_idx + 1}:")
#     print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# # Sentiment Analysis: use .loc to ensure modifications are reflected in the DataFrame copy
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# # Analyze frequency of mentions and visualize
# sustainability_count_by_city = df.groupby('workplace_address.municipality')['mentions_sustainability'].sum()
# sns.barplot(x=sustainability_count_by_city.index, y=sustainability_count_by_city.values)
# plt.title('Frequency of Sustainability Mentions by Municipality')
# plt.ylabel('Count')
# plt.xlabel('Municipality')
# plt.xticks(rotation=45)
# plt.show()

# # Compiled regular expressions for keyword contexts
# pattern_hallbarhet = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?hållbarhet(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)
# pattern_sustainability = re.compile(r'(\b\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s)?sustainability(\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\b)?', re.IGNORECASE)

# # Extract context for keywords and limit to 10 instances for each
# results_hallbarhet, results_sustainability = [], []
# for description in df['description.text'].dropna():
#     if len(results_hallbarhet) < 10:
#         matches = pattern_hallbarhet.findall(description)
#         results_hallbarhet.extend(matches[:10-len(results_hallbarhet)])
#     if len(results_sustainability) < 10:
#         matches = pattern_sustainability.findall(description)
#         results_sustainability.extend(matches[:10-len(results_sustainability)])
#     if len(results_hallbarhet) >= 10 and len(results_sustainability) >= 10:
#         break

# print("\nHållbarhet Contexts:")
# for result in results_hallbarhet:
#     print(result)
# print("\nSustainability Contexts:")
# for result in results_sustainability:
#     print(result)









# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#########################



########################


######   ANALYSIS v.4



# import os
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm


# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [
#     'specific', 'general', 'terms', 'common', 'examples', 'including', 'etc',
#     'och', 'att', 'du', 'är', 'vi', 'med', 'som', 'en', 'för', 'av', 'ett',
#     'https', 'har', 'på', 'kommer', 'dig', 'oss', 'till', 'vara', 'samt', 
#     'eftersom', 'vår', 'jobbet', 'vill', 'egen', 'jag', 'work', 'den', 'jobba', 
#     'jobbet', 'within', 'våra', 'inom', 'det', 'din', 'söker', 'kan', 'om', 'alla', 
#     'kunna', 'då', 'ser', 'din', 'får', 'kan', 'working', 'experience', 'us', 'utan', 
#     'role', 'både', 'arbetar', 'hos', 'eller', 'arbeta', 'tillsammans', 'ditt', 'efter',
#     'arbete', 'tiden', 'fått', 'air', 'schema', 'god', 'åker', 'inget', 'finns', 'mycket', 
#     'något', 'där', 'genom', 'så', 'från', 'dina', 'uppdrag', 'annat', 'ska', 'vid', 'stad',
#     'även', 'rollen', 'mer', 'erbjuder', 'de', 'arbetet', 'medarbetare', 'fast', 'del', 'vårt',
#     'inte', 'lön', 'plus', 'solid', 'ta', 'andra', 'olika', 'se', 'sky', 
# ]

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())
# os.chdir('c:/Systemvetenskap/Python/Molkor First Fork/MPTJB/Data Analysis')
# print("New Working Directory:", os.getcwd())

# # Load data
# file_path = '2023.csv'
# df = pd.read_csv(file_path, low_memory=False)

# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = [
#     'sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig', 'clean energy', 'green economy',
#     'sustainable tech', 'technology', 'biodegradable', 'sustainable production', 'green technology', 'carbon neutral', 'circular economy', 'hållbar',
#     'hållbar teknologi', 'hållbar produktion',
#     ]
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].copy()

# # Process text for analysis, integrating all stopwords
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sustainable_df['description.text'])

# # Apply LDA for topic modeling
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# topics = lda.fit_transform(tfidf_matrix)

# # Output topics
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for topic_idx, topic in enumerate(lda.components_):
#     print(f"Topic {topic_idx + 1}:")
#     print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))


# # Sentiment Analysis
# print("Analyzing sentiments...")
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# # Print out the summary statistics of the sentiment scores
# print("Sentiment Analysis Summary:")
# print(sustainable_df['sentiment'].describe())

# # Visualizing sentiment distribution
# plt.figure(figsize=(10, 6))
# plt.hist(sustainable_df['sentiment'], bins=30, color='blue', alpha=0.7)
# plt.title('Distribution of Sentiment Scores')
# plt.xlabel('Sentiment Polarity')
# plt.ylabel('Frequency')
# plt.show()
