# Here Goes Analysis Code

# This code picks out 10 most common headlines related to 'sustainability' or 'hållbarhet'

import pandas as pd
import re  # For regular expression operations

def load_and_filter_data(csv_path):
    print("Loading data from CSV file...")
    data = pd.read_csv(csv_path, low_memory=False)
    print("Data loaded successfully.")
    
    # Filter data to find mentions of 'sustainability' or 'hållbarhet' in the 'description.text' column
    filtered_data = data[data['description.text'].str.contains('sustainability|hållbarhet', case=False, na=False)]
    return filtered_data

def find_common_headlines(filtered_data):
    if not filtered_data.empty:
        # Filter out headlines that are likely to be dates (pattern: YYYY-MM-DDTHH:MM:SS)
        filtered_data = filtered_data[~filtered_data['headline'].str.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$')]
        
        # Count occurrences of each headline
        headline_counts = filtered_data['headline'].value_counts()
        return headline_counts.head(10)  # Return top 10 most common headlines
    else:
        print("No entries mention 'sustainability' or 'hållbarhet'.")
        return pd.Series()  # Return empty series if no data

def main():
    csv_path = '2023.csv'
    filtered_data = load_and_filter_data(csv_path)
    common_headlines = find_common_headlines(filtered_data)
    print("Most common headlines for entries related to sustainability or hållbarhet (excluding dates):")
    print(common_headlines)

if __name__ == "__main__":
    main()


