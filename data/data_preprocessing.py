# data/data_preprocessing.py

import pandas as pd
from utils.preprocessing_utils import clean_data

# Log verilerini yükleyin
csv_file_path = 'data/logfiles.csv'
df = pd.read_csv(csv_file_path)

# Veriyi temizleyin
df_cleaned = clean_data(df)

# Temizlenmiş veriyi kaydedin
cleaned_csv_file_path = 'data/cleaned_logfiles.csv'
df_cleaned.to_csv(cleaned_csv_file_path, index=False)

print(f"Temizlenmiş veri '{cleaned_csv_file_path}' konumuna kaydedildi.")
