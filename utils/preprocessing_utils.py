# utils/preprocessing_utils.py

import pandas as pd
import re

def clean_data(df):
    # Gereksiz sütunları kaldır
    df = df.drop(columns=['User_Agent', 'Referrer'])
    
    # IP adreslerini filtrele (örnek bir filtreleme)
    df = df[df['IP'].str.startswith('192.') == False]
    
    # Eksik verileri doldur
    df = df.dropna()
    
    # Diğer temizleme işlemleri (örneğin, gereksiz karakterlerin çıkarılması)
    df['Request'] = df['Request'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    
    return df
