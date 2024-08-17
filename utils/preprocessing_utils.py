import pandas as pd
import re

def clean_data(df):
    """
    Gereksiz sütunları düşürür ve veriyi temizler.

    :param df: Temizlenecek veri çerçevesi
    :return: Temizlenmiş veri çerçevesi
    """
    # Sadece gerekli sütunları seç
    required_columns = ['IP', 'Timestamp', 'Request', 'Status_Code', 'Bytes', 'Response_Time', 'Request_Type']
    df = df[required_columns]

    # 'Request', 'Status_Code', 'Bytes', ve 'Response_Time' sütunlarında eksik verileri temizleme
    df = df.dropna(subset=['Request', 'Status_Code', 'Bytes', 'Response_Time'])

    # 'Request' sütununda temizlik işlemleri
    df['Request'] = df['Request'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df = df[df['Request'].str.strip() != '']

    # 'Bytes' ve 'Response_Time' sütunlarındaki veri türlerini kontrol et ve dönüştür
    df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce')  # Eksik değerleri NaN yapar
    df['Response_Time'] = pd.to_numeric(df['Response_Time'], errors='coerce')  # Eksik değerleri NaN yapar

    # Eksik 'Bytes' ve 'Response_Time' değerlerini temizle
    df = df.dropna(subset=['Bytes', 'Response_Time'])

    return df