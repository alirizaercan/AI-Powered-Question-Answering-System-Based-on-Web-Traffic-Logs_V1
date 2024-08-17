import pandas as pd
import numpy as np

def generate_synthetic_data(num_records):
    """
    Sentetik log verilerini üretir.

    :param num_records: Üretilecek kayıt sayısı
    :return: Pandas DataFrame
    """
    np.random.seed(42)  # Tekrar edilebilir sonuçlar için
    data = {
        'IP': np.random.choice(['192.168.1.1', '192.168.1.2', '192.168.1.3'], num_records),
        'Timestamp': pd.date_range(start='2024-01-01', periods=num_records, freq='T').astype(str),
        'Request': np.random.choice(['/home', '/login', '/signup', '/logout'], num_records),
        'Status_Code': np.random.choice([200, 404, 500], num_records),
        'Bytes': np.random.randint(100, 5000, num_records),
        'Response_Time': np.random.uniform(0.1, 5.0, num_records),
        'Request_Type': np.random.choice(['GET', 'POST', 'PUT', 'DELETE'], num_records)
    }
    return pd.DataFrame(data)

def main():
    num_records = 1000  # Üretilecek veri sayısı
    df = generate_synthetic_data(num_records)
    df.to_csv('data/logfiles.csv', index=False)
    print(f"Sentetik veri '{num_records}' kayıt üretildi ve 'data/logfiles.csv' konumuna kaydedildi.")

if __name__ == "__main__":
    main()
