import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(file_path):
    """CSV dosyasından veri çerçevesi yükleme."""
    return pd.read_csv(file_path)

def vectorize_data(df, model):
    """
    Log veri çerçevesini vektörlere dönüştürür.
    
    :param df: Veri çerçevesi (DataFrame)
    :param model: Vektörleştirme modeli (SentenceTransformer)
    :return: Vektörler (numpy array)
    """
    vectors = model.encode(df['Request'].tolist())
    return vectors

def create_faiss_index(vectors):
    """
    Vektörleri kullanarak FAISS index oluşturur.
    
    :param vectors: Vektörler (numpy array)
    :return: FAISS index
    """
    dimension = vectors.shape[1]  # Vektör boyutunu al
    index = faiss.IndexFlatL2(dimension)  # Vektör boyutunu burada kullanın
    index.add(vectors.astype('float32'))  # Vektörleri ekleyin
    return index

def search_in_faiss_index(index, query_vector):
    """
    FAISS index içinde sorgu yapar.
    
    :param index: FAISS index
    :param query_vector: Sorgu vektörü (numpy array)
    :return: Mesafeler ve indeksler
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)  # 2D array'e dönüştür
    D, I = index.search(query_vector, k=5)
    return D, I

def save_index(index, file_path):
    """FAISS index'i diske kaydeder."""
    faiss.write_index(index, file_path)
