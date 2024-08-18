#models/retrieval.py

import faiss
import numpy as np

class RetrievalModel:
    def __init__(self, index_path):
        """
        Retrieval modelini başlatır ve FAISS vektör veri tabanını yükler.
        
        :param index_path: FAISS vektör veri tabanının yolu
        """
        self.index_path = index_path
        self.index = faiss.read_index(self.index_path)
        self.dimension = self.index.d

    def search(self, query_vector, top_k=5):
        """
        Sorgu vektörünü kullanarak en yakın komşuları arar.
        
        :param query_vector: Sorgu vektörü
        :param top_k: Döndürülecek en yakın komşu sayısı
        :return: Mesafeler ve indeksler
        """
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Sorgu vektörü boyutu ({query_vector.shape[1]}) FAISS index boyutuna ({self.dimension}) uymuyor.")
        
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices
