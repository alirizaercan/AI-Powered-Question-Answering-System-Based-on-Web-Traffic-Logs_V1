import faiss
import numpy as np

def load_index(file_path):
    """FAISS index'i dosyadan yükleme."""
    return faiss.read_index(file_path)

def test_index(index, num_queries=5):
    """Index üzerinde test sorguları yapma."""
    # Rastgele vektörler oluşturma (sorgu vektörleri)
    dimension = index.d
    queries = np.random.rand(num_queries, dimension).astype('float32')
    
    # Sorguları index'e gönderme
    distances, indices = index.search(queries, k=5)  # En yakın 5 komşuyu bulma
    
    # Sonuçları yazdırma
    print("Distances:\n", distances)
    print("Indices:\n", indices)

if __name__ == "__main__":
    # Index'i yükleme
    index = load_index('data/faiss_index.idx')
    
    # Test yapma
    test_index(index)
