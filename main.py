# main.py
import pandas as pd
from utils.preprocessing_utils import clean_data
from utils.vector_store import load_data, vectorize_data, create_faiss_index, save_index
from models.rag_model import RAGModel
from sentence_transformers import SentenceTransformer

raw_csv_file_path = 'data/logfiles.csv'
cleaned_csv_file_path = 'data/cleaned_logfiles.csv'
faiss_index_path = 'data/faiss_index.idx'

def main():
    try:
        # Veri Üretimi
        print("Veri oluşturuluyor...")
        from scripts.generate_logs import generate_synthetic_data
        generate_synthetic_data(1000).to_csv(raw_csv_file_path, index=False)
        
        # Veri Temizleme
        print("Veri temizleniyor...")
        df = pd.read_csv(raw_csv_file_path)
        df_cleaned = clean_data(df)
        df_cleaned.to_csv(cleaned_csv_file_path, index=False)
        print(f"Temizlenmiş veri '{cleaned_csv_file_path}' konumuna kaydedildi.")

        # SentenceTransformer Modelini Yükleme
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Vektörleştirme ve FAISS İndeksleme
        print("Vektörleştirme ve FAISS index oluşturuluyor...")
        vectors = vectorize_data(df_cleaned, model)
        print(f"Vektör boyutu: {vectors.shape[1]}")
        index = create_faiss_index(vectors)
        save_index(index, faiss_index_path)
        print(f"FAISS index '{faiss_index_path}' konumuna kaydedildi.")

        # RAG Modelini Başlatma
        print("RAG modelini başlatıyor...")
        rag_model = RAGModel(
            retrieval_index_path=faiss_index_path,
            generation_model_name='gpt2',
            log_file_path=cleaned_csv_file_path
        )

        # Soru Cevaplama
        question = "User ile ilgili detaylar nedir?"
        print(f"Soru: {question}")
        answer = rag_model.answer_question(question)
        print(f"Cevap: {answer}")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
