from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from utils.preprocessing_utils import clean_data
from utils.vector_store import load_data, vectorize_data, create_faiss_index, save_index
from models.rag_model import RAGModel
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Yolu ve isimleri ayarlayın
RAW_CSV_FILE_PATH = 'data/logfiles.csv'
CLEANED_CSV_FILE_PATH = 'data/cleaned_logfiles.csv'
FAISS_INDEX_PATH = 'data/faiss_index.idx'

# Model ve veri hazırlığı
def initialize_model():
    # Dosyaları kontrol et
    if not os.path.exists(CLEANED_CSV_FILE_PATH) or not os.path.exists(FAISS_INDEX_PATH):
        print("Veri oluşturuluyor...")
        from scripts.generate_logs import generate_synthetic_data
        generate_synthetic_data(1000).to_csv(RAW_CSV_FILE_PATH, index=False)

        print("Veri temizleniyor...")
        df = pd.read_csv(RAW_CSV_FILE_PATH)
        df_cleaned = clean_data(df)
        df_cleaned.to_csv(CLEANED_CSV_FILE_PATH, index=False)
        print(f"Temizlenmiş veri '{CLEANED_CSV_FILE_PATH}' konumuna kaydedildi.")

        # SentenceTransformer Modelini Yükleme
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Vektörleştirme ve FAISS İndeksleme
        print("Vektörleştirme ve FAISS index oluşturuluyor...")
        vectors = vectorize_data(df_cleaned, model)
        print(f"Vektör boyutu: {vectors.shape[1]}")
        index = create_faiss_index(vectors)
        save_index(index, FAISS_INDEX_PATH)
        print(f"FAISS index '{FAISS_INDEX_PATH}' konumuna kaydedildi.")
    else:
        print("Veri ve indeks dosyaları mevcut. Yeniden oluşturma yapılmayacak.")

    # RAG Modelini Başlatma
    print("RAG modelini başlatıyor...")
    rag_model = RAGModel(
        retrieval_index_path=FAISS_INDEX_PATH,
        generation_model_name='gpt2',
        log_file_path=CLEANED_CSV_FILE_PATH
    )
    return rag_model

# RAG modelini başlat
rag_model = initialize_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    if question:
        try:
            response = rag_model.answer_question(question)
            return jsonify({'answer': response})
        except Exception as e:
            return jsonify({'error': f'Bir hata oluştu: {str(e)}'})
    else:
        return jsonify({'error': 'Soru boş olamaz!'})

if __name__ == '__main__':
    app.run(debug=True)
