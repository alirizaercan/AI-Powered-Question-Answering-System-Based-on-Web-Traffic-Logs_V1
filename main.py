from data.data_preprocessing import preprocess_log_file
from models.rag_model import setup_rag_model
from utils.vector_store import VectorStore

def main():
    df = preprocess_log_file('data/logfiles.log')
    
    vector_store = VectorStore(df)
    
    tokenizer, retriever, model = setup_rag_model()
    
    print("Proje başlatıldı ve sistem hazır.")
    
if __name__ == "__main__":
    main()
