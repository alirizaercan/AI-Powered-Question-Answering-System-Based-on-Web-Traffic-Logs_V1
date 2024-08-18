#scripts/test_system.py

import os
import sys

# Proje kök dizinini PYTHONPATH'a ekleyin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.rag_model import RAGModel

def test_rag_model():
    # Model dosyalarının yollarını buraya yerleştirin
    retrieval_index_path = "../data/faiss_index.idx"
    log_file_path = "../data/logfiles.csv"
    rag_model = RAGModel(retrieval_index_path, generation_model_name='gpt2', log_file_path=log_file_path)

    # Test soruları
    questions = [
        "What is the response time for the API request?",
        "How many bytes were transferred in the last log entry?",
        "Which IP addresses have errors in their requests?"
    ]

    for question in questions:
        print(f"Question: {question}")
        answer = rag_model.answer_question(question)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    test_rag_model()
