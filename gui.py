import tkinter as tk
from tkinter import scrolledtext
from main import RAGModel  # main.py'den RAGModel sınıfını içe aktaracağız

class QA_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Web Trafik Logları Soru-Cevap Sistemi")

        # Soru giriş alanı
        self.question_label = tk.Label(root, text="Sorunuzu Girin:")
        self.question_label.pack(pady=5)

        self.question_entry = tk.Entry(root, width=50)
        self.question_entry.pack(pady=5)

        # Cevap çıkış alanı
        self.answer_label = tk.Label(root, text="Cevap:")
        self.answer_label.pack(pady=5)

        self.answer_text = scrolledtext.ScrolledText(root, width=60, height=15, wrap=tk.WORD)
        self.answer_text.pack(pady=5)

        # Gönder düğmesi
        self.ask_button = tk.Button(root, text="Soru Sor", command=self.ask_question)
        self.ask_button.pack(pady=10)

        # RAG modelini başlat
        self.rag_model = RAGModel(
            retrieval_index_path='data/faiss_index.idx',
            generation_model_name='gpt2',
            log_file_path='data/cleaned_logfiles.csv'
        )

    def ask_question(self):
        question = self.question_entry.get()
        if question:
            answer = self.rag_model.answer_question(question)
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(tk.END, answer)
        else:
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(tk.END, "Lütfen bir soru girin.")

if __name__ == "__main__":
    root = tk.Tk()
    gui = QA_GUI(root)
    root.mainloop()
