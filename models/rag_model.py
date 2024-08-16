from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

def setup_rag_model():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
    
    return tokenizer, retriever, model

if __name__ == "__main__":
    tokenizer, retriever, model = setup_rag_model()
    print("RAG model setup complete.")
