# models/generation.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GenerationModel:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def preprocess_text(self, text):
        # Metni küçük harfe çevir ve temizle
        return text.lower()

    def generate_response(self, context):
        context = self.preprocess_text(context)
        inputs = self.tokenizer.encode(context, return_tensors='pt')
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Attention mask oluştur
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=500,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id  # pad_token_id olarak eos_token_id kullanımı
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
