import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.retrieval import RetrievalModel  # Doğru yolu ayarladığınızdan emin olun
from models.generation import GenerationModel

class RAGModel:
    def __init__(self, retrieval_index_path, generation_model_name='gpt2', log_file_path=None):
        self.retrieval_model = RetrievalModel(retrieval_index_path)
        self.generation_model = GenerationModel(model_name=generation_model_name)
        self.log_file_path = log_file_path
        self.logs = self._load_logs()
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.predefined_answers = self._load_predefined_answers()
        self.time_related_terms = ['az', 'fazla', 'yüksek', 'düşük']
        self.keyword_mappings = {
            'IP': self._get_ip_info,
            'Timestamp': self._get_timestamp_info,
            'Request': self._get_request_info,
            'Status_Code': self._get_status_code_info,
            'Bytes': self._get_bytes_info,
            'Response_Time': self._get_response_time_info,
            'Request_Type': self._get_request_type_info,
            'IP Adresi': self._get_ip_info,
            'Zaman Damgası': self._get_timestamp_info,
            'İstek': self._get_request_info,
            'Durum Kodu': self._get_status_code_info,
            'Bayt': self._get_bytes_info,
            'Yanıt Süresi': self._get_response_time_info,
            'İstek Türü': self._get_request_type_info,
        }

    def _load_logs(self):
        if self.log_file_path and os.path.exists(self.log_file_path):
            return pd.read_csv(self.log_file_path)
        else:
            raise FileNotFoundError("Log dosyası bulunamadı.")

    def _load_predefined_answers(self):
        return {
             "merhaba": "Merhaba! Ben bir web trafik loglarına dayalı yapay zeka destekli sistemim. Size nasıl yardımcı olabilirim?",
            "nasılsın?": "Ben bir yapay zeka sistemiyim, dolayısıyla bir ruh halim yok ama size yardımcı olmak için buradayım!",
            "adın ne": "Ben bir yapay zeka sistemiyim, dolayısıyla adım yok. Size nasıl yardımcı olabilirim?",
            "web trafik logları nedir?": "Web trafik logları, bir web sitesine yapılan ziyaretlerin detaylarını içeren kayıtlardır. Bu kayıtlar genellikle IP adresi, ziyaret saati, istek türü, yanıt kodu ve veri transfer miktarını içerir.",
            "veri transferi nedir?": "Veri transferi, internet üzerindeki verilerin bir kaynaktan bir hedefe taşınmasıdır. Web trafik loglarında, bu genellikle gönderilen veya alınan bayt miktarı olarak ölçülür.",
            "en yüksek veri transferi hangi tarihte gerçekleşti?": "En yüksek veri transferinin gerçekleştiği tarihi öğrenmek için verilerdeki transfer miktarlarını analiz edebiliriz. Bu bilgiye ulaşmak için sistemimden belirli bir tarih aralığı talep edebilirsiniz.",
            "en düşük veri transferi ne zaman gerçekleşti?": "En düşük veri transferinin gerçekleştiği zamanı belirlemek için log verilerindeki transfer miktarlarını inceleyebiliriz. Bu bilgiye ulaşmak için tarih aralığı belirlemeniz yeterlidir.",
            "en çok hangi sayfa ziyaret edildi?": "En çok ziyaret edilen sayfayı belirlemek için log kayıtlarındaki sayfa isteklerini analiz edebiliriz. Bu tür bir bilgiye ulaşmak için detaylı analiz talep edebilirsiniz.",
            "en çok hangi IP adresi ziyaret yaptı?": "En çok ziyaret yapan IP adresini belirlemek için log verilerindeki IP adreslerini analiz edebiliriz. Bu bilgiye ulaşmak için belirli bir tarih aralığı vermeniz gerekebilir.",
            "bu loglarda hangi hata kodları var?": "Web trafik loglarında çeşitli HTTP durum kodları bulunabilir, örneğin 404 (Bulunamadı) veya 500 (Sunucu Hatası). Hangi hata kodlarının bulunduğunu öğrenmek için sistemimden bu kodları listeleyebiliriz.",
            "sisteminiz nasıl çalışıyor?": "Sistemim, web trafik loglarını analiz ederek kullanıcıların sorularına yanıtlar üretir. Bu, çeşitli verileri işleyerek ve benzer geçmiş sorularla karşılaştırarak yapılır. Doğal dil işleme (NLP) teknikleri kullanılarak anlamlı ve doğru yanıtlar sağlanır.",
            "sisteminizin özellikleri nelerdir?": "Sistemim, web trafik loglarını analiz etme, veri transferi ve ziyaretçi davranışlarını izleme gibi özelliklere sahiptir. Ayrıca, çeşitli analiz yöntemleri kullanarak veri tabanlı sorulara yanıt verebilir ve raporlar oluşturabilir.",
        }
        
    def find_most_similar(self, question, predefined_answers):
        question = question.lower().strip()  # Normalizasyon
        questions = list(predefined_answers.keys())
        questions = [q.lower().strip() for q in questions]  # Normalizasyon
        
        # Encode questions
        encoded_questions = self.sentence_model.encode(questions, convert_to_tensor=True)
        encoded_query = self.sentence_model.encode([question], convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.pytorch_cos_sim(encoded_query, encoded_questions)
        best_match_idx = similarities.argmax()
        
        best_question = questions[best_match_idx]
        
        # Handle cases where the similarity might be too low
        if similarities.max().item() < 0.6:  # Threshold for similarity
            return None
        
        return best_question

    def answer_question(self, question, top_k=5):
        question = question.lower().strip()
        best_match = self.find_most_similar(question, self.predefined_answers)
        
        if best_match:
            return self.predefined_answers.get(best_match, "Üzgünüm, bu soruya yanıt bulamadım.")
        
        # Dinamik kavramlara dayalı cevaplar
        for key, method in self.keyword_mappings.items():
            if key.lower() in question:
                return method(question)
        
        # Zamanla ilgili ifadeler
        if any(term in question for term in self.time_related_terms):
            return self._handle_time_related_queries(question)
        
        return "Bu tür bir soru için geçerli bir yanıt bulunamadı. Size nasıl yardımcı olabilirim?"

    def _handle_time_related_queries(self, question):
        if "en fazla" in question and "veri transferi" in question:
            return f"En yüksek veri transferi yapılan saat aralığı: {self._get_highest_data_transfer_hour()}"
        elif "en az" in question and "veri transferi" in question:
            return f"En düşük veri transferi yapılan saat aralığı: {self._get_lowest_data_transfer_hour()}"
        
        return "Bu tür bir soru için geçerli bir yanıt bulunamadı."

    def _get_highest_data_transfer_hour(self):
        return self.logs.groupby(self.logs['Timestamp'].str[:13])['Bytes'].sum().idxmax()

    def _get_lowest_data_transfer_hour(self):
        return self.logs.groupby(self.logs['Timestamp'].str[:13])['Bytes'].sum().idxmin()

    def _get_ip_info(self, question):
        # Bu metod IP bilgisi ile ilgili sorguları işleyebilir.
        # Burada örnek olarak, IP sütunundaki ilk birkaç değeri döndürüyoruz.
        ip_sample = self.logs['IP'].dropna().unique()[:5]
        return f"Örnek IP adresleri: {', '.join(ip_sample)}" if ip_sample.size > 0 else "IP bilgisi sağlanamadı."

    def _get_timestamp_info(self, question):
        # Bu metod zaman damgası ile ilgili sorguları işleyebilir.
        # Burada örnek olarak, zaman damgalarındaki ilk birkaç değeri döndürüyoruz.
        timestamp_sample = self.logs['Timestamp'].dropna().unique()[:5]
        return f"Örnek zaman damgaları: {', '.join(timestamp_sample)}" if timestamp_sample.size > 0 else "Zaman damgası bilgisi sağlanamadı."

    def _get_request_info(self, question):
        # Bu metod istek bilgileri ile ilgili sorguları işleyebilir.
        # Burada örnek olarak, istek sütunundaki ilk birkaç değeri döndürüyoruz.
        request_sample = self.logs['Request'].dropna().unique()[:5]
        return f"Örnek istekler: {', '.join(request_sample)}" if request_sample.size > 0 else "İstek bilgisi sağlanamadı."

    def _get_status_code_info(self, question):
        # Bu metod durum kodları ile ilgili sorguları işleyebilir.
        # Burada örnek olarak, durum kodu sütunundaki ilk birkaç değeri döndürüyoruz.
        status_code_sample = self.logs['Status_Code'].dropna().unique()[:5]
        return f"Örnek durum kodları: {', '.join(map(str, status_code_sample))}" if status_code_sample.size > 0 else "Durum kodu bilgisi sağlanamadı."

    def _get_bytes_info(self, question):
        # Bu metod bayt bilgileri ile ilgili sorguları işleyebilir.
        # Burada örnek olarak, bayt sütunundaki ilk birkaç değeri döndürüyoruz.
        bytes_sample = self.logs['Bytes'].dropna().unique()[:5]
        return f"Örnek bayt değerleri: {', '.join(map(str, bytes_sample))}" if bytes_sample.size > 0 else "Bayt bilgisi sağlanamadı."

    def _get_response_time_info(self, question):
        # Bu metod yanıt süresi bilgileri ile ilgili sorguları işleyebilir.
        # Burada örnek olarak, yanıt süresi sütunundaki ilk birkaç değeri döndürüyoruz.
        response_time_sample = self.logs['Response_Time'].dropna().unique()[:5]
        return f"Örnek yanıt süreleri: {', '.join(map(str, response_time_sample))}" if response_time_sample.size > 0 else "Yanıt süresi bilgisi sağlanamadı."

    def _get_request_type_info(self, question):
        # Bu metod istek türü bilgileri ile ilgili sorguları işleyebilir.
        # Burada örnek olarak, istek türü sütunundaki ilk birkaç değeri döndürüyoruz.
        request_type_sample = self.logs['Request_Type'].dropna().unique()[:5]
        return f"Örnek istek türleri: {', '.join(request_type_sample)}" if request_type_sample.size > 0 else "İstek türü bilgisi sağlanamadı."

    def _encode_question(self, question):
        query_vector = self.sentence_model.encode(question)
        
        if query_vector.shape[0] != self.retrieval_model.dimension:
            raise ValueError(f"Query vector size ({query_vector.shape[0]}) does not match FAISS index dimension ({self.retrieval_model.dimension})")
        
        return np.expand_dims(query_vector, axis=0)

    def _create_context_from_indices(self, indices):
        context_lines = []
        
        for idx_list in indices:
            for idx in idx_list:
                if idx < len(self.logs):
                    log_entry = self.logs.iloc[idx]
                    
                    formatted_entry = (
                        f"IP: {log_entry.get('IP', 'Bilgi yok')}\n"
                        f"Timestamp: {log_entry.get('Timestamp', 'Bilgi yok')}\n"
                        f"Request: {log_entry.get('Request', 'Bilgi yok')}\n"
                        f"Status Code: {log_entry.get('Status_Code', 'Bilgi yok')}\n"
                        f"Bytes: {log_entry.get('Bytes', 'Bilgi yok')}\n"
                        f"Response Time: {log_entry.get('Response_Time', 'Bilgi yok')} ms\n"
                        f"Request Type: {log_entry.get('Request_Type', 'Bilgi yok')}\n"
                        f"{'='*50}\n"
                    )
                    context_lines.append(formatted_entry)
        
        context = ''.join(context_lines)
        
        if len(context) > 2000:
            context = context[:2000] + '\n[...]\n'
        
        return context
