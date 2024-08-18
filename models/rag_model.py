#models/rag_model.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from models.retrieval import RetrievalModel
from models.generation import GenerationModel

class RAGModel:
    def __init__(self, retrieval_index_path, generation_model_name='gpt2', log_file_path=None):
        self.retrieval_model = RetrievalModel(retrieval_index_path)
        self.generation_model = GenerationModel(model_name=generation_model_name)
        self.log_file_path = log_file_path
        self.logs = self._load_logs()
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        self.responses = {
            "sahte_soru": "Bu tür bir soru yapay zeka sistemlerine uygun değildir. Lütfen başka bir şey deneyin.",
            "bilmiyorum": "Bu konuda yeterli bilgiye sahip değilim. Başka bir şey sormak ister misiniz?",
            "yardimci_olamam": "Maalesef bu soruyu yanıtlayamıyorum. Yardımcı olabileceğim başka bir konu var mı?"
        }

    def _load_logs(self):
        """
        Log dosyasını yükler.
        
        :return: Log verileri (DataFrame)
        """
        return pd.read_csv(self.log_file_path)

    def answer_question(self, question, top_k=5):
        question_type = self._determine_question_type(question)
        
        if question_type == 'general':
            return self._handle_general_question()
        elif question_type == 'data_related':
            return self._handle_data_related_question(question, top_k)
        elif question_type == 'fake':
            return self._handle_fake_question(question)
        else:
            return self._handle_unknown_question()

    def _encode_question(self, question):
        query_vector = self.sentence_model.encode(question)
        if query_vector.shape[0] != self.retrieval_model.dimension:
            raise ValueError(f"Query vector size ({query_vector.shape[0]}) does not match FAISS index dimension ({self.retrieval_model.dimension})")
        return np.expand_dims(query_vector, axis=0)

    def _create_summary_from_indices(self, indices):
        """
        İndekslerden özet oluşturur.

        :param indices: Sorgulama sonuçları için indeksler
        :return: Özeti oluşturur (str)
        """
        summaries = []
        for idx_list in indices:
            for idx in idx_list:
                log_entry = self.logs.iloc[idx]
                summaries.append(
                    f"IP: {log_entry['IP']}\n"
                    f"Timestamp: {log_entry['Timestamp']}\n"
                    f"Request: {log_entry['Request']}\n"
                    f"Status Code: {log_entry['Status_Code']}\n"
                    f"Bytes: {log_entry['Bytes']}\n"
                    f"Response Time: {log_entry['Response_Time']}\n"
                    f"Request Type: {log_entry['Request_Type']}\n"
                    f"---"
                )
        summary_text = '\n'.join(summaries)
        return summary_text
    
    def _get_most_frequent_time(self):
        """
        En sık rastlanan saati döndürür.
        
        :return: En sık rastlanan saat dilimi (str)
        """
        self.logs['Timestamp'] = pd.to_datetime(self.logs['Timestamp'])
        frequent_time = self.logs['Timestamp'].dt.hour.value_counts().idxmax()
        return f"{frequent_time}:00 - {frequent_time + 1}:00"

    def _get_data_statistics(self, column_name):
        """
        Belirli bir sütun için veri setinde kapsamlı istatistikler sağlar.
        
        :param column_name: Sütun adı (string)
        :return: Veri setinden istatistiksel yanıt (string)
        """
        if column_name not in self.logs.columns:
            return f"Sorgulanan sütun '{column_name}' veri setinde bulunamadı."

        column_data = self.logs[column_name]

        # Benzersiz değerler ve sayıları
        unique_values = column_data.unique()
        unique_count = len(unique_values)

        # En yaygın değeri ve sayısını hesapla
        most_common = column_data.mode().iloc[0]
        most_common_count = column_data.value_counts().iloc[0]

        # Sütun istatistikleri
        if column_data.dtype in [np.int64, np.float64]:
            # Sayısal veriler için özet istatistikler
            mean_value = column_data.mean()
            median_value = column_data.median()
            std_dev = column_data.std()
            min_value = column_data.min()
            max_value = column_data.max()
            missing_count = column_data.isna().sum()
            return (f"'{column_name}' sütununda toplam {unique_count} benzersiz değer bulunmaktadır. "
                    f"En yaygın '{column_name}' değeri: {most_common} ({most_common_count} kez). "
                    f"Ortalama: {mean_value:.2f}, Medyan: {median_value:.2f}, Standart Sapma: {std_dev:.2f}, "
                    f"Minimum: {min_value}, Maksimum: {max_value}, Eksik Değer Sayısı: {missing_count}.")
        elif column_data.dtype == object:
            # Kategorik veriler için özet bilgiler
            top_categories = column_data.value_counts().head(5)
            missing_count = column_data.isna().sum()
            return (f"'{column_name}' sütununda toplam {unique_count} benzersiz değer bulunmaktadır. "
                    f"En yaygın '{column_name}' değeri: {most_common} ({most_common_count} kez). "
                    f"En yaygın 5 kategori: {', '.join([f'{v} ({c} kez)' for v, c in top_categories.items()])}. "
                    f"Eksik Değer Sayısı: {missing_count}.")
        else:
            # Diğer veri türleri için temel bilgi
            missing_count = column_data.isna().sum()
            return (f"'{column_name}' sütununda toplam {unique_count} benzersiz değer bulunmaktadır. "
                    f"En yaygın '{column_name}' değeri: {most_common} ({most_common_count} kez). "
                    f"Eksik Değer Sayısı: {missing_count}.")

    def _determine_question_type(self, question):
        """
        Sorunun türünü belirler.

        :param question: Kullanıcı tarafından sorulan soru (str)
        :return: Sorunun türü (str)
        """
        question = question.lower()
        data_related_keywords = [
            "veri transferi", "IP","en sık", "ne zaman", "zaman dilimi", 
            "ip adresi", "zaman damgası", "istek", "durum kodu", 
            "bayt", "yanıt süresi", "istek türü", "ağ trafiği", 
            "kullanıcı", "sunucu", "işlem süresi", "yanıt kodu", 
            "veri boyutu", "web isteği", "sunucu yanıtı", "hata kodu", 
            "gelen istek", "giden istek", "zaman aralığı", "trafik analizi", 
            "veri paketi", "bant genişliği", "veri akışı", "sistem yükü", 
            "istek süresi", "oturum", "çerez", "oturum süresi", 
            "güvenlik duvarı", "ağ gecikmesi", "performans ölçümü", "veri tabanı", 
            "veri erişimi", "yeni bağlantı", "yanıt kodları", "trafik yükü", 
            "kullanıcı etkileşimi", "veri toplama", "giriş çıkış", "ağ trafiği verisi", 
            "veri günlüğü", "sunucu günlükleri", "analiz raporu", "veri işleme", 
            "trafik paternleri", "anlık veriler", "sunucu yükü", "sistem izleme", 
            "uç nokta", "servis yanıt süresi", "sunucu yapılandırması", "veri güncellemesi", 
            "işlem raporu", "günlük kaydı", "istek detayları", "yanıt detayları", 
            "ağ trafik analizi", "sistem performansı", "işlem hızı", "veri yönetimi", 
            "trafik yoğunluğu", "veri sorgulama", "sunucu performansı", "yanıt süresi ölçümü", 
            "günlük verileri", "istek zamanlaması", "trafik analiz raporu", "ağ performansı", 
            "veri erişim süresi", "çalışma süresi", "ağ bağlantısı", "oturum izleme", 
            "istek analizi", "trafik analiz raporu", "zaman analizi", "veri toplama araçları", 
            "ağ güvenliği", "istek tipi", "sunucu yanıt süresi", "sunucu günlük kaydı", 
            "veri akışı analizi", "yanıt süresi raporu", "istek süresi ölçümü", 
            "veri erişim raporu", "trafik yönetimi", "performans izleme", "ağ trafiği raporu", 
            "çalışma süresi ölçümü", "günlük analiz", "sunucu performans verileri", 
            "web trafiği verileri", "ağ performans ölçümü", "veri analiz araçları", 
            
                        # İngilizce
            "data transfer", "most frequent", "when", "time period", 
            "ip address", "timestamp", "request", "status code", 
            "bytes", "response time", "request type", "network traffic", 
            "user", "server", "processing time", "response code", 
            "data size", "web request", "server response", "error code", 
            "incoming request", "outgoing request", "time range", "traffic analysis", 
            "data packet", "bandwidth", "data flow", "system load", 
            "request duration", "session", "cookie", "session duration", 
            "firewall", "network latency", "performance measurement", "database", 
            "data access", "new connection", "response codes", "traffic load", 
            "user interaction", "data collection", "input output", "network traffic data", 
            "log data", "server logs", "analysis report", "data processing", 
            "traffic patterns", "real-time data", "server load", "system monitoring", 
            "endpoint", "service response time", "server configuration", "data update", 
            "operation report", "log entry", "request details", "response details", 
            "network traffic analysis", "system performance", "operation speed", "data management", 
            "traffic density", "data query", "server performance", "response time measurement", 
            "log data", "request timing", "traffic analytics", "network performance", 
            "data access time", "uptime", "network connection", "session monitoring", 
            "request analysis", "traffic analysis report", "time analysis", "data collection tools", 
            "network security", "request type", "server response time", "server log entry", 
            "data flow analysis", "response time report", "request duration measurement", 
            "data access report", "traffic management", "performance monitoring", "network traffic report", 
            "uptime measurement", "log analysis", "server performance data", 
            "web traffic data", "network performance measurement", "data analysis tools"
        ]
        fake_question_keywords = ["sahte_soru", "bilmiyorum", "yardimci_olamam"]
        
        if any(keyword in question for keyword in data_related_keywords):
            return 'data_related'
        elif any(keyword in question for keyword in fake_question_keywords):
            return 'fake'
        else:
            return 'general'

    def _handle_general_question(self):
        """
        Genel sorular için yanıt döndürür.
        
        :return: Yanıt (str)
        """
        return "Bu konuda size yardımcı olamam. Başka bir şey sormak ister misiniz?"

    def _handle_fake_question(self, question):
        """
        Sahte ve bilinmeyen sorular için yanıt döndürür.
        
        :param question: Kullanıcı tarafından sorulan sahte veya bilinmeyen soru (str)
        :return: Yanıt (str)
        """
        return self.responses.get(question.lower(), self.responses["bilmiyorum"])

    def _handle_data_related_question(self, question, top_k=5):
        """
        Veri ile ilgili sorular için yanıt döndürür.
        
        :param question: Kullanıcı tarafından sorulan veri ile ilgili soru (str)
        :param top_k: En benzer sonuçların sayısı (int)
        :return: Yanıt (str)
        """
        query_vector = self._encode_question(question)
        similar_indices = self.retrieval_model.search(query_vector, top_k)
        summary = self._create_summary_from_indices(similar_indices)
        most_frequent_time = self._get_most_frequent_time()
        status_statistics = self._get_data_statistics('Status_Code')

        response = (
            f"Bilgi isteğinizle ilgili web trafik loglarından elde edilen veriler:\n\n{summary}\n\n"
            f"En sık rastlanan saat dilimi: {most_frequent_time}\n\n"
            f"Durum kodu istatistikleri:\n{status_statistics}\n"
        )
        return response

    def _handle_unknown_question(self):
        """
        Bilinmeyen veya işlenemeyen sorular için yanıt döndürür.
        
        :return: Yanıt (str)
        """
        return "Bu soruya yanıt veremiyorum. Başka bir şey sormak ister misiniz?"
