from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import time

# WebDriver'ı ayarlayın
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Tarayıcıyı arka planda çalıştırma, bu satırı yorum satırı haline getirin
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ChromeDriver'ı başlatın
service = Service('../chromedriver.exe')  # ChromeDriver yolunu buraya girin
driver = webdriver.Chrome(service=service, options=chrome_options)

# Test soruları
test_questions = [
    # Genel Sorular
    "Web trafiği nedir?",
    "Veri analizi hakkında bilgi verir misiniz?",
    "Sistem performansını nasıl ölçebilirim?",
    "İnternet güvenliği hakkında neler biliyorsunuz?",
    "Veri yönetimi ile ilgili en iyi uygulamalar nelerdir?",

    # Veri ile İlgili Sorular
    "En sık rastlanan saat dilimi nedir?",
    "Status_Code sütununda kaç farklı değer var?",
    "Son 24 saatte gelen isteklerin sayısı nedir?",
    "IP adreslerinin dağılımını nasıl analiz edebilirim?",
    "Web isteklerinin yanıt süreleri hakkında bilgi verir misiniz?",

    # Sahte veya Bilinmeyen Sorular
    "sahte_soru",
    "bilmiyorum",
    "yardimci_olamam",
    "Bana Python hakkında bilgi verir misiniz?",
    "Nasılsınız?",
    
    
    # IP
    "En sık rastlanan IP adresleri nelerdir ve bu IP'lerin trafik hacmi nasıl analiz edilir?",
    
    # Timestamp
    "Trafik yoğunluğu hangi zaman dilimlerinde artış göstermektedir ve bu trend nasıl analiz edilir?",
    
    # Request
    "En sık yapılan HTTP istek türleri nelerdir ve bu türlerin trafik üzerindeki etkileri nasıl değerlendirilir?",
    
    # Status_Code
    "En sık rastlanan durum kodları nelerdir ve bu kodların analiz edilmesi ne tür sorunları gösterebilir?",
    
    # Bytes
    "En yüksek veri transferi yapılan istekler hangileridir ve bu verilerin analizi nasıl yapılır?",
    
    # Response_Time
    "En uzun ve en kısa yanıt süreleri nedir ve bu sürelerin analiz edilmesi nasıl yapılır?",
    
    # Request_Type
    "Farklı istek türlerinin yanıt süreleri nasıl karşılaştırılır ve bu karşılaştırma ne tür bilgiler sağlar?",

    # Bilinmeyen Sorular
    "Gelecekteki hava durumu nedir?",
    "Türkiye'nin başkenti neresi?",
    "En iyi pizza tarifi nedir?",
    "Hangi renk en popüler?",
    "Yapay zeka gelecekte nereye gidecek?"
]

try:
    # Flask uygulamanızın yerel adresine gidin
    driver.get("http://127.0.0.1:5000")

    for question in test_questions:
        # Soru giriş alanını bulun ve soruyu girin
        question_input = driver.find_element(By.ID, "question")
        question_input.clear()  # Önceki soruları temizle
        question_input.send_keys(question)

        # ENTER tuşuna basın
        question_input.send_keys(Keys.RETURN)

        # Yükleniyor mesajının kaybolmasını bekleyin
        WebDriverWait(driver, 15).until(
            EC.invisibility_of_element((By.ID, "loading"))
        )

        # Yanıtın mevcut olduğunu kontrol edin
        chat_container = driver.find_element(By.ID, "chat-container")
        messages = chat_container.find_elements(By.CLASS_NAME, "chat-message")

        # Yanıtları yazdırma
        for msg in messages:
            print(f"{datetime.now()}: {msg.text}")

        # Bir sonraki test sorusu için kısa bir bekleme süresi
        time.sleep(2)

finally:
    # Tarayıcıyı kapatın
    driver.quit()
