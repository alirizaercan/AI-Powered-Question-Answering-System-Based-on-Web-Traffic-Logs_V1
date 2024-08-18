[TR] 

# Web Trafik Loglarına Dayalı Yapay Zeka Destekli Soru-Cevap Sistemi

Bu proje, web trafik loglarını kullanarak yapay zeka destekli bir soru-cevap (Q&A) sistemi geliştirmeyi amaçlamaktadır. Sistem, kullanıcıların doğal dilde sordukları soruları alır, log verilerini analiz eder ve uygun yanıtları oluşturur. Bu sistem, Retrieval-Augmented Generation (RAG) modeline dayanmaktadır ve Flask kullanılarak bir web uygulaması olarak yapılandırılmıştır.

## Proje Yapısı

Bu proje aşağıdaki ana bileşenleri içerir:

### Dosya ve Dizinler

- **app.py**: Flask tabanlı web uygulamasının ana dosyasıdır. Kullanıcıların web üzerinden soru sormasını sağlar ve yanıtları geri döndürür.
- **main.py**: Proje başlangıç noktasıdır ve uygulamanın başlatılmasını sağlar.
- **requirements.txt**: Proje için gerekli olan Python kütüphanelerini listeleyen dosyadır.
- **test_main.py**: Ana uygulama için birim testlerini içerir.
- **test_scipt_selenium.py**: Selenium kullanarak sistemin entegrasyon testlerini gerçekleştirir.
- **README.md**: Projenizin genel bilgilerini ve nasıl kullanılacağını açıklar.

### Data Dizin Yapısı

- **data/**: Ham ve temizlenmiş log dosyalarını, veri işleme betiklerini ve vektör veritabanı dosyalarını içerir.
  - **cleaned_logfiles.csv**: Temizlenmiş log verilerini içerir.
  - **cleaned_logfiles.zip**: Temizlenmiş log verilerini içeren sıkıştırılmış dosyadır.
  - **data_preprocessing.py**: Log verilerini temizlemek ve yapılandırmak için kullanılan Python betiği.
  - **faiss_index.idx**: FAISS vektör veritabanı dosyası.
  - **logfiles.csv**: Ham log verilerini içerir.

### Models Dizin Yapısı

- **models/**: RAG modelinin bileşenlerini içerir.
  - **generation.py**: Jeneratif model bileşeni, kullanıcıdan gelen sorulara yanıt üretir.
  - **rag_model.py**: Retrieval-Augmented Generation (RAG) modelinin birleşim noktasını sağlar.
  - **retrieval.py**: Bilgi alma bileşeni, en uygun log kayıtlarını arar.
  - **__init__.py**: Model modülünü başlatan dosya.

### Notebooks Dizin Yapısı

- **notebooks/**: Verinin keşfi ve analizi için Jupyter Notebook dosyaları içerir.
  - **exploration.ipynb**: Veri analizi ve keşfi için kullanılan notebook.

### Scripts Dizin Yapısı

- **scripts/**: Sistem ile ilgili ek betikler içerir.
  - **generate_logs.py**: Test ve örnek log verileri oluşturma betiği.
  - **test_system.py**: Sistem testleri için kullanılan betik.

### Templates Dizin Yapısı

- **templates/**: Web uygulaması için HTML şablonlarını içerir.
  - **index.html**: Uygulamanın ana sayfası.

### Utils Dizin Yapısı

- **utils/**: Yardımcı işlevler ve araçlar içerir.
  - **preprocessing_utils.py**: Veriyi işlemek için kullanılan yardımcı işlevler.
  - **vector_store.py**: Vektör veritabanı ile etkileşim sağlayan işlevler.
  - **__init__.py**: Utils modülünü başlatan dosya.

## Kurulum ve Kullanım

1. **Gerekli Kütüphaneleri Kurun**:
   ```bash
   pip install -r requirements.txt

    Uygulamayı Başlatın:
    python app.py

    Uygulamanızı Web Tarayıcınızda Görüntüleyin:
    Tarayıcınızda http://127.0.0.1:5000 adresine gidin.

Katkıda Bulunanlar
Ali Riza Ercan - Proje geliştirme ve entegrasyon.
Lisans
Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına bakabilirsiniz.


[EN]

# AI-Powered Question Answering System Based on Web Traffic Logs

This project aims to develop an AI-powered question-answering (Q&A) system using web traffic logs. The system takes user questions in natural language, analyzes the log data, and generates appropriate responses. It is based on the Retrieval-Augmented Generation (RAG) model and is implemented as a web application using Flask.

## Project Structure

This project includes the following main components:

### Files and Directories

- **app.py**: The main file for the Flask-based web application. It allows users to submit questions through the web and returns the responses.
- **main.py**: The entry point for the project, which initializes the application.
- **requirements.txt**: Lists the Python libraries required for the project.
- **test_main.py**: Contains unit tests for the main application.
- **test_scipt_selenium.py**: Performs integration tests of the system using Selenium.
- **README.md**: Provides an overview of the project and instructions for use.

### Data Directory Structure

- **data/**: Contains raw and cleaned log files, data processing scripts, and vector database files.
  - **cleaned_logfiles.csv**: Contains cleaned log data.
  - **cleaned_logfiles.zip**: A compressed file containing cleaned log data.
  - **data_preprocessing.py**: Python script used for cleaning and structuring log data.
  - **faiss_index.idx**: FAISS vector database file.
  - **logfiles.csv**: Contains raw log data.

### Models Directory Structure

- **models/**: Contains components of the RAG model.
  - **generation.py**: The generative model component that generates responses to user questions.
  - **rag_model.py**: The core RAG model integration point.
  - **retrieval.py**: The retrieval component that searches for the most relevant log entries.
  - **__init__.py**: Initializes the model module.

### Notebooks Directory Structure

- **notebooks/**: Contains Jupyter Notebook files for data exploration and analysis.
  - **exploration.ipynb**: Notebook used for data analysis and exploration.

### Scripts Directory Structure

- **scripts/**: Contains additional scripts related to the system.
  - **generate_logs.py**: Script for generating test and example log data.
  - **test_system.py**: Script used for system testing.

### Templates Directory Structure

- **templates/**: Contains HTML templates for the web application.
  - **index.html**: The main page of the application.

### Utils Directory Structure

- **utils/**: Contains utility functions and tools.
  - **preprocessing_utils.py**: Helper functions used for data processing.
  - **vector_store.py**: Functions for interacting with the vector database.
  - **__init__.py**: Initializes the utils module.

## Installation and Usage

1. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
    Start the Application:
    python app.py

    View Your Application in a Web Browser:
    Navigate to http://127.0.0.1:5000 in your browser.

Contributors
Ali Riza Ercan - Project development and integration.
License
This project is licensed under the MIT License. For more details, see the LICENSE file.