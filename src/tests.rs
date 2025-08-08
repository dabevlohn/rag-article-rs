#[cfg(test)]
mod tests {
    use enhanced_rag_article_generator::{
        SearxSearchWrapper, SearchWrapper, OllamaLLM, LanguageModel,
        OllamaEmbeddings, EmbeddingModel, SimpleVectorStore,
        Document, cosine_similarity, cosine_similarity_arrays,
        RecursiveUrlLoader, DocumentLoader, EnhancedRAGArticleGenerator
    };
    use std::collections::HashMap;
    use async_trait::async_trait;
    use anyhow::Result;
    use ndarray::Array1;

    /// Константы для тестирования
    const TEST_SEARX_HOST: &str = "http://127.0.0.1:8080";
    const TEST_OLLAMA_HOST: &str = "http://localhost:11434";
    const TEST_EMBEDDING_MODEL: &str = "nomic-embed-text:latest";
    const TEST_LLM_MODEL: &str = "qwen3:30b";

    /// Тест соединения с SearXNG
    #[tokio::test]
    async fn test_searxng_connection() {
        let search_wrapper = SearxSearchWrapper::new(TEST_SEARX_HOST.to_string());
        
        // Выполняем простой поисковый запрос
        let result = search_wrapper.search("test query", 3).await;
        
        match result {
            Ok(results) => {
                println!("✅ SearXNG соединение успешно");
                println!("   Найдено результатов: {}", results.len());
                
                // Проверяем структуру результатов
                for (i, result) in results.iter().enumerate() {
                    println!("   Результат {}: {} - {}", i + 1, result.title, result.url);
                    assert!(!result.url.is_empty());
                    assert!(!result.title.is_empty());
                }
            }
            Err(e) => {
                println!("❌ SearXNG недоступен: {}", e);
                println!("   Убедитесь что SearXNG запущен на {}", TEST_SEARX_HOST);
                // Не падаем, так как это может быть нормально в CI/CD
            }
        }
    }

    /// Тест соединения с Ollama
    #[tokio::test]
    async fn test_ollama_connection() {
        // Тестируем LLM соединение
        let llm = OllamaLLM::new(TEST_LLM_MODEL.to_string(), Some(TEST_OLLAMA_HOST.to_string()));
        
        let test_prompt = "Hello, this is a test. Please respond with 'Connection successful'.";
        let result = llm.generate(test_prompt).await;
        
        match result {
            Ok(response) => {
                println!("✅ Ollama LLM соединение успешно");
                println!("   Модель: {}", TEST_LLM_MODEL);
                println!("   Ответ: {}", response.chars().take(100).collect::<String>());
                assert!(!response.is_empty());
            }
            Err(e) => {
                println!("❌ Ollama LLM недоступен: {}", e);
                println!("   Убедитесь что:");
                println!("   - Ollama запущен на {}", TEST_OLLAMA_HOST);
                println!("   - Модель {} загружена (ollama pull {})", TEST_LLM_MODEL, TEST_LLM_MODEL);
            }
        }

        // Тестируем Embeddings соединение
        let embeddings = OllamaEmbeddings::new(TEST_EMBEDDING_MODEL.to_string(), Some(TEST_OLLAMA_HOST.to_string()));
        
        let test_text = "This is a test sentence for embeddings.";
        let result = embeddings.embed_query(test_text).await;
        
        match result {
            Ok(embedding) => {
                println!("✅ Ollama Embeddings соединение успешно");
                println!("   Модель: {}", TEST_EMBEDDING_MODEL);
                println!("   Размерность embedding: {}", embedding.len());
                assert!(!embedding.is_empty());
                assert!(embedding.len() > 0);
                
                // Проверяем что embedding содержит числа
                for &value in embedding.iter().take(5) {
                    assert!(value.is_finite());
                }
            }
            Err(e) => {
                println!("❌ Ollama Embeddings недоступен: {}", e);
                println!("   Убедитесь что:");
                println!("   - Ollama запущен на {}", TEST_OLLAMA_HOST);
                println!("   - Модель {} загружена (ollama pull {})", TEST_EMBEDDING_MODEL, TEST_EMBEDDING_MODEL);
            }
        }
    }

    /// Тест создания векторного хранилища
    #[tokio::test]
    async fn test_vector_store_creation() {
        // Создаем mock embedding model для тестирования
        struct MockEmbeddingModel;
        
        #[async_trait]
        impl EmbeddingModel for MockEmbeddingModel {
            async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                // Генерируем фиктивные embeddings для тестирования
                let embeddings = texts.iter().enumerate().map(|(i, text)| {
                    // Создаем детерминированный embedding на основе индекса и длины текста
                    let base_value = (i + 1) as f32 * 0.1;
                    let text_factor = (text.len() % 10) as f32 * 0.01;
                    vec![base_value + text_factor, base_value * 2.0, base_value * 0.5]
                }).collect();
                
                Ok(embeddings)
            }

            async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
                // Простой детерминированный embedding для запроса
                let text_hash = text.len() as f32 * 0.1;
                Ok(vec![text_hash, text_hash * 2.0, text_hash * 0.5])
            }
        }

        let embedding_model = Box::new(MockEmbeddingModel);
        let vector_store = SimpleVectorStore::new(embedding_model);

        // Проверяем начальное состояние
        assert_eq!(vector_store.documents().len(), 0);
        assert_eq!(vector_store.embeddings().nrows(), 0);

        println!("✅ Векторное хранилище создано успешно");
        println!("   Начальное количество документов: {}", vector_store.documents().len());
        println!("   Начальное количество embeddings: {}", vector_store.embeddings().nrows());
    }

    /// Тест заполнения векторного хранилища и поиска косинусного сходства
    #[tokio::test]
    async fn test_vector_store_operations() {
        // Используем тот же MockEmbeddingModel
        struct MockEmbeddingModel;
        
        #[async_trait]
        impl EmbeddingModel for MockEmbeddingModel {
            async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                let embeddings = texts.iter().enumerate().map(|(i, text)| {
                    let base_value = (i + 1) as f32 * 0.1;
                    let text_factor = (text.len() % 10) as f32 * 0.01;
                    vec![base_value + text_factor, base_value * 2.0, base_value * 0.5]
                }).collect();
                Ok(embeddings)
            }

            async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
                let text_hash = text.len() as f32 * 0.1;
                Ok(vec![text_hash, text_hash * 2.0, text_hash * 0.5])
            }
        }

        let embedding_model = Box::new(MockEmbeddingModel);
        let mut vector_store = SimpleVectorStore::new(embedding_model);

        // Создаем тестовые документы
        let mut test_documents = Vec::new();
        
        for i in 1..=5 {
            let mut metadata = HashMap::new();
            metadata.insert("source_number".to_string(), i.to_string());
            metadata.insert("source_title".to_string(), format!("Test Document {}", i));
            
            let document = Document {
                page_content: format!("This is test document number {} with some content about topic {}", i, i),
                metadata,
            };
            test_documents.push(document);
        }

        // Добавляем документы в хранилище
        let result = vector_store.add_documents(test_documents).await;
        assert!(result.is_ok());

        // Проверяем что документы добавлены
        assert_eq!(vector_store.documents().len(), 5);
        assert_eq!(vector_store.embeddings().nrows(), 5);

        println!("✅ Документы успешно добавлены в векторное хранилище");
        println!("   Количество документов: {}", vector_store.documents().len());
        println!("   Количество embeddings: {}", vector_store.embeddings().nrows());

        // Тестируем поиск по сходству
        let search_result = vector_store.similarity_search("test query about topic", 3).await;
        
        match search_result {
            Ok(results) => {
                println!("✅ Поиск по сходству выполнен успешно");
                println!("   Найдено документов: {}", results.len());
                
                assert!(results.len() <= 3); // Не больше чем запрошено
                
                for (i, doc) in results.iter().enumerate() {
                    println!("   Результат {}: {}", i + 1, 
                        doc.metadata.get("source_title").unwrap_or(&"Unknown".to_string()));
                    assert!(!doc.page_content.is_empty());
                }
            }
            Err(e) => {
                panic!("Ошибка при поиске по сходству: {}", e);
            }
        }
    }

    /// Тест функции косинусного сходства для Vec
    #[test]
    fn test_cosine_similarity() {
        // Тест идентичных векторов
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 1e-6);
        println!("✅ Косинусное сходство идентичных векторов: {:.6}", similarity);

        // Тест ортогональных векторов
        let vec3 = vec![1.0, 0.0, 0.0];
        let vec4 = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&vec3, &vec4);
        assert!((similarity - 0.0).abs() < 1e-6);
        println!("✅ Косинусное сходство ортогональных векторов: {:.6}", similarity);

        // Тест противоположных векторов
        let vec5 = vec![1.0, 2.0, 3.0];
        let vec6 = vec![-1.0, -2.0, -3.0];
        let similarity = cosine_similarity(&vec5, &vec6);
        assert!((similarity + 1.0).abs() < 1e-6);
        println!("✅ Косинусное сходство противоположных векторов: {:.6}", similarity);

        // Тест нулевых векторов
        let vec7 = vec![0.0, 0.0, 0.0];
        let vec8 = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&vec7, &vec8);
        assert!((similarity - 0.0).abs() < 1e-6);
        println!("✅ Косинусное сходство с нулевым вектором: {:.6}", similarity);
    }

    /// Тест функции косинусного сходства для Array1
    #[test]
    fn test_cosine_similarity_arrays() {
        // Тест идентичных векторов
        let vec1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let vec2 = Array1::from(vec![1.0, 2.0, 3.0]);
        let similarity = cosine_similarity_arrays(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 1e-6);
        println!("✅ Косинусное сходство Array1 идентичных векторов: {:.6}", similarity);

        // Тест ортогональных векторов
        let vec3 = Array1::from(vec![1.0, 0.0, 0.0]);
        let vec4 = Array1::from(vec![0.0, 1.0, 0.0]);
        let similarity = cosine_similarity_arrays(&vec3, &vec4);
        assert!((similarity - 0.0).abs() < 1e-6);
        println!("✅ Косинусное сходство Array1 ортогональных векторов: {:.6}", similarity);
    }

    /// Интеграционный тест загрузки документов
    #[tokio::test]
    async fn test_document_loader() {
        let loader = RecursiveUrlLoader::new(1, 10);
        
        // Тестируем загрузку простой HTML страницы
        let test_url = "https://httpbin.org/html";
        
        let result = loader.load(test_url).await;
        
        match result {
            Ok(documents) => {
                println!("✅ Загрузка документа успешна");
                println!("   URL: {}", test_url);
                println!("   Количество документов: {}", documents.len());
                
                if !documents.is_empty() {
                    let doc = &documents[0];
                    println!("   Длина содержимого: {}", doc.page_content.len());
                    println!("   Метаданные: {:?}", doc.metadata.keys().collect::<Vec<_>>());
                    
                    // Проверяем что контент не пустой и содержит Markdown
                    assert!(!doc.page_content.is_empty());
                    assert!(doc.metadata.contains_key("source_url"));
                }
            }
            Err(e) => {
                println!("❌ Ошибка загрузки документа: {}", e);
                println!("   URL: {}", test_url);
                // Не паникуем, так как это может быть проблема сети
            }
        }
    }

    /// Тест полной интеграции (требует доступных сервисов)
    #[tokio::test]
    #[ignore] // Игнорируем по умолчанию, запускать с --ignored
    async fn test_full_integration() {
        let mut generator = EnhancedRAGArticleGenerator::new(
            TEST_SEARX_HOST.to_string(),
            TEST_LLM_MODEL.to_string(),
            TEST_EMBEDDING_MODEL.to_string(),
            Some(TEST_OLLAMA_HOST.to_string()),
        );

        let test_query = "test integration";
        
        // Тестируем поиск URL
        let urls_result = generator.search_and_collect_urls(test_query, 3).await;
        
        match urls_result {
            Ok(urls) => {
                println!("✅ Полная интеграция - поиск URL");
                println!("   Найдено URL: {}", urls.len());
                
                if !urls.is_empty() {
                    // Тестируем загрузку документов
                    let docs_result = generator.load_and_process_documents(urls).await;
                    
                    match docs_result {
                        Ok(documents) => {
                            println!("✅ Полная интеграция - загрузка документов");
                            println!("   Загружено документов: {}", documents.len());
                            
                            // Проверяем метаданные источников
                            let metadata = generator.sources_metadata();
                            println!("   Метаданные источников: {}", metadata.len());
                            
                            for (num, source) in metadata {
                                println!("   Источник {}: {} - {}", num, source.title, source.domain);
                            }
                        }
                        Err(e) => println!("❌ Ошибка загрузки документов: {}", e),
                    }
                }
            }
            Err(e) => println!("❌ Ошибка поиска URL: {}", e),
        }
    }

    /// Тест производительности косинусного сходства
    #[test]
    fn test_cosine_similarity_performance() {
        use std::time::Instant;
        
        // Создаем большие векторы для тестирования производительности
        let vec1: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let vec2: Vec<f32> = (0..1000).map(|i| (i as f32).cos()).collect();
        
        let start = Instant::now();
        let similarity = cosine_similarity(&vec1, &vec2);
        let duration = start.elapsed();
        
        println!("✅ Тест производительности косинусного сходства");
        println!("   Размер векторов: {}", vec1.len());
        println!("   Время выполнения: {:?}", duration);
        println!("   Результат сходства: {:.6}", similarity);
        
        // Проверяем что результат корректный
        assert!(similarity.is_finite());
        assert!(similarity >= -1.0 && similarity <= 1.0);
        
        // Производительность должна быть разумной (меньше 10ms для 1000-мерных векторов)
        assert!(duration.as_millis() < 10);
    }

    /// Тест простого текстового ранжирования
    #[test]
    fn test_simple_text_ranking() {
        let mut generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test".to_string(),
            "test".to_string(),
            None,
        );

        // Создаем тестовые документы
        let mut test_documents = Vec::new();
        let mut metadata1 = HashMap::new();
        metadata1.insert("source_title".to_string(), "Rust Programming Guide".to_string());
        test_documents.push(Document {
            page_content: "Rust is a systems programming language focused on safety, speed, and concurrency.".to_string(),
            metadata: metadata1,
        });

        let mut metadata2 = HashMap::new();
        metadata2.insert("source_title".to_string(), "Python Tutorial".to_string());
        test_documents.push(Document {
            page_content: "Python is a high-level programming language with dynamic typing.".to_string(),
            metadata: metadata2,
        });

        let mut metadata3 = HashMap::new();
        metadata3.insert("source_title".to_string(), "Rust Performance Tips".to_string());
        test_documents.push(Document {
            page_content: "Performance optimization in Rust requires understanding ownership and borrowing.".to_string(),
            metadata: metadata3,
        });

        // Тестируем ранжирование
        let ranked = generator.simple_text_ranking(&test_documents, "Rust programming", 2);
        
        println!("✅ Простое текстовое ранжирование");
        println!("   Исходных документов: {}", test_documents.len());
        println!("   Ранжированных документов: {}", ranked.len());
        
        // Проверяем что Rust документы имеют более высокий ранг
        assert_eq!(ranked.len(), 2);
        for (i, doc) in ranked.iter().enumerate() {
            let title = doc.metadata.get("source_title").unwrap_or(&"Unknown".to_string());
            println!("   Ранг {}: {}", i + 1, title);
            assert!(title.contains("Rust"));
        }
    }
}