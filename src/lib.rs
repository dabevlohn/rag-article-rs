use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};
use url::Url;
use ndarray::{Array1, Array2};

pub mod cli;

/// Структура для метаданных источника
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub url: String,
    pub title: String,
    pub snippet: String,
    pub domain: String,
    pub content_summary: String,
    pub topics_covered: Vec<String>,
}

/// Структура для результата поиска SearXNG
#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub results: Vec<SearchResultItem>,
}

#[derive(Debug, Deserialize)]
pub struct SearchResultItem {
    pub url: String,
    pub title: String,
    pub content: Option<String>,
}

/// Структура для документа с контентом
#[derive(Debug, Clone)]
pub struct Document {
    pub page_content: String,
    pub metadata: HashMap<String, String>,
}

/// Трейт для работы с поисковыми системами
#[async_trait]
pub trait SearchWrapper {
    async fn search(&self, query: &str, num_results: u32) -> Result<Vec<SearchResultItem>>;
}

/// Реализация SearXNG поиска
pub struct SearxSearchWrapper {
    client: Client,
    host: String,
}

impl SearxSearchWrapper {
    pub fn new(host: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Enhanced RAG Article Generator/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self { client, host }
    }
}

#[async_trait]
impl SearchWrapper for SearxSearchWrapper {
    async fn search(&self, query: &str, num_results: u32) -> Result<Vec<SearchResultItem>> {
        let search_url = format!(
            "{}/search?q={}&format=json&safesearch=0&pageno=1&time_range=year",
            self.host,
            urlencoding::encode(query)
        );

        info!("Выполняем поиск: {}", query);
        
        let response = self.client
            .get(&search_url)
            .send()
            .await?;

        let search_result: SearchResult = response
            .json()
            .await?;

        let mut results = search_result.results;
        results.truncate(num_results as usize);

        info!("Найдено {} результатов", results.len());
        Ok(results)
    }
}

/// Трейт для загрузки документов
#[async_trait]
pub trait DocumentLoader {
    async fn load(&self, url: &str) -> Result<Vec<Document>>;
}

/// Реализация загрузчика документов с поддержкой рекурсивной загрузки
pub struct RecursiveUrlLoader {
    client: Client,
}

impl RecursiveUrlLoader {
    pub fn new(_max_depth: u32, timeout_secs: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("Mozilla/5.0 (compatible; Enhanced-RAG-Bot/1.0)")
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    /// Конвертация HTML в Markdown
    fn convert_html_to_markdown(&self, html_content: &str) -> String {
        // Используем html2text для конвертации
        let markdown = html2text::from_read(html_content.as_bytes(), 80);
        
        // Очистка от лишних переносов строк
        let re_newlines = Regex::new(r"\n{3,}").unwrap();
        let cleaned = re_newlines.replace_all(&markdown, "\n\n");
        
        // Удаление навигационных элементов
        let re_nav = Regex::new(r"(?m)^\s*\*\s*(Home|Menu|Navigation|Skip to|Back to top).*$").unwrap();
        let cleaned = re_nav.replace_all(&cleaned, "");
        
        // Фильтрация коротких строк
        let lines: Vec<&str> = cleaned.lines().collect();
        let filtered_lines: Vec<&str> = lines
            .into_iter()
            .filter(|line| {
                let stripped = line.trim();
                stripped.len() > 10 || stripped.starts_with('#')
            })
            .collect();
        
        filtered_lines.join("\n")
    }
}

#[async_trait]
impl DocumentLoader for RecursiveUrlLoader {
    async fn load(&self, url: &str) -> Result<Vec<Document>> {
        info!("Загрузка документа: {}", url);
        
        let response = self.client
            .get(url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP ошибка: {}", response.status()));
        }

        let html_content = response
            .text()
            .await?;

        // Конвертируем HTML в Markdown
        let markdown_content = self.convert_html_to_markdown(&html_content);

        if markdown_content.trim().len() < 100 {
            warn!("Документ слишком короткий: {}", url);
            return Ok(vec![]);
        }

        let mut metadata = HashMap::new();
        metadata.insert("source_url".to_string(), url.to_string());
        
        // Извлекаем заголовок страницы
        let document = Html::parse_document(&html_content);
        let title_selector = Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|elem| elem.text().collect::<String>())
            .unwrap_or_else(|| "Untitled".to_string());
        
        metadata.insert("source_title".to_string(), title);
        
        // Извлекаем домен
        if let Ok(parsed_url) = Url::parse(url) {
            if let Some(domain) = parsed_url.domain() {
                metadata.insert("source_domain".to_string(), domain.to_string());
            }
        }

        let document = Document {
            page_content: markdown_content,
            metadata,
        };

        Ok(vec![document])
    }
}

/// Трейт для векторных embeddings
#[async_trait]
pub trait EmbeddingModel {
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

/// Простая реализация эмбеддингов через Ollama API
pub struct OllamaEmbeddings {
    client: Client,
    model_name: String,
    ollama_host: String,
}

impl OllamaEmbeddings {
    pub fn new(model_name: String, ollama_host: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            model_name,
            ollama_host: ollama_host.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }
}

#[async_trait]
impl EmbeddingModel for OllamaEmbeddings {
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        
        for text in texts {
            let embedding = self.embed_query(text).await?;
            embeddings.push(embedding);
            
            // Небольшая пауза между запросами
            sleep(Duration::from_millis(100)).await;
        }
        
        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.ollama_host);
        
        let request_body = serde_json::json!({
            "model": self.model_name,
            "prompt": text
        });

        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await?;

        #[derive(Deserialize)]
        struct EmbeddingResponse {
            embedding: Vec<f32>,
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await?;

        Ok(embedding_response.embedding)
    }
}

/// Простое векторное хранилище в памяти с базовой математикой
pub struct SimpleVectorStore {
    documents: Vec<Document>,
    embeddings: Array2<f32>,
    embedding_model: Box<dyn EmbeddingModel + Send + Sync>,
    embedding_dim: usize,
}

impl SimpleVectorStore {
    pub fn new(embedding_model: Box<dyn EmbeddingModel + Send + Sync>) -> Self {
        Self {
            documents: Vec::new(),
            embeddings: Array2::zeros((0, 0)),
            embedding_model,
            embedding_dim: 0,
        }
    }

    pub async fn add_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        info!("Добавление {} документов в векторное хранилище", documents.len());
        
        let texts: Vec<String> = documents
            .iter()
            .map(|doc| doc.page_content.clone())
            .collect();

        let new_embeddings = self.embedding_model.embed_documents(&texts).await?;
        
        if new_embeddings.is_empty() {
            return Ok(());
        }
        
        // Определяем размерность embeddings
        if self.embedding_dim == 0 {
            self.embedding_dim = new_embeddings[0].len();
            self.embeddings = Array2::zeros((0, self.embedding_dim));
        }
        
        // Конвертируем Vec<Vec<f32>> в Array2<f32>
        let new_embeddings_array = Array2::from_shape_vec(
            (new_embeddings.len(), self.embedding_dim),
            new_embeddings.into_iter().flatten().collect(),
        )?;
        
        // Объединяем старые и новые embeddings
        if self.embeddings.nrows() == 0 {
            self.embeddings = new_embeddings_array;
        } else {
            let old_embeddings = self.embeddings.clone();
            self.embeddings = Array2::zeros((
                old_embeddings.nrows() + new_embeddings_array.nrows(), 
                self.embedding_dim
            ));
            
            // Копируем старые embeddings
            self.embeddings
                .slice_mut(ndarray::s![..old_embeddings.nrows(), ..])
                .assign(&old_embeddings);
            
            // Копируем новые embeddings
            self.embeddings
                .slice_mut(ndarray::s![old_embeddings.nrows().., ..])
                .assign(&new_embeddings_array);
        }
        
        self.documents.extend(documents);
        
        info!("Документы успешно добавлены в векторное хранилище");
        Ok(())
    }

    pub async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }

        let query_embedding = self.embedding_model.embed_query(query).await?;
        let query_array = Array1::from(query_embedding);
        
        // Вычисляем косинусное сходство для каждого документа
        let mut similarities = Vec::new();
        
        for (idx, doc_embedding) in self.embeddings.outer_iter().enumerate() {
            let similarity = cosine_similarity_arrays(&query_array, &doc_embedding.to_owned());
            similarities.push((idx, similarity));
        }
        
        // Сортируем по убыванию сходства
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Возвращаем топ-k документов
        let results = similarities
            .into_iter()
            .take(k)
            .map(|(idx, _)| self.documents[idx].clone())
            .collect();

        Ok(results)
    }

    /// Геттер для документов (для тестирования)
    pub fn documents(&self) -> &Vec<Document> {
        &self.documents
    }

    /// Геттер для embeddings (для тестирования)
    pub fn embeddings(&self) -> &Array2<f32> {
        &self.embeddings
    }
}

/// Вычисление косинусного сходства для Array1
pub fn cosine_similarity_arrays(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Вычисление косинусного сходства для Vec (для обратной совместимости)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let array_a = Array1::from(a.to_vec());
    let array_b = Array1::from(b.to_vec());
    cosine_similarity_arrays(&array_a, &array_b)
}

/// Трейт для языковых моделей
#[async_trait]
pub trait LanguageModel {
    async fn generate(&self, prompt: &str) -> Result<String>;
}

/// Реализация для Ollama LLM
pub struct OllamaLLM {
    client: Client,
    model_name: String,
    ollama_host: String,
}

impl OllamaLLM {
    pub fn new(model_name: String, ollama_host: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(3600)) // 5 минут на генерацию
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            model_name,
            ollama_host: ollama_host.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }
}

#[async_trait]
impl LanguageModel for OllamaLLM {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.ollama_host);
        
        let request_body = serde_json::json!({
            "model": self.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        });

        info!("Отправляем запрос к Ollama для генерации текста");
        
        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await?;

        #[derive(Deserialize)]
        struct GenerationResponse {
            response: String,
        }

        let generation_response: GenerationResponse = response
            .json()
            .await?;

        Ok(generation_response.response)
    }
}

/// Основной генератор статей
pub struct EnhancedRAGArticleGenerator {
    search_wrapper: Box<dyn SearchWrapper + Send + Sync>,
    document_loader: Box<dyn DocumentLoader + Send + Sync>,
    language_model: Box<dyn LanguageModel + Send + Sync>,
    sources_metadata: HashMap<u32, SourceMetadata>,
    source_counter: u32,
}

impl EnhancedRAGArticleGenerator {
    pub fn new(
        searx_host: String,
        model_name: String,
        _embedding_model_name: String, // Не используется в упрощенной версии
        ollama_host: Option<String>,
    ) -> Self {
        let search_wrapper = Box::new(SearxSearchWrapper::new(searx_host));
        let document_loader = Box::new(RecursiveUrlLoader::new(2, 20));
        let language_model = Box::new(OllamaLLM::new(model_name, ollama_host));

        Self {
            search_wrapper,
            document_loader,
            language_model,
            sources_metadata: HashMap::new(),
            source_counter: 1,
        }
    }

    pub async fn search_and_collect_urls(&mut self, query: &str, num_results: u32) -> Result<Vec<String>> {
        info!("Поиск URL для запроса: {}", query);
        
        let search_results = self.search_wrapper.search(query, num_results).await?;
        
        let mut urls = Vec::new();
        
        for result in search_results {
            if !urls.contains(&result.url) {
                urls.push(result.url.clone());
                
                // Извлекаем домен
                let domain = Url::parse(&result.url)
                    .ok()
                    .and_then(|url| url.domain().map(|d| d.to_string()))
                    .unwrap_or_else(|| "unknown".to_string());
                
                // Сохраняем метаданные
                let metadata = SourceMetadata {
                    url: result.url,
                    title: result.title,
                    snippet: result.content.unwrap_or_default(),
                    domain,
                    content_summary: String::new(), // Будет заполнено позже
                    topics_covered: Vec::new(),     // Будет заполнено позже
                };
                
                self.sources_metadata.insert(self.source_counter, metadata);
                self.source_counter += 1;
            }
        }
        
        info!("Найдено {} уникальных URL", urls.len());
        Ok(urls)
    }

    pub async fn load_and_process_documents(&mut self, urls: Vec<String>) -> Result<Vec<Document>> {
        info!("Загрузка и обработка {} документов", urls.len());
        
        let mut all_documents = Vec::new();
        
        for (i, url) in urls.iter().enumerate() {
            let source_number = (i + 1) as u32;
            
            match self.document_loader.load(url).await {
                Ok(mut documents) => {
                    if !documents.is_empty() {
                        // Добавляем метаданные источника
                        for doc in &mut documents {
                            doc.metadata.insert("source_number".to_string(), source_number.to_string());
                        }
                        
                        // Обновляем метаданные источника с содержанием
                        let combined_content: String = documents
                            .iter()
                            .map(|doc| doc.page_content.clone())
                            .collect::<Vec<_>>()
                            .join("\n");
                        
                        let (summary, topics) = self.extract_content_summary(&combined_content);
                        
                        if let Some(metadata) = self.sources_metadata.get_mut(&source_number) {
                            metadata.content_summary = summary;
                            metadata.topics_covered = topics;
                        }
                        
                        all_documents.extend(documents);
                        info!("Загружен документ {}/{}: {}", i + 1, urls.len(), url);
                    }
                }
                Err(e) => {
                    warn!("Ошибка при загрузке {}: {}", url, e);
                }
            }
        }
        
        info!("Всего загружено {} документов", all_documents.len());
        Ok(all_documents)
    }

    fn extract_content_summary(&self, content: &str) -> (String, Vec<String>) {
        // Извлекаем заголовки
        let re_headers = Regex::new(r"(?m)^#+\s+(.+)$").unwrap();
        let topics: Vec<String> = re_headers
            .captures_iter(content)
            .map(|cap| cap[1].trim().to_string())
            .take(5)
            .collect();

        // Краткое содержание
        let summary = if content.len() > 300 {
            format!("{}...", &content[..300].replace('\n', " ").trim())
        } else {
            content.replace('\n', " ").trim().to_string()
        };

        (summary, topics)
    }

    pub async fn generate_article_simple(&mut self, query: &str, max_retrieved_docs: usize) -> Result<String> {
        info!("Начинаем генерацию статьи для запроса: {}", query);
        
        // 1. Поиск URL
        let urls = self.search_and_collect_urls(query, 15).await?;
        
        if urls.is_empty() {
            return Ok("Не найдено источников для создания статьи.".to_string());
        }
        
        // 2. Загрузка документов
        let documents = self.load_and_process_documents(urls).await?;
        
        if documents.is_empty() {
            return Ok("Не удалось загрузить документы из найденных источников.".to_string());
        }
        
        // 3. Простое ранжирование по релевантности без векторного поиска
        let retrieved_docs = self.simple_text_ranking(&documents, query, max_retrieved_docs);
        
        // 4. Подготовка контекста
        let context_with_sources = self.prepare_context_with_sources(&retrieved_docs);
        
        // 5. Создание промпта
        let article_prompt = format!(
            "You are an expert academic writer creating a comprehensive research article based on provided context documents.\n\n\
            AVAILABLE SOURCE DOCUMENTS:\n{}\n\n\
            TASK: Write a detailed, well-structured article about: {}\n\n\
            CRITICAL CITATION REQUIREMENTS:\n\
            1. When referencing specific information, data, quotes, or concepts from a source, immediately follow with a citation [X] where X is the source number\n\
            2. Citations must be semantically meaningful - each citation should guide readers to sources containing MORE DETAILED information about that specific topic\n\
            3. Place citations at the end of sentences or paragraphs that contain information from that source\n\
            4. Use multiple citations [1][2] when information is supported by multiple sources\n\
            5. Ensure each citation logically connects the content to the source that elaborates on that topic\n\n\
            WRITING STYLE:\n\
            - Academic but accessible tone\n\
            - Clear section headings (use ##, ###)\n\
            - Logical progression of ideas\n\
            - Incorporate specific technical details, examples, and methodologies from sources\n\
            - Each paragraph should develop a distinct aspect of the topic\n\n\
            STRUCTURE REQUIREMENTS:\n\
            - Engaging introduction with context and importance\n\
            - Multiple main sections covering different aspects\n\
            - Technical implementation details where relevant\n\
            - Real-world applications and use cases\n\
            - Critical analysis and limitations\n\
            - Future directions and conclusions\n\
            - DO NOT add a references section (will be added automatically)\n\n\
            CONTENT DEPTH:\n\
            - Minimum 1500 words\n\
            - Include technical specifications, code examples, and implementation details from sources\n\
            - Discuss advantages, limitations, and comparison with alternatives\n\
            - Provide practical insights and recommendations\n\n\
            Begin writing the comprehensive article now:",
            context_with_sources, query
        );
        
        // 6. Генерация статьи
        info!("Генерация статьи с помощью LLM...");
        let article_text = self.language_model.generate(&article_prompt).await?;
        
        // 7. Добавление списка источников
        let article_with_sources = self.add_enhanced_sources_list(&article_text);
        
        info!("Статья успешно сгенерирована");
        Ok(article_with_sources)
    }

    /// Простое ранжирование текста по релевантности без векторных вычислений
    pub fn simple_text_ranking(&self, documents: &[Document], query: &str, max_docs: usize) -> Vec<Document> {
        // Исправляем проблему с временными значениями
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        
        let mut scored_docs: Vec<(Document, f32)> = documents
            .iter()
            .map(|doc| {
                let content_lower = doc.page_content.to_lowercase();
                let title_lower = doc.metadata.get("source_title")
                    .unwrap_or(&String::new())
                    .to_lowercase();
                
                let mut score = 0.0;
                
                for word in &query_words {
                    // Подсчет упоминаний в контенте
                    let content_matches = content_lower.matches(word).count() as f32;
                    score += content_matches * 1.0;
                    
                    // Бонус за упоминание в заголовке
                    let title_matches = title_lower.matches(word).count() as f32;
                    score += title_matches * 3.0;
                }
                
                // Нормализуем по длине документа
                let normalized_score = score / (doc.page_content.len() as f32 + 1.0) * 1000.0;
                
                (doc.clone(), normalized_score)
            })
            .collect();
        
        // Сортируем по убыванию релевантности
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        scored_docs
            .into_iter()
            .take(max_docs)
            .map(|(doc, _)| doc)
            .collect()
    }

    pub async fn generate_article(&mut self, query: &str, max_retrieved_docs: usize) -> Result<String> {
        // Используем упрощенную версию без векторных вычислений
        self.generate_article_simple(query, max_retrieved_docs).await
    }

    fn prepare_context_with_sources(&self, retrieved_docs: &[Document]) -> String {
        let mut context_parts = Vec::new();
        
        for doc in retrieved_docs {
            let source_number = doc.metadata
                .get("source_number")
                .unwrap_or(&"Unknown".to_string())
                .clone();
            
            let source_title = doc.metadata
                .get("source_title")
                .unwrap_or(&"Untitled".to_string())
                .clone();
            
            let source_domain = doc.metadata
                .get("source_domain")
                .unwrap_or(&"".to_string())
                .clone();
            
            // Получаем темы из метаданных
            let topics = if let Ok(source_num) = source_number.parse::<u32>() {
                self.sources_metadata
                    .get(&source_num)
                    .map(|metadata| metadata.topics_covered.join(", "))
                    .unwrap_or_else(|| "General information".to_string())
            } else {
                "General information".to_string()
            };
            
            let context_part = format!(
                "\n=== ИСТОЧНИК {}: {} ===\nДомен: {}\nОсновные темы: {}\nКонтент:\n{}\n",
                source_number, source_title, source_domain, topics, doc.page_content
            );
            
            context_parts.push(context_part);
        }
        
        context_parts.join("\n")
    }

    fn add_enhanced_sources_list(&self, article_text: &str) -> String {
        let mut sources_list = String::from("\n\n## Источники\n\n");
        
        for (source_num, metadata) in &self.sources_metadata {
            let topics_str = metadata.topics_covered
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            
            sources_list.push_str(&format!(
                "{}. **{}**\n   - URL: {}\n   - Домен: {}\n",
                source_num, metadata.title, metadata.url, metadata.domain
            ));
            
            if !topics_str.is_empty() {
                sources_list.push_str(&format!("   - Основные темы: {}\n", topics_str));
            }
            
            if !metadata.content_summary.is_empty() {
                sources_list.push_str(&format!("   - Краткое содержание: {}\n", metadata.content_summary));
            }
            
            sources_list.push('\n');
        }
        
        format!("{}{}", article_text, sources_list)
    }

    /// Геттер для метаданных источников (для тестирования)
    pub fn sources_metadata(&self) -> &HashMap<u32, SourceMetadata> {
        &self.sources_metadata
    }
}
