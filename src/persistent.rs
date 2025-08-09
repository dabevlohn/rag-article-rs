use anyhow::Result;
use arroy::distances::Euclidean;
use chrono::{DateTime, Utc};
use heed::types::*;
use heed::{Database, Env, EnvOpenOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::{Document, EnhancedRAGArticleGenerator, SourceMetadata, cosine_similarity, EmbeddingModel, OllamaEmbeddings};

/// Расширенные метаданные источника с аналитикой
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSourceMetadata {
    pub url: String,
    pub title: String,
    pub domain: String,
    pub content_summary: String,
    pub topics_covered: Vec<String>,
    
    // Новые поля для аналитики
    pub usage_count: u32,
    pub quality_score: f32,
    pub language: String,
    pub content_type: SourceType,
    pub reliability_rating: ReliabilityRating,
    pub last_updated: DateTime<Utc>,
    pub content_hash: String,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceType {
    Academic,
    Documentation,
    Blog,
    News,
    Forum,
    Tutorial,
    Reference,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ReliabilityRating {
    VeryHigh,    // 0.9+: arXiv, официальная документация
    High,        // 0.7-0.9: проверенные блоги, GitHub
    Medium,      // 0.5-0.7: обычные сайты
    Low,         // 0.3-0.5: сомнительные источники
    VeryLow,     // 0.0-0.3: спам, дезинформация
}

/// Структура для кешированного документа с векторными данными
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedDocument {
    pub url: String,
    pub content_hash: String,
    pub page_content: String,
    pub metadata: HashMap<String, String>,
    pub processed_at: DateTime<Utc>,
    pub embedding: Option<Vec<f32>>,
    pub quality_metrics: DocumentQualityMetrics,
    pub language: String,
    pub topics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentQualityMetrics {
    pub content_length: usize,
    pub structure_score: f32,  // Наличие заголовков, списков
    pub readability_score: f32, // Сложность текста
    pub technical_depth: f32,   // Глубина технического контента
}

/// Кешированный запрос с семантическим анализом
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedQuery {
    pub query: String,
    pub query_hash: String,
    pub related_urls: Vec<String>,
    pub processed_at: DateTime<Utc>,
    pub embedding: Option<Vec<f32>>,
    pub query_type: QueryType,
    pub semantic_topics: Vec<String>,
    pub user_context: Option<UserContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    Tutorial,
    Comparison,
    Reference,
    Troubleshooting,
    BestPractices,
    Implementation,
    Concept,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub expertise_level: ExpertiseLevel,
    pub preferred_languages: Vec<String>,
    pub frequent_topics: Vec<String>,
    pub interaction_history: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

impl CachedDocument {
    /// Проверяет свежесть документа
    pub fn is_fresh(&self, max_days: i64) -> bool {
        let age = Utc::now() - self.processed_at;
        age.num_days() < max_days
    }

    /// Создает хеш содержимого
    pub fn content_hash(content: &str) -> String {
        format!("{:x}", md5::compute(content))
    }

    /// Анализирует качество документа
    pub fn analyze_quality(content: &str) -> DocumentQualityMetrics {
        let content_length = content.len();
        
        // Анализ структуры (заголовки, списки, код)
        let header_count = content.matches('#').count() as f32;
        let list_count = content.matches("- ").count() as f32;
        let code_count = content.matches("```").count() as f32;
        let structure_score = (header_count + list_count + code_count * 2.0) / (content_length as f32 / 1000.0).max(1.0);

        // Оценка читабельности (простая метрика)
        let sentences = content.split('.').count() as f32;
        let words = content.split_whitespace().count() as f32;
        let avg_sentence_length = if sentences > 0.0 { words / sentences } else { 0.0 };
        let readability_score = (20.0 - avg_sentence_length.min(20.0)) / 20.0;

        // Техническая глубина (ключевые слова, примеры кода)
        let technical_keywords = ["function", "class", "implementation", "algorithm", "performance", "optimization"];
        let tech_count = technical_keywords.iter()
            .map(|&word| content.to_lowercase().matches(word).count())
            .sum::<usize>() as f32;
        let technical_depth = (tech_count / (content_length as f32 / 1000.0).max(1.0)).min(1.0);

        DocumentQualityMetrics {
            content_length,
            structure_score: structure_score.min(1.0),
            readability_score: readability_score.clamp(0.0, 1.0),
            technical_depth,
        }
    }

    /// Определяет язык контента (упрощенная версия)
    pub fn detect_language(content: &str) -> String {
        let russian_chars = content.chars().filter(|c| "абвгдеёжзийклмнопрстуфхцчшщъыьэюя".contains(*c)).count();
        let total_chars = content.chars().filter(|c| c.is_alphabetic()).count();
        
        if total_chars > 0 && (russian_chars as f32 / total_chars as f32) > 0.1 {
            "ru".to_string()
        } else {
            "en".to_string()
        }
    }

    /// Извлекает основные темы из контента
    pub fn extract_topics(content: &str) -> Vec<String> {
        // Простое извлечение тем на основе заголовков
        let lines: Vec<&str> = content.lines().collect();
        let mut topics = Vec::new();
        
        for line in lines {
            if line.starts_with('#') {
                let topic = line.trim_start_matches('#').trim();
                if topic.len() > 3 && topic.len() < 100 {
                    topics.push(topic.to_string());
                }
            }
        }
        
        topics.truncate(10);
        topics
    }
}

impl EnhancedSourceMetadata {
    /// Создает расширенные метаданные из базовых
    pub fn from_basic(basic: &SourceMetadata, content: &str) -> Self {
        let quality_score = Self::calculate_quality_score(&basic.domain, content);
        let reliability_rating = Self::determine_reliability(&basic.domain, quality_score);
        let content_type = Self::classify_source_type(&basic.url, content);
        let language = CachedDocument::detect_language(content);
        
        Self {
            url: basic.url.clone(),
            title: basic.title.clone(),
            domain: basic.domain.clone(),
            content_summary: basic.content_summary.clone(),
            topics_covered: basic.topics_covered.clone(),
            usage_count: 0,
            quality_score,
            language,
            content_type,
            reliability_rating,
            last_updated: Utc::now(),
            content_hash: CachedDocument::content_hash(content),
            embedding: None,
        }
    }

    /// Вычисляет оценку качества источника
    fn calculate_quality_score(domain: &str, content: &str) -> f32 {
        let mut score = 0.5; // базовая оценка
        
        // Бонус за авторитетные домены
        let trusted_domains = [
            "arxiv.org", "github.com", "stackoverflow.com", "docs.rs", 
            "rust-lang.org", "mozilla.org", "wikipedia.org", "medium.com"
        ];
        
        if trusted_domains.iter().any(|&d| domain.contains(d)) {
            score += 0.3;
        }

        // Бонус за качество контента
        let quality_metrics = CachedDocument::analyze_quality(content);
        score += quality_metrics.structure_score * 0.2;
        score += quality_metrics.technical_depth * 0.2;

        // Штраф за короткий контент
        if content.len() < 500 {
            score -= 0.2;
        }

        score.clamp(0.0, 1.0)
    }

    /// Определяет рейтинг надежности
    fn determine_reliability(domain: &str, quality_score: f32) -> ReliabilityRating {
        let academic_domains = ["arxiv.org", "acm.org", "ieee.org", "springer.com"];
        let official_docs = ["docs.rs", "rust-lang.org", "github.com"];
        
        if academic_domains.iter().any(|&d| domain.contains(d)) {
            ReliabilityRating::VeryHigh
        } else if official_docs.iter().any(|&d| domain.contains(d)) {
            ReliabilityRating::VeryHigh
        } else if quality_score >= 0.7 {
            ReliabilityRating::High
        } else if quality_score >= 0.5 {
            ReliabilityRating::Medium
        } else if quality_score >= 0.3 {
            ReliabilityRating::Low
        } else {
            ReliabilityRating::VeryLow
        }
    }

    /// Классифицирует тип источника
    fn classify_source_type(url: &str, content: &str) -> SourceType {
        if url.contains("arxiv.org") || url.contains("acm.org") {
            SourceType::Academic
        } else if url.contains("docs.") || content.contains("# API Reference") {
            SourceType::Documentation
        } else if url.contains("blog") || url.contains("medium.com") {
            SourceType::Blog
        } else if url.contains("stackoverflow.com") || url.contains("reddit.com") {
            SourceType::Forum
        } else if content.contains("tutorial") || content.contains("how to") {
            SourceType::Tutorial
        } else if content.contains("news") || url.contains("news") {
            SourceType::News
        } else {
            SourceType::Unknown
        }
    }

    /// Увеличивает счетчик использования
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
        // Небольшое увеличение качества при частом использовании
        if self.usage_count % 10 == 0 {
            self.quality_score = (self.quality_score + 0.01).min(1.0);
        }
    }
}

impl CachedQuery {
    /// Создает хеш запроса
    pub fn query_hash(query: &str) -> String {
        format!("{:x}", md5::compute(query.to_lowercase().trim()))
    }

    /// Анализирует тип запроса
    pub fn analyze_query_type(query: &str) -> QueryType {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("tutorial") || query_lower.contains("how to") {
            QueryType::Tutorial
        } else if query_lower.contains("vs") || query_lower.contains("compare") || query_lower.contains("difference") {
            QueryType::Comparison
        } else if query_lower.contains("reference") || query_lower.contains("documentation") {
            QueryType::Reference
        } else if query_lower.contains("error") || query_lower.contains("fix") || query_lower.contains("troubleshoot") {
            QueryType::Troubleshooting
        } else if query_lower.contains("best practices") || query_lower.contains("guidelines") {
            QueryType::BestPractices
        } else if query_lower.contains("implement") || query_lower.contains("example") {
            QueryType::Implementation
        } else {
            QueryType::Concept
        }
    }

    /// Извлекает семантические темы из запроса
    pub fn extract_semantic_topics(query: &str) -> Vec<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut topics = Vec::new();
        
        // Простое извлечение ключевых технических терминов
        let tech_terms = [
            "rust", "python", "javascript", "async", "web", "framework", 
            "database", "api", "performance", "security", "testing", "deployment"
        ];
        
        for word in words {
            let word_lower = word.to_lowercase();
            if tech_terms.contains(&word_lower.as_str()) {
                topics.push(word_lower);
            }
        }
        
        topics.dedup();
        topics
    }

    /// Вычисляет семантическое сходство с другим запросом
    pub fn semantic_similarity(&self, other: &CachedQuery) -> f32 {
        if let (Some(ref embedding1), Some(ref embedding2)) = (&self.embedding, &other.embedding) {
            cosine_similarity(embedding1, embedding2)
        } else {
            // Fallback на текстовое сходство
            let topics1: std::collections::HashSet<_> = self.semantic_topics.iter().collect();
            let topics2: std::collections::HashSet<_> = other.semantic_topics.iter().collect();
            
            let intersection = topics1.intersection(&topics2).count() as f32;
            let union = topics1.union(&topics2).count() as f32;
            
            if union > 0.0 { intersection / union } else { 0.0 }
        }
    }
}

/// Расширенные настройки кеширования
#[derive(Debug, Clone)]
pub struct CacheSettings {
    pub max_document_age_days: i64,
    pub min_query_similarity: f32,
    pub max_cached_docs: usize,
    pub embedding_dim: Option<usize>,
    
    // Новые настройки
    pub enable_semantic_search: bool,
    pub min_quality_score: f32,
    pub enable_personalization: bool,
    pub auto_reindex_interval_hours: u64,
    pub max_vector_cache_size: usize,
    pub max_concurrent_downloads: usize, // НОВОЕ ПОЛЕ
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            max_document_age_days: 7,
            min_query_similarity: 0.7,
            max_cached_docs: 10,
            embedding_dim: None,
            enable_semantic_search: true,
            min_quality_score: 0.3,
            enable_personalization: false,
            auto_reindex_interval_hours: 24,
            max_vector_cache_size: 10000,
            max_concurrent_downloads: 8, // НОВОЕ ЗНАЧЕНИЕ ПО УМОЛЧАНИЮ
        }
    }
}

/// Расширенный Persistent Enhanced RAG с AI возможностями
pub struct PersistentEnhancedRAG {
    inner: EnhancedRAGArticleGenerator,
    env: Option<Env>,
    document_cache: Option<Database<Str, SerdeBincode<CachedDocument>>>,
    query_cache: Option<Database<Str, SerdeBincode<CachedQuery>>>,
    metadata_cache: Option<Database<U32<byteorder::NativeEndian>, SerdeBincode<EnhancedSourceMetadata>>>,
    
    /// Векторная база данных (arroy) для семантического поиска
    /// Используется для хранения и поиска векторных представлений документов
    #[allow(dead_code)]
    vector_db: Option<arroy::Database<Euclidean>>,
    
    cache_settings: CacheSettings,
    embedding_model: Option<Box<dyn EmbeddingModel + Send + Sync>>,
    next_source_id: u32,
}

impl PersistentEnhancedRAG {
    /// Создает новый экземпляр с памятью
    pub fn new_in_memory(
        searx_host: String,
        model_name: String,
        embedding_model_name: String,
        ollama_host: Option<String>,
    ) -> Result<Self> {
        let inner = EnhancedRAGArticleGenerator::new(
            searx_host,
            model_name,
            embedding_model_name,
            ollama_host.clone(),
        );

        let embedding_model = Some(Box::new(OllamaEmbeddings::new(
            "nomic-embed-text:latest".to_string(),
            ollama_host,
        )) as Box<dyn EmbeddingModel + Send + Sync>);

        Ok(Self {
            inner,
            env: None,
            document_cache: None,
            query_cache: None,
            metadata_cache: None,
            vector_db: None,
            cache_settings: CacheSettings::default(),
            embedding_model,
            next_source_id: 1,
        })
    }

    /// Создает новый экземпляр с персистентным хранилищем
    pub fn new_with_persistent_storage<P: AsRef<Path>>(
        db_path: P,
        searx_host: String,
        model_name: String,
        embedding_model_name: String,
        ollama_host: Option<String>,
        cache_settings: Option<CacheSettings>,
    ) -> Result<Self> {
        info!("Инициализация расширенного персистентного хранилища: {:?}", db_path.as_ref());

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(4 * 1024 * 1024 * 1024) // 4GB для векторов
                .max_dbs(15)
                .open(db_path)?
        };

        let mut wtxn = env.write_txn()?;

        let document_cache = env.create_database(&mut wtxn, Some("documents"))?;
        let query_cache = env.create_database(&mut wtxn, Some("queries"))?;
        let metadata_cache = env.create_database(&mut wtxn, Some("metadata"))?;
        let vector_db = env.create_database(&mut wtxn, Some("vectors"))?;

        wtxn.commit()?;

        let inner = EnhancedRAGArticleGenerator::new(
            searx_host,
            model_name,
            embedding_model_name,
            ollama_host.clone(),
        );

        let embedding_model = Some(Box::new(OllamaEmbeddings::new(
            "nomic-embed-text:latest".to_string(),
            ollama_host,
        )) as Box<dyn EmbeddingModel + Send + Sync>);

        info!("Расширенное персистентное хранилище инициализировано");

        Ok(Self {
            inner,
            env: Some(env),
            document_cache: Some(document_cache),
            query_cache: Some(query_cache),
            metadata_cache: Some(metadata_cache),
            vector_db: Some(vector_db),
            cache_settings: cache_settings.unwrap_or_default(),
            embedding_model,
            next_source_id: 1,
        })
    }

    /// Генерирует статью с использованием расширенного AI кеша
    pub async fn generate_article_with_enhanced_cache(
        &mut self,
        query: &str,
        max_retrieved_docs: usize,
        user_context: Option<UserContext>,
    ) -> Result<String> {
        info!("Генерация статьи с расширенным AI кешем для: {}", query);

        if self.env.is_none() {
            return self.inner.generate_article(query, max_retrieved_docs).await;
        }

        // 1. Семантический поиск похожих запросов
        let similar_queries = if self.cache_settings.enable_semantic_search {
            self.semantic_query_search(query).await?
        } else {
            self.find_similar_queries(query).await?
        };

        // 2. Извлечение качественных документов
        let mut cached_docs = Vec::new();
        let mut fresh_urls = Vec::new();

        for cached_query in similar_queries {
            for url in cached_query.related_urls {
                if let Some(doc) = self.get_cached_document(&url).await? {
                    if doc.is_fresh(self.cache_settings.max_document_age_days) && 
                       self.meets_quality_threshold(&doc) {
                        cached_docs.push(self.cached_to_document(&doc));
                    } else {
                        fresh_urls.push(url);
                    }
                }
            }
        }

        // 3. Интеллектуальная загрузка новых документов
        if cached_docs.len() < max_retrieved_docs {
            let needed_docs = max_retrieved_docs - cached_docs.len();
            let new_urls = self.inner.search_and_collect_urls(query, needed_docs as u32).await?;
            
            // Фильтрация по качественным источникам
            let filtered_urls = self.filter_quality_urls(new_urls).await?;
            fresh_urls.extend(filtered_urls);
        }

        // 4. Загрузка и расширенное кеширование с параллельной загрузкой
        if !fresh_urls.is_empty() {
            // ИСПОЛЬЗУЕМ ПАРАЛЛЕЛЬНУЮ ЗАГРУЗКУ
            let new_docs = self.inner.load_documents_with_concurrency_limit(
                fresh_urls.clone(), 
                self.cache_settings.max_concurrent_downloads
            ).await?;
            
            for doc in &new_docs {
                self.enhanced_cache_document(doc).await?;
            }
            
            cached_docs.extend(new_docs);
        }

        // 5. Кеширование запроса с семантическим анализом
        self.enhanced_cache_query(query, &fresh_urls, user_context).await?;

        // 6. Интеллектуальное ранжирование
        let retrieved_docs = self.intelligent_ranking(&cached_docs, query, max_retrieved_docs).await?;
        let context = self.inner.prepare_context_with_sources(&retrieved_docs);
        
        // 7. Генерация с расширенным промптом
        let article_prompt = self.build_ai_enhanced_prompt(query, &context, &cached_docs).await?;
        let article_text = self.inner.language_model.generate(&article_prompt).await?;
        
        info!("Статья сгенерирована с использованием {} документов (AI-enhanced)", retrieved_docs.len());
        Ok(self.inner.add_enhanced_sources_list(&article_text))
    }

    /// Семантический поиск запросов с использованием embeddings
    async fn semantic_query_search(&self, query: &str) -> Result<Vec<CachedQuery>> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(Vec::new()),
        };

        if let Some(ref embedding_model) = self.embedding_model {
            let query_embedding = embedding_model.embed_query(query).await?;
            let query_cache = self.query_cache.as_ref().unwrap();
            let rtxn = env.read_txn()?;

            let mut similar_queries = Vec::new();

            for entry in query_cache.iter(&rtxn)? {
                let (_, cached_query) = entry?;
                let similarity = cached_query.semantic_similarity(&CachedQuery {
                    query: query.to_string(),
                    query_hash: CachedQuery::query_hash(query),
                    related_urls: Vec::new(),
                    processed_at: Utc::now(),
                    embedding: Some(query_embedding.clone()),
                    query_type: CachedQuery::analyze_query_type(query),
                    semantic_topics: CachedQuery::extract_semantic_topics(query),
                    user_context: None,
                });

                if similarity >= self.cache_settings.min_query_similarity {
                    similar_queries.push(cached_query);
                }
            }

            // Сортировка по убыванию сходства
            similar_queries.sort_by(|a, b| {
                let sim_a = a.semantic_similarity(&CachedQuery {
                    query: query.to_string(),
                    query_hash: CachedQuery::query_hash(query),
                    related_urls: Vec::new(),
                    processed_at: Utc::now(),
                    embedding: Some(query_embedding.clone()),
                    query_type: CachedQuery::analyze_query_type(query),
                    semantic_topics: CachedQuery::extract_semantic_topics(query),
                    user_context: None,
                });
                let sim_b = b.semantic_similarity(&CachedQuery {
                    query: query.to_string(),
                    query_hash: CachedQuery::query_hash(query),
                    related_urls: Vec::new(),
                    processed_at: Utc::now(),
                    embedding: Some(query_embedding.clone()),
                    query_type: CachedQuery::analyze_query_type(query),
                    semantic_topics: CachedQuery::extract_semantic_topics(query),
                    user_context: None,
                });
                sim_b.partial_cmp(&sim_a).unwrap_or(std::cmp::Ordering::Equal)
            });

            similar_queries.truncate(5);
            Ok(similar_queries)
        } else {
            // Fallback на текстовый поиск
            self.find_similar_queries(query).await
        }
    }

    /// Проверяет соответствие документа порогу качества
    fn meets_quality_threshold(&self, doc: &CachedDocument) -> bool {
        doc.quality_metrics.structure_score >= self.cache_settings.min_quality_score ||
        doc.quality_metrics.technical_depth >= self.cache_settings.min_quality_score
    }

    /// Фильтрует URL по качественным источникам
    async fn filter_quality_urls(&self, urls: Vec<String>) -> Result<Vec<String>> {
        let mut filtered = Vec::new();
        
        for url in urls {
            // Проверяем метаданные источника если есть
            if let Some(metadata) = self.get_source_metadata_by_url(&url).await? {
                if metadata.reliability_rating >= ReliabilityRating::Medium {
                    filtered.push(url);
                }
            } else {
                // Если метаданных нет, добавляем (будет оценено при загрузке)
                filtered.push(url);
            }
        }
        
        Ok(filtered)
    }

    /// Расширенное кеширование документа с аналитикой
    async fn enhanced_cache_document(&mut self, doc: &Document) -> Result<()> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(()),
        };

        let url = doc.metadata.get("source_url").unwrap();
        
        // Создаем embedding если включен семантический поиск
        let embedding = if self.cache_settings.enable_semantic_search {
            if let Some(ref embedding_model) = self.embedding_model {
                Some(embedding_model.embed_query(&doc.page_content).await?)
            } else {
                None
            }
        } else {
            None
        };

        // Создаем расширенный кешированный документ
        let quality_metrics = CachedDocument::analyze_quality(&doc.page_content);
        let language = CachedDocument::detect_language(&doc.page_content);
        let topics = CachedDocument::extract_topics(&doc.page_content);

        let cached_doc = CachedDocument {
            url: url.clone(),
            content_hash: CachedDocument::content_hash(&doc.page_content),
            page_content: doc.page_content.clone(),
            metadata: doc.metadata.clone(),
            processed_at: Utc::now(),
            embedding,
            quality_metrics,
            language,
            topics,
        };

        // Сохраняем документ
        let document_cache = self.document_cache.as_ref().unwrap();
        let mut wtxn = env.write_txn()?;
        document_cache.put(&mut wtxn, url, &cached_doc)?;

        // Создаем и сохраняем расширенные метаданные источника
        if let Some(source_metadata) = self.inner.sources_metadata().get(&(self.next_source_id)) {
            let enhanced_metadata = EnhancedSourceMetadata::from_basic(source_metadata, &doc.page_content);
            let metadata_cache = self.metadata_cache.as_ref().unwrap();
            metadata_cache.put(&mut wtxn, &self.next_source_id, &enhanced_metadata)?;
            self.next_source_id += 1;
        }

        wtxn.commit()?;
        Ok(())
    }

    /// Расширенное кеширование запроса с семантическим анализом
    async fn enhanced_cache_query(
        &self, 
        query: &str, 
        urls: &[String], 
        user_context: Option<UserContext>
    ) -> Result<()> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(()),
        };

        // Создаем embedding для запроса
        let embedding = if self.cache_settings.enable_semantic_search {
            if let Some(ref embedding_model) = self.embedding_model {
                Some(embedding_model.embed_query(query).await?)
            } else {
                None
            }
        } else {
            None
        };

        let cached_query = CachedQuery {
            query: query.to_string(),
            query_hash: CachedQuery::query_hash(query),
            related_urls: urls.to_vec(),
            processed_at: Utc::now(),
            embedding,
            query_type: CachedQuery::analyze_query_type(query),
            semantic_topics: CachedQuery::extract_semantic_topics(query),
            user_context,
        };

        let query_cache = self.query_cache.as_ref().unwrap();
        let mut wtxn = env.write_txn()?;
        query_cache.put(&mut wtxn, &cached_query.query_hash, &cached_query)?;
        wtxn.commit()?;

        Ok(())
    }

    /// Интеллектуальное ранжирование документов
    async fn intelligent_ranking(
        &self, 
        documents: &[Document], 
        query: &str, 
        max_docs: usize
    ) -> Result<Vec<Document>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Семантическое ранжирование если доступно
        if self.cache_settings.enable_semantic_search && self.embedding_model.is_some() {
            return self.semantic_ranking(documents, query, max_docs).await;
        }

        // Fallback на улучшенное текстовое ранжирование
        Ok(self.enhanced_text_ranking(documents, query, max_docs))
    }

    /// Семантическое ранжирование с использованием embeddings
    async fn semantic_ranking(
        &self,
        documents: &[Document],
        query: &str, 
        max_docs: usize
    ) -> Result<Vec<Document>> {
        let embedding_model = self.embedding_model.as_ref().unwrap();
        let query_embedding = embedding_model.embed_query(query).await?;

        let mut scored_docs = Vec::new();

        for doc in documents {
            // Получаем embedding документа из кеша или создаем новый
            let doc_embedding = if let Some(cached_doc) = self.get_cached_document(
                doc.metadata.get("source_url").unwrap()
            ).await? {
                cached_doc.embedding.unwrap_or_else(|| vec![0.0; query_embedding.len()])
            } else {
                embedding_model.embed_query(&doc.page_content).await?
            };

            let semantic_score = cosine_similarity(&query_embedding, &doc_embedding);
            
            // Комбинируем семантическую оценку с качественными метриками
            let quality_bonus = if let Some(cached_doc) = self.get_cached_document(
                doc.metadata.get("source_url").unwrap()
            ).await? {
                cached_doc.quality_metrics.technical_depth * 0.2 +
                cached_doc.quality_metrics.structure_score * 0.1
            } else {
                0.0
            };

            let final_score = semantic_score + quality_bonus;
            scored_docs.push((doc.clone(), final_score));
        }

        // Сортируем и возвращаем топ документы
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored_docs.into_iter().take(max_docs).map(|(doc, _)| doc).collect())
    }

    /// Улучшенное текстовое ранжирование с учетом качества
    fn enhanced_text_ranking(&self, documents: &[Document], query: &str, max_docs: usize) -> Vec<Document> {
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
                
                // Базовое текстовое сходство
                for word in &query_words {
                    let content_matches = content_lower.matches(word).count() as f32;
                    score += content_matches * 1.0;
                    
                    let title_matches = title_lower.matches(word).count() as f32;
                    score += title_matches * 3.0;
                }
                
                // Нормализация по длине
                score = score / (doc.page_content.len() as f32 + 1.0) * 1000.0;
                
                // Бонус за качество документа (если доступны метрики)
                // В простом режиме даем небольшой бонус за длину и структуру
                let length_bonus = if doc.page_content.len() > 1000 { 0.1 } else { 0.0 };
                let structure_bonus = if doc.page_content.contains("```") || 
                                        doc.page_content.matches('#').count() > 2 { 0.1 } else { 0.0 };
                
                score += length_bonus + structure_bonus;
                
                (doc.clone(), score)
            })
            .collect();
        
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_docs.into_iter().take(max_docs).map(|(doc, _)| doc).collect()
    }

    /// Создает улучшенный промпт с учетом AI аналитики
    async fn build_ai_enhanced_prompt(
        &self, 
        query: &str, 
        context: &str, 
        cached_docs: &[Document]
    ) -> Result<String> {
        let semantic_info = if self.cache_settings.enable_semantic_search {
            "\nNOTE: This response utilizes advanced semantic analysis and quality-filtered sources for enhanced accuracy and relevance.\n"
        } else {
            ""
        };

        let parallel_info = format!(
            "\nPARALLEL PROCESSING: Documents were loaded using {} concurrent streams for optimal performance.\n",
            self.cache_settings.max_concurrent_downloads
        );

        let quality_info = if cached_docs.len() > 5 {
            format!(
                "\nSOURCE QUALITY: Using {} high-quality sources selected through AI-powered filtering and ranking.\n",
                cached_docs.len()
            )
        } else {
            String::new()
        };

        Ok(format!(
            "You are an expert academic writer with access to AI-enhanced research capabilities.{}{}{}\
            \nAVAILABLE SOURCE DOCUMENTS:\n{}\n\n\
            TASK: Write a comprehensive, well-structured article about: {}\n\n\
            AI-ENHANCED REQUIREMENTS:\n\
            1. Leverage semantic understanding to connect related concepts across sources\n\
            2. Prioritize information from high-quality, verified sources\n\
            3. Provide deeper technical insights by synthesizing information intelligently\n\
            4. Include relevant cross-references and contextual connections\n\
            5. Maintain academic rigor while ensuring accessibility\n\n\
            CRITICAL CITATION REQUIREMENTS:\n\
            1. Use citations [X] for specific information, data, quotes, or concepts\n\
            2. Ensure citations guide readers to detailed source information\n\
            3. Use multiple citations [1][2] when supported by multiple sources\n\
            4. Connect citations logically to elaborating sources\n\n\
            WRITING STYLE:\n\
            - Expert-level academic tone with practical insights\n\
            - Clear section headings (##, ###) with logical flow\n\
            - Integration of technical details, examples, and methodologies\n\
            - Each paragraph developing distinct aspects with cross-connections\n\n\
            STRUCTURE REQUIREMENTS:\n\
            - Engaging introduction with contextual importance\n\
            - Multiple comprehensive sections covering different aspects\n\
            - Technical implementation details with real-world applications\n\
            - Critical analysis, limitations, and comparative insights\n\
            - Future directions and actionable conclusions\n\
            - NO references section (added automatically)\n\n\
            CONTENT DEPTH:\n\
            - Minimum 2000 words with substantial technical depth\n\
            - Include specifications, code examples, implementation details\n\
            - Discuss advantages, limitations, alternatives with nuanced analysis\n\
            - Provide actionable insights and expert recommendations\n\n\
            Begin writing the AI-enhanced comprehensive article:",
            semantic_info, parallel_info, quality_info, context, query
        ))
    }

    /// Получает метаданные источника по URL
    async fn get_source_metadata_by_url(&self, url: &str) -> Result<Option<EnhancedSourceMetadata>> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(None),
        };

        let metadata_cache = self.metadata_cache.as_ref().unwrap();
        let rtxn = env.read_txn()?;

        // Поиск по всем метаданным (неэффективно, но работает)
        for entry in metadata_cache.iter(&rtxn)? {
            let (_, metadata) = entry?;
            if metadata.url == url {
                return Ok(Some(metadata));
            }
        }

        Ok(None)
    }

    /// Обновляет статистику использования источника
    pub async fn update_source_usage(&mut self, source_id: u32) -> Result<()> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(()),
        };

        let metadata_cache = self.metadata_cache.as_ref().unwrap();
        let mut wtxn = env.write_txn()?;

        if let Some(mut metadata) = metadata_cache.get(&wtxn, &source_id)? {
            metadata.increment_usage();
            metadata_cache.put(&mut wtxn, &source_id, &metadata)?;
        }

        wtxn.commit()?;
        Ok(())
    }

    /// Получает статистику по качеству источников
    pub async fn get_quality_stats(&self) -> Result<QualityStats> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(QualityStats::default()),
        };

        let metadata_cache = self.metadata_cache.as_ref().unwrap();
        let rtxn = env.read_txn()?;

        let mut stats = QualityStats::default();

        for entry in metadata_cache.iter(&rtxn)? {
            let (_, metadata) = entry?;
            stats.total_sources += 1;
            
            match metadata.reliability_rating {
                ReliabilityRating::VeryHigh => stats.very_high_quality += 1,
                ReliabilityRating::High => stats.high_quality += 1,
                ReliabilityRating::Medium => stats.medium_quality += 1,
                ReliabilityRating::Low => stats.low_quality += 1,
                ReliabilityRating::VeryLow => stats.very_low_quality += 1,
            }

            stats.average_quality_score += metadata.quality_score;
        }

        if stats.total_sources > 0 {
            stats.average_quality_score /= stats.total_sources as f32;
        }

        Ok(stats)
    }

    /// Векторный поиск по документам (заготовка для будущих версий)
    /// Требует дополнительной интеграции с arroy для полнофункционального поиска
    #[allow(dead_code)]
    pub async fn vector_similarity_search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<Document>> {
        // TODO: Полная реализация векторного поиска с arroy
        // Текущая версия использует семантический поиск через embeddings в документах
        // Для production использования требуется:
        // 1. Создание arroy индекса из всех document embeddings
        // 2. Поиск k ближайших соседей для query_embedding
        // 3. Извлечение соответствующих документов из кеша
        // 4. Возврат отсортированного списка документов
        
        // Временная заглушка - возвращает пустой список
        let _ = (query_embedding, k); // убираем warnings о неиспользуемых параметрах
        Ok(Vec::new())
    }

    // Наследуем остальные методы с обновленными сигнатурами
    pub async fn generate_article_with_cache(
        &mut self,
        query: &str,
        max_retrieved_docs: usize,
    ) -> Result<String> {
        // Используем расширенную версию без пользовательского контекста
        self.generate_article_with_enhanced_cache(query, max_retrieved_docs, None).await
    }

    // Остальные базовые методы остаются теми же...
    async fn find_similar_queries(&self, query: &str) -> Result<Vec<CachedQuery>> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(Vec::new()),
        };

        let query_cache = self.query_cache.as_ref().unwrap();
        let rtxn = env.read_txn()?;
        
        let mut similar_queries = Vec::new();
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        for entry in query_cache.iter(&rtxn)? {
            let (_, cached_query) = entry?;
            let cached_lower = cached_query.query.to_lowercase();
            let cached_words: Vec<&str> = cached_lower.split_whitespace().collect();
            let similarity = self.calculate_word_similarity(&query_words, &cached_words);
            
            if similarity >= self.cache_settings.min_query_similarity {
                similar_queries.push(cached_query);
                if similar_queries.len() >= 3 {
                    break;
                }
            }
        }

        Ok(similar_queries)
    }

    fn calculate_word_similarity(&self, words1: &[&str], words2: &[&str]) -> f32 {
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        for word1 in words1 {
            if words2.contains(word1) {
                matches += 1;
            }
        }

        matches as f32 / words1.len().max(words2.len()) as f32
    }

    async fn get_cached_document(&self, url: &str) -> Result<Option<CachedDocument>> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(None),
        };

        let document_cache = self.document_cache.as_ref().unwrap();
        let rtxn = env.read_txn()?;
        
        Ok(document_cache.get(&rtxn, url)?)
    }

    fn cached_to_document(&self, cached: &CachedDocument) -> Document {
        Document {
            page_content: cached.page_content.clone(),
            metadata: cached.metadata.clone(),
        }
    }

    pub async fn cache_stats(&self) -> Result<CacheStats> {
        let env = match &self.env {
            Some(env) => env,
            None => {
                return Ok(CacheStats {
                    total_documents: 0,
                    total_queries: 0,
                    fresh_documents: 0,
                    database_size_mb: 0.0,
                });
            }
        };

        let rtxn = env.read_txn()?;
        
        let doc_count = self.document_cache.as_ref().unwrap().len(&rtxn)? as usize;
        let query_count = self.query_cache.as_ref().unwrap().len(&rtxn)? as usize;

        let mut fresh_count = 0;
        for entry in self.document_cache.as_ref().unwrap().iter(&rtxn)? {
            let (_, doc) = entry?;
            if doc.is_fresh(self.cache_settings.max_document_age_days) {
                fresh_count += 1;
            }
        }

        let info = env.info();
        let db_size_mb = info.map_size as f64 / (1024.0 * 1024.0);

        Ok(CacheStats {
            total_documents: doc_count,
            total_queries: query_count,
            fresh_documents: fresh_count,
            database_size_mb: db_size_mb,
        })
    }

    pub async fn cleanup_cache(&mut self) -> Result<CleanupStats> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(CleanupStats::default()),
        };

        let mut wtxn = env.write_txn()?;
        let mut deleted_docs = 0;
        let mut deleted_queries = 0;

        // Удаляем устаревшие документы
        let documents_to_delete: Vec<String> = {
            let mut docs = Vec::new();
            for entry in self.document_cache.as_ref().unwrap().iter(&wtxn)? {
                let (url, doc) = entry?;
                if !doc.is_fresh(self.cache_settings.max_document_age_days) {
                    docs.push(url.to_string());
                }
            }
            docs
        };

        for url in documents_to_delete {
            self.document_cache.as_ref().unwrap().delete(&mut wtxn, &url)?;
            deleted_docs += 1;
        }

        // Удаляем устаревшие запросы (ИСПРАВЛЕНО: используем wtxn вместо rtxn)
        let queries_to_delete: Vec<String> = {
            let mut queries = Vec::new();
            for entry in self.query_cache.as_ref().unwrap().iter(&wtxn)? {
                let (hash, query) = entry?;
                let age = Utc::now() - query.processed_at;
                if age.num_days() > 30 {
                    queries.push(hash.to_string());
                }
            }
            queries
        };

        for hash in queries_to_delete {
            self.query_cache.as_ref().unwrap().delete(&mut wtxn, &hash)?;
            deleted_queries += 1;
        }

        wtxn.commit()?;

        info!("Расширенная очистка кеша завершена: удалено {} документов и {} запросов", 
              deleted_docs, deleted_queries);

        Ok(CleanupStats {
            deleted_documents: deleted_docs,
            deleted_queries,
        })
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub total_documents: usize,
    pub total_queries: usize,
    pub fresh_documents: usize,
    pub database_size_mb: f64,
}

#[derive(Debug, Default)]
pub struct CleanupStats {
    pub deleted_documents: usize,
    pub deleted_queries: usize,
}

#[derive(Debug, Default)]
pub struct QualityStats {
    pub total_sources: usize,
    pub very_high_quality: usize,
    pub high_quality: usize,
    pub medium_quality: usize,
    pub low_quality: usize,
    pub very_low_quality: usize,
    pub average_quality_score: f32,
}