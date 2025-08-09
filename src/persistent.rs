use anyhow::Result;
use arroy::distances::Euclidean;
use chrono::{DateTime, Utc};
use heed::types::*;
use heed::{Database, Env, EnvOpenOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{info, warn};

use crate::{Document, EnhancedRAGArticleGenerator, SourceMetadata, cosine_similarity};

/// Структура для кешированного документа с метаданными
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedDocument {
    pub url: String,
    pub content_hash: String,
    pub page_content: String,
    pub metadata: HashMap<String, String>,
    pub processed_at: DateTime<Utc>,
    pub embedding: Option<Vec<f32>>,
}

/// Структура для кешированного запроса
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedQuery {
    pub query: String,
    pub query_hash: String,
    pub related_urls: Vec<String>,
    pub processed_at: DateTime<Utc>,
    pub embedding: Option<Vec<f32>>,
}

impl CachedDocument {
    /// Проверяет свежесть документа (не старше N дней)
    pub fn is_fresh(&self, max_days: i64) -> bool {
        let age = Utc::now() - self.processed_at;
        age.num_days() < max_days
    }

    /// Создает хеш содержимого для проверки изменений
    pub fn content_hash(content: &str) -> String {
        format!("{:x}", md5::compute(content))
    }
}

impl CachedQuery {
    /// Создает хеш запроса для индексации
    pub fn query_hash(query: &str) -> String {
        format!("{:x}", md5::compute(query.to_lowercase().trim()))
    }

    /// Вычисляет семантическое сходство между запросами
    pub fn similarity(&self, other_embedding: &[f32]) -> Option<f32> {
        if let Some(ref embedding) = self.embedding {
            Some(cosine_similarity(embedding, other_embedding))
        } else {
            None
        }
    }
}

/// Persistent Enhanced RAG Generator с поддержкой LMDB
pub struct PersistentEnhancedRAG {
    /// Базовая функциональность RAG
    inner: EnhancedRAGArticleGenerator,
    
    /// LMDB окружение
    env: Option<Env>,
    
    /// База данных для документов
    document_cache: Option<Database<Str, SerdeBincode<CachedDocument>>>,
    
    /// База данных для запросов
    query_cache: Option<Database<Str, SerdeBincode<CachedQuery>>>,
    
    /// База данных для метаданных источников
    metadata_cache: Option<Database<OwnedType<u32>, SerdeBincode<SourceMetadata>>>,
    
    /// Векторная база данных (arroy)
    vector_db: Option<arroy::Database<Euclidean>>,
    
    /// Настройки кеширования
    cache_settings: CacheSettings,
}

#[derive(Debug, Clone)]
pub struct CacheSettings {
    /// Максимальный возраст документов в днях
    pub max_document_age_days: i64,
    /// Минимальное сходство для использования кешированных запросов
    pub min_query_similarity: f32,
    /// Максимальное количество релевантных документов из кеша
    pub max_cached_docs: usize,
    /// Размер embedding для векторного поиска
    pub embedding_dim: Option<usize>,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            max_document_age_days: 7,
            min_query_similarity: 0.7,
            max_cached_docs: 10,
            embedding_dim: None,
        }
    }
}

impl PersistentEnhancedRAG {
    /// Создает новый экземпляр с памятью (без персистентного хранилища)
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
            ollama_host,
        );

        Ok(Self {
            inner,
            env: None,
            document_cache: None,
            query_cache: None,
            metadata_cache: None,
            vector_db: None,
            cache_settings: CacheSettings::default(),
        })
    }

    /// Создает новый экземпляр с персистентным хранилищем LMDB
    pub fn new_with_persistent_storage<P: AsRef<Path>>(
        db_path: P,
        searx_host: String,
        model_name: String,
        embedding_model_name: String,
        ollama_host: Option<String>,
        cache_settings: Option<CacheSettings>,
    ) -> Result<Self> {
        info!("Инициализация персистентного хранилища: {:?}", db_path.as_ref());

        // Создаем LMDB окружение
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(2 * 1024 * 1024 * 1024) // 2GB
                .max_dbs(10)
                .open(db_path)?
        };

        let mut wtxn = env.write_txn()?;

        // Создаем базы данных
        let document_cache = env.create_database(&mut wtxn, Some("documents"))?;
        let query_cache = env.create_database(&mut wtxn, Some("queries"))?;
        let metadata_cache = env.create_database(&mut wtxn, Some("metadata"))?;

        // Инициализируем векторную базу данных (arroy)
        let vector_db = env.create_database(&mut wtxn, Some("vectors"))?;

        wtxn.commit()?;

        let inner = EnhancedRAGArticleGenerator::new(
            searx_host,
            model_name,
            embedding_model_name,
            ollama_host,
        );

        info!("Персистентное хранилище успешно инициализировано");

        Ok(Self {
            inner,
            env: Some(env),
            document_cache: Some(document_cache),
            query_cache: Some(query_cache),
            metadata_cache: Some(metadata_cache),
            vector_db: Some(vector_db),
            cache_settings: cache_settings.unwrap_or_default(),
        })
    }

    /// Генерирует статью с использованием кеша
    pub async fn generate_article_with_cache(
        &mut self,
        query: &str,
        max_retrieved_docs: usize,
    ) -> Result<String> {
        info!("Генерация статьи с кешированием для запроса: {}", query);

        // Если персистентное хранилище не настроено, используем обычную генерацию
        if self.env.is_none() {
            return self.inner.generate_article(query, max_retrieved_docs).await;
        }

        // 1. Проверяем кеш похожих запросов
        let similar_queries = self.find_similar_queries(query).await?;
        
        // 2. Собираем кешированные документы
        let mut cached_docs = Vec::new();
        let mut fresh_urls = Vec::new();

        if !similar_queries.is_empty() {
            info!("Найдено {} похожих запросов в кеше", similar_queries.len());
            
            for cached_query in similar_queries {
                for url in cached_query.related_urls {
                    if let Some(doc) = self.get_cached_document(&url).await? {
                        if doc.is_fresh(self.cache_settings.max_document_age_days) {
                            cached_docs.push(self.cached_to_document(&doc));
                        } else {
                            fresh_urls.push(url);
                        }
                    } else {
                        fresh_urls.push(url);
                    }
                }
            }
        }

        // 3. Если недостаточно кешированных документов, ищем новые
        let needed_docs = max_retrieved_docs.saturating_sub(cached_docs.len());
        if needed_docs > 0 || cached_docs.is_empty() {
            info!("Загружаем {} новых документов", needed_docs);
            
            let new_urls = self.inner.search_and_collect_urls(query, needed_docs as u32).await?;
            fresh_urls.extend(new_urls);
        }

        // 4. Загружаем и кешируем новые документы
        if !fresh_urls.is_empty() {
            let new_docs = self.inner.load_and_process_documents(fresh_urls.clone()).await?;
            
            for doc in &new_docs {
                self.cache_document(doc).await?;
            }
            
            cached_docs.extend(new_docs);
        }

        // 5. Кешируем текущий запрос
        self.cache_query(query, &fresh_urls).await?;

        // 6. Ранжируем и генерируем статью
        if cached_docs.is_empty() {
            return Ok("Не удалось найти релевантные документы для запроса.".to_string());
        }

        let retrieved_docs = self.inner.simple_text_ranking(&cached_docs, query, max_retrieved_docs);
        let context = self.inner.prepare_context_with_sources(&retrieved_docs);
        
        // Создание промпта с учетом кешированных данных
        let article_prompt = self.build_enhanced_prompt(query, &context, cached_docs.len(), fresh_urls.len());
        
        let article_text = self.inner.language_model.generate(&article_prompt).await?;
        let final_article = self.inner.add_enhanced_sources_list(&article_text);

        info!("Статья сгенерирована с использованием {} документов ({} из кеша)", 
              retrieved_docs.len(), cached_docs.len() - fresh_urls.len());

        Ok(final_article)
    }

    /// Ищет похожие запросы в кеше
    async fn find_similar_queries(&self, query: &str) -> Result<Vec<CachedQuery>> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(Vec::new()),
        };

        let query_cache = self.query_cache.as_ref().unwrap();
        let rtxn = env.read_txn()?;
        
        let mut similar_queries = Vec::new();
        let query_words: Vec<&str> = query.to_lowercase().split_whitespace().collect();

        // Простой поиск по ключевым словам (для демонстрации)
        for entry in query_cache.iter(&rtxn)? {
            let (_, cached_query) = entry?;
            
            let cached_words: Vec<&str> = cached_query.query.to_lowercase().split_whitespace().collect();
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

    /// Простое вычисление сходства между словами запросов
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

    /// Получает кешированный документ
    async fn get_cached_document(&self, url: &str) -> Result<Option<CachedDocument>> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(None),
        };

        let document_cache = self.document_cache.as_ref().unwrap();
        let rtxn = env.read_txn()?;
        
        if let Some(doc) = document_cache.get(&rtxn, url)? {
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }

    /// Кеширует документ
    async fn cache_document(&self, doc: &Document) -> Result<()> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(()),
        };

        let document_cache = self.document_cache.as_ref().unwrap();
        let mut wtxn = env.write_txn()?;

        let url = doc.metadata.get("source_url").unwrap();
        let cached_doc = CachedDocument {
            url: url.clone(),
            content_hash: CachedDocument::content_hash(&doc.page_content),
            page_content: doc.page_content.clone(),
            metadata: doc.metadata.clone(),
            processed_at: Utc::now(),
            embedding: None, // TODO: Добавить embedding если доступно
        };

        document_cache.put(&mut wtxn, url, &cached_doc)?;
        wtxn.commit()?;

        Ok(())
    }

    /// Кеширует запрос
    async fn cache_query(&self, query: &str, urls: &[String]) -> Result<()> {
        let env = match &self.env {
            Some(env) => env,
            None => return Ok(()),
        };

        let query_cache = self.query_cache.as_ref().unwrap();
        let mut wtxn = env.write_txn()?;

        let query_hash = CachedQuery::query_hash(query);
        let cached_query = CachedQuery {
            query: query.to_string(),
            query_hash: query_hash.clone(),
            related_urls: urls.to_vec(),
            processed_at: Utc::now(),
            embedding: None, // TODO: Добавить embedding если доступно
        };

        query_cache.put(&mut wtxn, &query_hash, &cached_query)?;
        wtxn.commit()?;

        Ok(())
    }

    /// Конвертирует кешированный документ в обычный документ
    fn cached_to_document(&self, cached: &CachedDocument) -> Document {
        Document {
            page_content: cached.page_content.clone(),
            metadata: cached.metadata.clone(),
        }
    }

    /// Создает улучшенный промпт с информацией о кеше
    fn build_enhanced_prompt(&self, query: &str, context: &str, total_docs: usize, new_docs: usize) -> String {
        let cache_info = if new_docs > 0 && total_docs > new_docs {
            format!(
                "\n\nNOTE: This response utilizes {} documents total - {} from previous research cache and {} newly retrieved sources. \
                This ensures comprehensive coverage while building upon previous knowledge.\n\n",
                total_docs, total_docs - new_docs, new_docs
            )
        } else {
            String::new()
        };

        format!(
            "You are an expert academic writer creating a comprehensive research article based on provided context documents.{}\
            \nAVAILABLE SOURCE DOCUMENTS:\n{}\n\n\
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
            cache_info, context, query
        )
    }

    /// Статистика кеша
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

        // Подсчитываем свежие документы
        let mut fresh_count = 0;
        for entry in self.document_cache.as_ref().unwrap().iter(&rtxn)? {
            let (_, doc) = entry?;
            if doc.is_fresh(self.cache_settings.max_document_age_days) {
                fresh_count += 1;
            }
        }

        let stats = env.stat()?;
        let db_size_mb = (stats.page_size * stats.depth as usize) as f64 / (1024.0 * 1024.0);

        Ok(CacheStats {
            total_documents: doc_count,
            total_queries: query_count,
            fresh_documents: fresh_count,
            database_size_mb: db_size_mb,
        })
    }

    /// Очистка устаревших записей
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

        // Удаляем устаревшие запросы (старше 30 дней)
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

        info!("Очистка кеша завершена: удалено {} документов и {} запросов", 
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