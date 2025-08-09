use anyhow::Result;
use futures::{stream, StreamExt};
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::{Document, EnhancedRAGArticleGenerator};

/// Константа для ограничения одновременных загрузок
const MAX_CONCURRENT_DOWNLOADS: usize = 8;

/// Статистика загрузки документов
#[derive(Debug)]
pub struct DownloadStats {
    pub total_urls: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_bytes: usize,
    pub elapsed_time: Duration,
    pub throughput: f64, // документов в секунду
}

impl EnhancedRAGArticleGenerator {
    /// Параллельная загрузка документов с ограничением concurrency
    /// Использует futures::stream::buffer_unordered для эффективного управления
    pub async fn load_and_process_documents_parallel(
        &self,
        urls: Vec<String>,
    ) -> Result<Vec<Document>> {
        let (documents, stats) = self
            .load_documents_with_stats(urls, MAX_CONCURRENT_DOWNLOADS)
            .await?;

        info!("📊 Статистика загрузки:");
        info!("  ✅ Успешно: {} из {}", stats.successful, stats.total_urls);
        info!("  ❌ Ошибок: {}", stats.failed);
        info!("  ⏱️ Время: {:.2}с", stats.elapsed_time.as_secs_f32());
        info!("  🚀 Скорость: {:.1} док/сек", stats.throughput);
        info!(
            "  💾 Данных: {:.2} МБ",
            stats.total_bytes as f64 / 1_048_576.0
        );

        Ok(documents)
    }

    /// Параллельная загрузка с настраиваемым лимитом concurrency
    pub async fn load_documents_with_concurrency_limit(
        &self,
        urls: Vec<String>,
        concurrent_limit: usize,
    ) -> Result<Vec<Document>> {
        let (documents, _) = self
            .load_documents_with_stats(urls, concurrent_limit)
            .await?;
        Ok(documents)
    }

    /// Загрузка документов с детальной статистикой
    pub async fn load_documents_with_stats(
        &self,
        urls: Vec<String>,
        concurrent_limit: usize,
    ) -> Result<(Vec<Document>, DownloadStats)> {
        if urls.is_empty() {
            return Ok((Vec::new(), DownloadStats::default()));
        }

        info!(
            "🚀 Параллельная загрузка {} документов (concurrency: {})",
            urls.len(),
            concurrent_limit
        );

        let start_time = Instant::now();

        // Создаем HTTP клиент с оптимизированными настройками
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .user_agent("Enhanced-RAG-Generator/2.0")
            .build()?;

        // Счетчики для статистики
        let mut successful = 0;
        let mut failed = 0;
        let mut total_bytes = 0;

        // Создаем поток futures для параллельной загрузки
        let download_stream = stream::iter(urls.iter().enumerate())
            .map(|(index, url)| {
                let client = client.clone();
                let url = url.clone();

                async move {
                    info!(
                        "📥 Загрузка документа {} от {}",
                        index + 1,
                        Self::truncate_url(&url, 50)
                    );

                    match self.download_and_process_document(&client, &url).await {
                        Ok(doc) => {
                            // info!(
                            //     "✅ Документ {} загружен успешно ({} символов)",
                            //     index + 1,
                            //     doc.clone().page_content.len()
                            // );
                            Ok((doc.clone(), doc.page_content.len()))
                        }
                        Err(e) => {
                            warn!(
                                "⚠️ Ошибка загрузки документа {} ({}): {}",
                                index + 1,
                                Self::truncate_url(&url, 30),
                                e
                            );
                            Err(e)
                        }
                    }
                }
            })
            // ⭐ КЛЮЧЕВАЯ СТРОКА: buffer_unordered ограничивает concurrency
            .buffer_unordered(concurrent_limit);

        // Собираем результаты с разделением успешных и неудачных
        let mut documents = Vec::new();
        let mut results = download_stream.collect::<Vec<_>>().await;

        for result in results.drain(..) {
            match result {
                Ok((doc, bytes)) => {
                    successful += 1;
                    total_bytes += bytes;
                    documents.push(doc);
                }
                Err(_) => {
                    failed += 1;
                    // Ошибка уже залогирована выше
                }
            }
        }

        let elapsed_time = start_time.elapsed();
        let throughput = successful as f64 / elapsed_time.as_secs_f64();

        let stats = DownloadStats {
            total_urls: urls.len(),
            successful,
            failed,
            total_bytes,
            elapsed_time,
            throughput,
        };

        info!(
            "🎉 Параллельная загрузка завершена за {:.2}с",
            elapsed_time.as_secs_f32()
        );

        Ok((documents, stats))
    }

    /// Загружает и обрабатывает один документ
    async fn download_and_process_document(
        &self,
        client: &reqwest::Client,
        url: &str,
    ) -> Result<Document> {
        // HTTP запрос с обработкой ошибок
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Ошибка HTTP запроса: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "HTTP ошибка {}: {}",
                response.status(),
                url
            ));
        }

        let content = response
            .text()
            .await
            .map_err(|e| anyhow::anyhow!("Ошибка чтения содержимого: {}", e))?;

        // Валидация контента
        if content.len() < 100 {
            return Err(anyhow::anyhow!(
                "Контент слишком короткий: {} символов (минимум 100)",
                content.len()
            ));
        }

        if content.len() > 1_000_000 {
            // 1MB лимит
            return Err(anyhow::anyhow!(
                "Контент слишком большой: {} символов (максимум 1M)",
                content.len()
            ));
        }

        // Создаем документ с расширенными метаданными
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("source_url".to_string(), url.to_string());
        metadata.insert("content_length".to_string(), content.len().to_string());
        metadata.insert("download_time".to_string(), chrono::Utc::now().to_rfc3339());
        metadata.insert(
            "content_hash".to_string(),
            format!("{:x}", md5::compute(&content)),
        );

        // Извлекаем домен для метаданных
        if let Ok(parsed_url) = url::Url::parse(url) {
            if let Some(host) = parsed_url.host_str() {
                metadata.insert("domain".to_string(), host.to_string());
            }
        }

        Ok(Document {
            page_content: content,
            metadata,
        })
    }

    /// Загрузка документов с retry механизмом
    pub async fn download_with_retry(
        &self,
        client: &reqwest::Client,
        url: &str,
        max_retries: u32,
    ) -> Result<Document> {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= max_retries {
            match self.download_and_process_document(client, url).await {
                Ok(doc) => return Ok(doc),
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;

                    if attempts <= max_retries {
                        let delay = Duration::from_secs(2u64.pow(attempts.min(5))); // Cap at 32s
                        warn!(
                            "⚠️ Попытка {} неудачна для {}, повторяем через {:?}",
                            attempts,
                            Self::truncate_url(url, 40),
                            delay
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Неизвестная ошибка retry")))
    }

    /// Утилита для обрезки длинных URL в логах
    fn truncate_url(url: &str, max_len: usize) -> String {
        if url.len() <= max_len {
            url.to_string()
        } else {
            format!("{}...", &url[..max_len.saturating_sub(3)])
        }
    }
}

impl Default for DownloadStats {
    fn default() -> Self {
        Self {
            total_urls: 0,
            successful: 0,
            failed: 0,
            total_bytes: 0,
            elapsed_time: Duration::from_secs(0),
            throughput: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_url() {
        let long_url = "https://example.com/very/long/path/that/exceeds/limit";
        let truncated = EnhancedRAGArticleGenerator::truncate_url(long_url, 20);
        assert!(truncated.len() <= 20);
        assert!(truncated.ends_with("..."));
    }

    #[tokio::test]
    async fn test_parallel_download_empty_urls() {
        // Тест с пустым списком URL
        let generator = EnhancedRAGArticleGenerator::new(
            "http://test".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let result = generator.load_documents_with_stats(vec![], 5).await;
        assert!(result.is_ok());

        let (docs, stats) = result.unwrap();
        assert_eq!(docs.len(), 0);
        assert_eq!(stats.total_urls, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
    }
}
