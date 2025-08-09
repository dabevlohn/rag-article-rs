use crate::{EnhancedRAGArticleGenerator, persistent::*};
use anyhow::Result;
use clap::{Arg, Command};
use tracing::{info, error};
use std::path::PathBuf;

pub fn cli() -> Command {
    Command::new("enhanced-rag-generator")
        .about("Enhanced RAG Article Generator - создание статей с автоматическим цитированием источников")
        .version("1.0.0")
        .arg(
            Arg::new("query")
                .help("Запрос для генерации статьи")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("searx-host")
                .long("searx-host")
                .help("Адрес SearXNG сервера")
                .default_value("http://127.0.0.1:8080"),
        )
        .arg(
            Arg::new("ollama-host")
                .long("ollama-host")
                .help("Адрес Ollama сервера")
                .default_value("http://localhost:11434"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .short('m')
                .help("Название модели Ollama для генерации текста")
                .default_value("qwen2.5:32b"),
        )
        .arg(
            Arg::new("embedding-model")
                .long("embedding-model")
                .help("Название модели для создания embeddings")
                .default_value("nomic-embed-text:latest"),
        )
        .arg(
            Arg::new("max-docs")
                .long("max-docs")
                .help("Максимальное количество документов для поиска")
                .default_value("15")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("Файл для сохранения результата")
                .default_value("enhanced_article.md"),
        )
        .arg(
            Arg::new("database")
                .long("database")
                .short('d')
                .help("Путь к базе данных для персистентного хранилища (необязательно)")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("cache-days")
                .long("cache-days")
                .help("Максимальный возраст документов в кеше (дни)")
                .default_value("7")
                .value_parser(clap::value_parser!(i64)),
        )
        .arg(
            Arg::new("similarity-threshold")
                .long("similarity-threshold")
                .help("Минимальное сходство для использования кешированных запросов (0.0-1.0)")
                .default_value("0.7")
                .value_parser(clap::value_parser!(f32)),
        )
        .arg(
            Arg::new("show-cache-stats")
                .long("show-cache-stats")
                .help("Показать статистику кеша")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("cleanup-cache")
                .long("cleanup-cache")
                .help("Очистить устаревшие записи из кеша")
                .action(clap::ArgAction::SetTrue),
        )
}

pub async fn run_cli() -> Result<()> {
    let matches = cli().get_matches();

    let query = matches.get_one::<String>("query").unwrap();
    let searx_host = matches.get_one::<String>("searx-host").unwrap().clone();
    let ollama_host = matches.get_one::<String>("ollama-host").unwrap().clone();
    let model = matches.get_one::<String>("model").unwrap().clone();
    let embedding_model = matches.get_one::<String>("embedding-model").unwrap().clone();
    let max_docs = *matches.get_one::<usize>("max-docs").unwrap();
    let output = matches.get_one::<String>("output").unwrap();
    
    // Новые параметры для персистентного хранилища
    let database_path = matches.get_one::<PathBuf>("database");
    let cache_days = *matches.get_one::<i64>("cache-days").unwrap();
    let similarity_threshold = *matches.get_one::<f32>("similarity-threshold").unwrap();
    let show_stats = matches.get_flag("show-cache-stats");
    let cleanup_cache = matches.get_flag("cleanup-cache");

    info!("Параметры запуска:");
    info!("  Запрос: {}", query);
    info!("  SearXNG: {}", searx_host);
    info!("  Ollama: {}", ollama_host);
    info!("  Модель: {}", model);
    info!("  Embedding модель: {}", embedding_model);
    info!("  Макс документов: {}", max_docs);
    info!("  Выходной файл: {}", output);
    
    if let Some(db_path) = database_path {
        info!("  База данных: {:?}", db_path);
        info!("  Срок кеша: {} дней", cache_days);
        info!("  Порог сходства: {:.2}", similarity_threshold);
    } else {
        info!("  Режим: только в памяти (без персистентного хранилища)");
    }

    // Настройки кеша
    let cache_settings = CacheSettings {
        max_document_age_days: cache_days,
        min_query_similarity: similarity_threshold,
        max_cached_docs: max_docs,
        embedding_dim: None,
    };

    // Создание генератора в зависимости от наличия БД
    let mut generator = if let Some(db_path) = database_path {
        info!("Инициализация с персистентным хранилищем...");
        PersistentEnhancedRAG::new_with_persistent_storage(
            db_path,
            searx_host,
            model,
            embedding_model,
            Some(ollama_host),
            Some(cache_settings),
        )?
    } else {
        info!("Инициализация в режиме памяти...");
        PersistentEnhancedRAG::new_in_memory(
            searx_host,
            model,
            embedding_model,
            Some(ollama_host),
        )?
    };

    // Показать статистику кеша
    if show_stats {
        let stats = generator.cache_stats().await?;
        println!("\n{}", "=".repeat(50));
        println!("СТАТИСТИКА КЕША");
        println!("{}", "=".repeat(50));
        println!("Всего документов: {}", stats.total_documents);
        println!("Свежих документов: {}", stats.fresh_documents);
        println!("Всего запросов: {}", stats.total_queries);
        println!("Размер БД: {:.2} МБ", stats.database_size_mb);
        println!("{}", "=".repeat(50));
        
        if !show_stats {
            return Ok(());
        }
    }

    // Очистка кеша
    if cleanup_cache {
        println!("\nВыполняется очистка кеша...");
        let cleanup_stats = generator.cleanup_cache().await?;
        println!("Удалено документов: {}", cleanup_stats.deleted_documents);
        println!("Удалено запросов: {}", cleanup_stats.deleted_queries);
        
        if cleanup_cache && !show_stats {
            return Ok(());
        }
    }

    // Генерация статьи
    match generator.generate_article_with_cache(query, max_docs).await {
        Ok(article) => {
            println!("\n{}", "=".repeat(80));
            println!("СГЕНЕРИРОВАННАЯ СТАТЬЯ:");
            println!("{}", "=".repeat(80));
            println!("\n{}", article);

            // Сохранение в файл
            tokio::fs::write(output, &article).await?;
            info!("Статья сохранена в файл: {}", output);

            // Показываем финальную статистику кеша
            if database_path.is_some() {
                let final_stats = generator.cache_stats().await?;
                info!("Финальная статистика кеша:");
                info!("  Документов: {} (свежих: {})", 
                     final_stats.total_documents, final_stats.fresh_documents);
                info!("  Запросов: {}", final_stats.total_queries);
                info!("  Размер БД: {:.2} МБ", final_stats.database_size_mb);
            }

            Ok(())
        }
        Err(e) => {
            error!("Ошибка при генерации статьи: {}", e);
            
            // Показываем дополнительную информацию о цепочке ошибок
            let mut source = e.source();
            while let Some(err) = source {
                error!("  Причина: {}", err);
                source = err.source();
            }
            
            Err(e)
        }
    }
}