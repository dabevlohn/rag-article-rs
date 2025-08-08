use crate::{EnhancedRAGArticleGenerator, Result};
use clap::{Arg, Command};
use tracing::{info, error};

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
                .default_value("qwen3:30b"),
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

    info!("Параметры запуска:");
    info!("  Запрос: {}", query);
    info!("  SearXNG: {}", searx_host);
    info!("  Ollama: {}", ollama_host);
    info!("  Модель: {}", model);
    info!("  Embedding модель: {}", embedding_model);
    info!("  Макс документов: {}", max_docs);
    info!("  Выходной файл: {}", output);

    let mut generator = EnhancedRAGArticleGenerator::new(
        searx_host,
        model,
        embedding_model,
        Some(ollama_host),
    );

    match generator.generate_article(query, max_docs).await {
        Ok(article) => {
            println!("\n{}", "=".repeat(80));
            println!("СГЕНЕРИРОВАННАЯ СТАТЬЯ:");
            println!("{}", "=".repeat(80));
            println!("\n{}", article);

            tokio::fs::write(output, &article).await?;
            info!("Статья сохранена в файл: {}", output);

            Ok(())
        }
        Err(e) => {
            error!("Ошибка при генерации статьи: {}", e);
            Err(e)
        }
    }
}