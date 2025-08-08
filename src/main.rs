use enhanced_rag_article_generator::cli::run_cli;
use anyhow::Result;
use tracing_subscriber;

// Константы приложения
const APP_NAME: &str = "Enhanced RAG Article Generator";
const VERSION: &str = "1.0.0";

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация системы логирования
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Запуск CLI интерфейса с обработкой ошибок
    match run_cli().await {
        Ok(_) => {
            println!("Приложение {} v{} завершено успешно", APP_NAME, VERSION);
            Ok(())
        }
        Err(e) => {
            eprintln!("Критическая ошибка приложения: {}", e);
            
            // Вывод цепочки причин ошибки для отладки
            let mut source = e.source();
            while let Some(err) = source {
                eprintln!("  Причина: {}", err);
                source = err.source();
            }
            
            std::process::exit(1);
        }
    }
}