use crate::ollama_utils::{parse_ollama_response, validate_environment};
use crate::persistent::*;
use anyhow::Result;
use clap::{Arg, Command};
use std::path::PathBuf;
use tracing::{error, info};

pub fn cli() -> Command {
    Command::new("enhanced-rag-generator")
        .about(
            "Enhanced RAG Article Generator - AI-powered article generation with advanced caching",
        )
        .version("2.0.0")
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
            Arg::new("quality-threshold")
                .long("quality-threshold")
                .help("Минимальный порог качества источников (0.0-1.0)")
                .default_value("0.3")
                .value_parser(clap::value_parser!(f32)),
        )
        .arg(
            Arg::new("enable-semantic")
                .long("enable-semantic")
                .help("Включить семантический поиск с embeddings")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("enable-personalization")
                .long("enable-personalization")
                .help("Включить персонализацию (экспериментально)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("show-cache-stats")
                .long("show-cache-stats")
                .help("Показать статистику кеша")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("show-quality-stats")
                .long("show-quality-stats")
                .help("Показать статистику качества источников")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("cleanup-cache")
                .long("cleanup-cache")
                .help("Очистить устаревшие записи из кеша")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("expertise-level")
                .long("expertise-level")
                .help("Уровень экспертизы пользователя")
                .value_parser(["beginner", "intermediate", "advanced", "expert"])
                .default_value("intermediate"),
        )
        .arg(
            Arg::new("validate-env")
                .long("validate-env")
                .help("Проверить окружение перед запуском")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("auto-install")
                .long("auto-install")
                .help("Автоматически установить модель если её нет")
                .action(clap::ArgAction::SetTrue),
        )
}

pub async fn run_cli() -> Result<()> {
    let matches = cli().get_matches();

    let query = matches.get_one::<String>("query").unwrap();
    let searx_host = matches.get_one::<String>("searx-host").unwrap().clone();
    let ollama_host = matches.get_one::<String>("ollama-host").unwrap().clone();
    let model = matches.get_one::<String>("model").unwrap().clone();
    let embedding_model = matches
        .get_one::<String>("embedding-model")
        .unwrap()
        .clone();
    let max_docs = *matches.get_one::<usize>("max-docs").unwrap();
    let output = matches.get_one::<String>("output").unwrap();

    // Расширенные параметры
    let database_path = matches.get_one::<PathBuf>("database");
    let cache_days = *matches.get_one::<i64>("cache-days").unwrap();
    let similarity_threshold = *matches.get_one::<f32>("similarity-threshold").unwrap();
    let quality_threshold = *matches.get_one::<f32>("quality-threshold").unwrap();
    let enable_semantic = matches.get_flag("enable-semantic");
    let enable_personalization = matches.get_flag("enable-personalization");
    let show_stats = matches.get_flag("show-cache-stats");
    let show_quality_stats = matches.get_flag("show-quality-stats");
    let cleanup_cache = matches.get_flag("cleanup-cache");
    let validate_env = matches.get_flag("validate-env");
    let auto_install = matches.get_flag("auto-install");
    let expertise_level = matches.get_one::<String>("expertise-level").unwrap();

    info!("🚀 Enhanced RAG Article Generator v2.0 - AI-Powered Edition");
    info!("Параметры запуска:");
    info!("  📝 Запрос: {}", query);
    info!("  🔍 SearXNG: {}", searx_host);
    info!("  🤖 Ollama: {}", ollama_host);
    info!("  🧠 LLM модель: {}", model);
    info!("  🎯 Embedding модель: {}", embedding_model);
    info!("  📊 Макс документов: {}", max_docs);
    info!("  💾 Выходной файл: {}", output);

    if let Some(db_path) = database_path {
        info!("  🗄️ База данных: {:?}", db_path);
        info!("  ⏰ Срок кеша: {} дней", cache_days);
        info!("  🎯 Порог сходства: {:.2}", similarity_threshold);
        info!("  ⭐ Порог качества: {:.2}", quality_threshold);
        info!(
            "  🧠 Семантический поиск: {}",
            if enable_semantic {
                "включен"
            } else {
                "отключен"
            }
        );
        info!(
            "  👤 Персонализация: {}",
            if enable_personalization {
                "включена"
            } else {
                "отключена"
            }
        );
        info!("  🎓 Уровень экспертизы: {}", expertise_level);
    } else {
        info!("  💭 Режим: только в памяти (без персистентного хранилища)");
    }

    // НОВОЕ: Валидация окружения перед началом работы
    if validate_env || auto_install {
        println!("\n{}", "=".repeat(60));
        println!("🔍 ПРОВЕРКА ОКРУЖЕНИЯ");
        println!("{}", "=".repeat(60));

        match validate_environment(&model, &ollama_host).await {
            Ok(()) => {
                info!("✅ Окружение готово к работе");
            }
            Err(e) => {
                if auto_install && e.to_string().contains("не найдена") {
                    println!("🔄 Попытка автоматической установки модели...");

                    match crate::ollama_utils::auto_install_model(&model).await {
                        Ok(()) => {
                            info!("✅ Модель успешно установлена, проверяем заново...");
                            validate_environment(&model, &ollama_host).await?;
                        }
                        Err(install_error) => {
                            error!(
                                "❌ Не удалось автоматически установить модель: {}",
                                install_error
                            );
                            return Err(e);
                        }
                    }
                } else {
                    error!("❌ {}", e);
                    return Err(e);
                }
            }
        }
    }

    // Настройки расширенного кеша
    let cache_settings = CacheSettings {
        max_document_age_days: cache_days,
        min_query_similarity: similarity_threshold,
        max_cached_docs: max_docs,
        embedding_dim: None,
        enable_semantic_search: enable_semantic,
        min_quality_score: quality_threshold,
        enable_personalization,
        auto_reindex_interval_hours: 24,
        max_vector_cache_size: 10000,
    };

    // Пользовательский контекст для персонализации
    let user_context = if enable_personalization {
        let expertise = match expertise_level.as_str() {
            "beginner" => ExpertiseLevel::Beginner,
            "intermediate" => ExpertiseLevel::Intermediate,
            "advanced" => ExpertiseLevel::Advanced,
            "expert" => ExpertiseLevel::Expert,
            _ => ExpertiseLevel::Intermediate,
        };

        Some(UserContext {
            expertise_level: expertise,
            preferred_languages: vec!["en".to_string(), "ru".to_string()],
            frequent_topics: vec![], // Будет заполняться из истории
            interaction_history: vec![chrono::Utc::now()],
        })
    } else {
        None
    };

    // Создание расширенного генератора с клонированием значений
    let mut generator = if let Some(db_path) = database_path {
        info!("🔧 Инициализация расширенного персистентного хранилища...");
        PersistentEnhancedRAG::new_with_persistent_storage(
            db_path,
            searx_host.clone(),
            model.clone(),
            embedding_model.clone(),
            Some(ollama_host.clone()),
            Some(cache_settings),
        )?
    } else {
        info!("🧠 Инициализация в режиме памяти с AI возможностями...");
        PersistentEnhancedRAG::new_in_memory(
            searx_host.clone(),
            model.clone(),
            embedding_model.clone(),
            Some(ollama_host.clone()),
        )?
    };

    // Показать статистику кеша
    if show_stats {
        let stats = generator.cache_stats().await?;
        println!("\n{}", "=".repeat(60));
        println!("📊 СТАТИСТИКА КЕША");
        println!("{}", "=".repeat(60));
        println!("📄 Всего документов: {}", stats.total_documents);
        println!("🆕 Свежих документов: {}", stats.fresh_documents);
        println!("🔍 Всего запросов: {}", stats.total_queries);
        println!("💾 Размер БД: {:.2} МБ", stats.database_size_mb);

        if database_path.is_some() {
            let cache_efficiency = if stats.total_documents > 0 {
                (stats.fresh_documents as f32 / stats.total_documents as f32) * 100.0
            } else {
                0.0
            };
            println!("⚡ Эффективность кеша: {:.1}%", cache_efficiency);
        }

        println!("{}", "=".repeat(60));

        if show_stats && !show_quality_stats && !cleanup_cache {
            return Ok(());
        }
    }

    // Показать статистику качества источников
    if show_quality_stats {
        let quality_stats = generator.get_quality_stats().await?;
        println!("\n{}", "=".repeat(60));
        println!("⭐ СТАТИСТИКА КАЧЕСТВА ИСТОЧНИКОВ");
        println!("{}", "=".repeat(60));
        println!("📊 Всего источников: {}", quality_stats.total_sources);
        println!(
            "🏆 Очень высокое качество: {}",
            quality_stats.very_high_quality
        );
        println!("✨ Высокое качество: {}", quality_stats.high_quality);
        println!("👍 Среднее качество: {}", quality_stats.medium_quality);
        println!("⚠️ Низкое качество: {}", quality_stats.low_quality);
        println!(
            "❌ Очень низкое качество: {}",
            quality_stats.very_low_quality
        );
        println!(
            "📈 Средняя оценка качества: {:.3}",
            quality_stats.average_quality_score
        );
        println!("{}", "=".repeat(60));

        if show_quality_stats && !cleanup_cache {
            return Ok(());
        }
    }

    // Очистка кеша
    if cleanup_cache {
        println!("\n🧹 Выполняется очистка кеша...");
        let cleanup_stats = generator.cleanup_cache().await?;
        println!("✅ Удалено документов: {}", cleanup_stats.deleted_documents);
        println!("✅ Удалено запросов: {}", cleanup_stats.deleted_queries);

        if cleanup_cache && !show_stats && !show_quality_stats {
            return Ok(());
        }
    }

    // Генерация статьи с расширенными возможностями
    println!("\n{}", "=".repeat(80));
    println!("🚀 ГЕНЕРАЦИЯ AI-ENHANCED СТАТЬИ");
    println!("{}", "=".repeat(80));

    let start_time = std::time::Instant::now();

    match generator
        .generate_article_with_enhanced_cache(query, max_docs, user_context)
        .await
    {
        Ok(article) => {
            let generation_time = start_time.elapsed();

            println!("\n{}", "=".repeat(80));
            println!("✨ СГЕНЕРИРОВАННАЯ AI-ENHANCED СТАТЬЯ");
            println!("{}", "=".repeat(80));
            println!("\n{}", article);

            // Сохранение в файл
            tokio::fs::write(output, &article).await?;

            println!("\n{}", "=".repeat(60));
            println!("📊 РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ");
            println!("{}", "=".repeat(60));
            println!(
                "⏱️ Время генерации: {:.2} секунд",
                generation_time.as_secs_f32()
            );
            println!("📄 Длина статьи: {} символов", article.len());
            println!("📝 Сохранено в: {}", output);

            // Показываем финальную статистику
            if database_path.is_some() {
                let final_stats = generator.cache_stats().await?;
                println!("\n🔄 ОБНОВЛЕННАЯ СТАТИСТИКА КЕША:");
                println!(
                    "  📄 Документов: {} (свежих: {})",
                    final_stats.total_documents, final_stats.fresh_documents
                );
                println!("  🔍 Запросов: {}", final_stats.total_queries);
                println!("  💾 Размер БД: {:.2} МБ", final_stats.database_size_mb);

                let quality_stats = generator.get_quality_stats().await?;
                if quality_stats.total_sources > 0 {
                    println!(
                        "  ⭐ Средняя оценка качества: {:.3}",
                        quality_stats.average_quality_score
                    );
                }
            }

            Ok(())
        }
        Err(e) => {
            let generation_time = start_time.elapsed();

            error!(
                "❌ Ошибка при генерации статьи (через {:.2}с): {}",
                generation_time.as_secs_f32(),
                e
            );

            // Показываем детальную информацию об ошибке
            let mut source = e.source();
            let mut error_chain = 1;
            while let Some(err) = source {
                error!("  📍 Причина {}: {}", error_chain, err);
                source = err.source();
                error_chain += 1;
            }

            // Диагностическая информация
            error!("🔍 ДИАГНОСТИКА:");
            error!("  🌐 SearXNG доступен: {}", searx_host);
            error!("  🤖 Ollama доступен: {}", ollama_host);
            error!("  🧠 Модель: {}", model);

            if let Some(db_path) = database_path {
                error!("  🗄️ Путь к БД: {:?}", db_path);
            }

            // Предложения по исправлению
            error!("💡 ВОЗМОЖНЫЕ РЕШЕНИЯ:");
            if e.to_string().contains("not found") {
                error!("  • Установите модель: ollama pull {}", model);
                error!("  • Или используйте другую модель с параметром --model");
            }
            if e.to_string().contains("connection") || e.to_string().contains("network") {
                error!("  • Проверьте что Ollama запущен: ollama serve");
                error!("  • Проверьте адрес Ollama: {}", ollama_host);
            }
            error!("  • Запустите с --validate-env для диагностики");
            error!("  • Запустите с --auto-install для автоустановки");

            Err(e)
        }
    }
}
