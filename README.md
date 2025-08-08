# Enhanced RAG Article Generator (Rust)

Высокопроизводительный генератор статей с автоматическим цитированием источников, написанный на Rust. Полный аналог Python-версии с улучшенной производительностью и безопасностью типов.

## Особенности

- 🔍 **Автоматический поиск источников** через SearXNG
- 📝 **Конвертация HTML в Markdown** для эффективной токенизации
- 🧠 **Векторный поиск** с использованием embeddings через Ollama
- 📚 **Семантически правильные сноски** и цитирование
- 🚀 **Высокая производительность** благодаря асинхронной архитектуре на Rust
- 🔧 **Полная интеграция с Ollama** для локальных LLM
- 📊 **Детальные метаданные источников** с автоматическим извлечением тем

## Установка

### Предварительные требования

1. **Rust** (версия 1.70+):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. **Ollama** с необходимыми моделями:
```bash
# Установка Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Загрузка моделей
ollama pull qwen3:30b
ollama pull nomic-embed-text:latest
```

3. **SearXNG** сервер:
```bash
docker run -d -p 8080:8080 searxng/searxng
```

### Сборка проекта

```bash
# Клонирование репозитория
git clone <repository-url>
cd enhanced-rag-article-generator

# Сборка в релизном режиме
cargo build --release

# Запуск тестов
cargo test
```

## Использование

### Базовое использование

```bash
# Генерация статьи с параметрами по умолчанию
./target/release/enhanced-rag-generator "Write a comprehensive article about Rust programming language and its applications in AI development"
```

### Расширенные параметры

```bash
./target/release/enhanced-rag-generator \
  --searx-host "http://localhost:8080" \
  --ollama-host "http://localhost:11434" \
  --model "qwen3:30b" \
  --embedding-model "nomic-embed-text:latest" \
  --max-docs 20 \
  --output "my_article.md" \
  "Your comprehensive query here"
```

### Параметры командной строки

- `--searx-host`: Адрес SearXNG сервера (по умолчанию: http://127.0.0.1:8080)
- `--ollama-host`: Адрес Ollama сервера (по умолчанию: http://localhost:11434)
- `--model`, `-m`: Модель Ollama для генерации текста (по умолчанию: qwen3:30b)
- `--embedding-model`: Модель для embeddings (по умолчанию: nomic-embed-text:latest)
- `--max-docs`: Максимальное количество документов для анализа (по умолчанию: 15)
- `--output`, `-o`: Файл для сохранения результата (по умолчанию: enhanced_article.md)

## Архитектура

### Основные компоненты

1. **SearxSearchWrapper**: Интерфейс для поиска через SearXNG API
2. **RecursiveUrlLoader**: Загрузчик веб-страниц с конвертацией HTML → Markdown
3. **OllamaEmbeddings**: Генерация векторных представлений через Ollama API
4. **InMemoryVectorStore**: Векторное хранилище с косинусным поиском похожести
5. **OllamaLLM**: Интерфейс для генерации текста через Ollama
6. **EnhancedRAGArticleGenerator**: Основной класс-оркестратор

### Трейты и абстракции

```rust
#[async_trait]
pub trait SearchWrapper {
    async fn search(&self, query: &str, num_results: u32) -> Result<Vec<SearchResultItem>>;
}

#[async_trait]
pub trait DocumentLoader {
    async fn load(&self, url: &str) -> Result<Vec<Document>>;
}

#[async_trait]
pub trait EmbeddingModel {
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

#[async_trait]
pub trait LanguageModel {
    async fn generate(&self, prompt: &str) -> Result<String>;
}
```

### Процесс генерации статьи

1. **Поиск источников**: SearXNG ищет релевантные веб-страницы
2. **Загрузка контента**: Рекурсивная загрузка страниц с конвертацией HTML → Markdown  
3. **Векторизация**: Создание embeddings для семантического поиска
4. **Поиск релевантных фрагментов**: Cosine similarity для выбора лучших отрывков
5. **Генерация статьи**: LLM создает академическую статью с правильными сносками
6. **Добавление источников**: Автоматическое создание библиографии

## API и расширяемость

### Пример программного использования

```rust
use enhanced_rag_article_generator::EnhancedRAGArticleGenerator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut generator = EnhancedRAGArticleGenerator::new(
        "http://localhost:8080".to_string(),    // SearXNG
        "qwen3:30b".to_string(),                // LLM модель
        "nomic-embed-text:latest".to_string(),  // Embedding модель
        Some("http://localhost:11434".to_string()), // Ollama host
    );

    let article = generator.generate_article(
        "Comprehensive analysis of Rust vs Python for AI applications", 
        15 // max documents
    ).await?;

    println!("{}", article);
    Ok(())
}
```

### Кастомные реализации

Вы можете создать собственные реализации трейтов для интеграции с другими сервисами:

```rust
pub struct CustomSearchEngine {
    // ваша реализация
}

#[async_trait]
impl SearchWrapper for CustomSearchEngine {
    async fn search(&self, query: &str, num_results: u32) -> Result<Vec<SearchResultItem>> {
        // ваша логика поиска
    }
}
```

## Производительность

### Оптимизации

- **Асинхронная обработка**: Все I/O операции выполняются асинхронно
- **Пакетная обработка**: Embeddings создаются батчами для снижения нагрузки
- **Переиспользование соединений**: HTTP клиенты с connection pooling
- **Эффективная память**: Rust's zero-cost abstractions и ownership система

### Бенчмарки

На тестовом запросе с 15 источниками:
- **Время поиска**: ~2-3 секунды
- **Загрузка документов**: ~5-10 секунд  
- **Создание embeddings**: ~15-30 секунд (зависит от модели)
- **Генерация статьи**: ~30-60 секунд (зависит от LLM)
- **Общее время**: ~1-2 минуты

## Логирование и отладка

Включить детальное логирование:

```bash
RUST_LOG=debug ./target/release/enhanced-rag-generator "your query"
```

Уровни логирования:
- `error`: Только ошибки
- `warn`: Предупреждения и ошибки  
- `info`: Информационные сообщения (по умолчанию)
- `debug`: Детальная отладочная информация
- `trace`: Максимально детальное логирование

## Сравнение с Python версией

| Аспект | Python | Rust |
|--------|--------|------|
| **Производительность** | ~2-3 минуты | ~1-2 минуты |
| **Использование памяти** | ~500-800 MB | ~100-200 MB |
| **Безопасность типов** | Runtime ошибки | Compile-time проверки |
| **Конкурентность** | GIL limitations | True parallelism |
| **Размер бинарника** | Интерпретатор + зависимости | Один исполняемый файл |
| **Развертывание** | Python + pip | Статически слинкованный |

## Устранение неполадок

### Частые ошибки

1. **Ollama недоступен**:
```
Error: Ошибка при запросе к Ollama
```
**Решение**: Убедитесь что Ollama запущен и доступен по указанному адресу

2. **SearXNG недоступен**:
```  
Error: Ошибка при выполнении поискового запроса
```
**Решение**: Проверьте доступность SearXNG сервера

3. **Модель не найдена**:
```
Error: model not found
```
**Решение**: Загрузите необходимую модель через `ollama pull model_name`

### Отладка

```bash
# Проверка доступности сервисов
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8080           # SearXNG

# Тестирование отдельных компонентов
cargo test search_wrapper_test
cargo test document_loader_test
cargo test vector_store_test
```

## Лицензия

MIT License - подробности в файле LICENSE

## Контрибьюция

1. Форк репозитория
2. Создание feature branch (`git checkout -b feature/amazing-feature`)
3. Коммит изменений (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)  
5. Открытие Pull Request

## Roadmap

- [ ] Поддержка других поисковых движков (Google, Bing)
- [ ] Интеграция с внешними векторными БД (Pinecone, Weaviate)
- [ ] Web UI для интерактивного использования
- [ ] Поддержка других форматов экспорта (PDF, DOCX)
- [ ] Batch processing для множественной генерации
- [ ] Docker контейнеризация