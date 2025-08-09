use anyhow::Result;
use serde::Deserialize;
use reqwest;
use tracing::{info, warn, error};

/// Универсальная структура ответа Ollama API
/// Поддерживает все форматы: успешные ответы, ошибки, чат-формат
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    // Поля для успешного ответа
    pub model: Option<String>,
    pub created_at: Option<String>,
    pub response: Option<String>,        // /api/generate
    pub message: Option<ChatMessage>,    // /api/chat
    pub done: Option<bool>,
    
    // Поле для ошибок
    pub error: Option<String>,
    
    // Дополнительные поля производительности
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<i32>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<i32>,
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
}

impl OllamaResponse {
    /// Проверяет наличие ошибки в ответе
    pub fn has_error(&self) -> bool {
        self.error.is_some()
    }
    
    /// Возвращает текст ошибки если есть
    pub fn get_error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    
    /// Извлекает сгенерированный контент из любого формата
    pub fn get_content(&self) -> Result<String, String> {
        // Проверяем ошибку в первую очередь
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        
        // Проверяем формат /api/generate
        if let Some(response) = &self.response {
            return Ok(response.clone());
        }
        
        // Проверяем формат /api/chat
        if let Some(message) = &self.message {
            return Ok(message.content.clone());
        }
        
        // Если ничего не найдено
        Err("No content found in Ollama response".to_string())
    }
    
    /// Проверяет завершенность генерации
    pub fn is_done(&self) -> bool {
        self.done.unwrap_or(false)
    }
    
    /// Возвращает информацию о производительности
    pub fn get_performance_info(&self) -> Option<String> {
        if let (Some(total), Some(eval_count)) = (self.total_duration, self.eval_count) {
            let total_seconds = total as f64 / 1_000_000_000.0;
            let tokens_per_second = eval_count as f64 / total_seconds;
            Some(format!(
                "Generated {} tokens in {:.2}s ({:.1} tokens/s)",
                eval_count, total_seconds, tokens_per_second
            ))
        } else {
            None
        }
    }
}

/// Проверяет доступность модели в Ollama
pub async fn check_model_availability(model_name: &str, ollama_host: &str) -> Result<bool> {
    let url = format!("{}/api/tags", ollama_host);
    
    match reqwest::get(&url).await {
        Ok(response) => {
            let tags: OllamaTagsResponse = response.json().await?;
            Ok(tags.models.iter().any(|model| model.name == model_name))
        },
        Err(e) => {
            error!("Failed to check available models: {}", e);
            Ok(false)
        }
    }
}

/// Получает список всех доступных моделей
pub async fn get_available_models(ollama_host: &str) -> Result<Vec<String>> {
    let url = format!("{}/api/tags", ollama_host);
    
    match reqwest::get(&url).await {
        Ok(response) => {
            let tags: OllamaTagsResponse = response.json().await?;
            Ok(tags.models.into_iter().map(|model| model.name).collect())
        },
        Err(e) => {
            error!("Failed to get available models: {}", e);
            Ok(Vec::new())
        }
    }
}

/// Проверяет доступность Ollama сервера
pub async fn check_ollama_health(ollama_host: &str) -> Result<bool> {
    let health_url = format!("{}/api/tags", ollama_host);
    
    match reqwest::get(&health_url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// Валидирует окружение перед запуском генерации
pub async fn validate_environment(model_name: &str, ollama_host: &str) -> Result<()> {
    info!("🔍 Проверка окружения...");
    
    // Проверка доступности Ollama
    if !check_ollama_health(ollama_host).await? {
        return Err(anyhow::anyhow!(
            "❌ Ollama сервер недоступен по адресу: {}\n\
            Убедитесь что Ollama запущен: `ollama serve`", 
            ollama_host
        ));
    }
    info!("✅ Ollama сервер доступен");
    
    // Проверка модели
    if check_model_availability(model_name, ollama_host).await? {
        info!("✅ Модель '{}' доступна", model_name);
    } else {
        let available = get_available_models(ollama_host).await?;
        if available.is_empty() {
            return Err(anyhow::anyhow!(
                "❌ Нет установленных моделей.\n\
                Установите модель командой: ollama pull {}", 
                model_name
            ));
        } else {
            return Err(anyhow::anyhow!(
                "❌ Модель '{}' не найдена.\n\
                Доступные модели:\n  • {}\n\n\
                Установите нужную модель: ollama pull {}",
                model_name, 
                available.join("\n  • "), 
                model_name
            ));
        }
    }
    
    Ok(())
}

/// Автоматически устанавливает модель если её нет
pub async fn auto_install_model(model_name: &str) -> Result<()> {
    info!("🔄 Автоматическая установка модели {}...", model_name);
    
    let output = tokio::process::Command::new("ollama")
        .args(["pull", model_name])
        .output()
        .await?;
    
    if output.status.success() {
        info!("✅ Модель {} успешно установлена", model_name);
        Ok(())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(anyhow::anyhow!("❌ Ошибка установки модели: {}", error))
    }
}

/// Генерация с fallback на популярные модели
pub async fn generate_with_fallback<F, Fut>(
    prompt: &str, 
    primary_model: &str,
    ollama_host: &str,
    generate_fn: F
) -> Result<String> 
where
    F: Fn(&str, &str) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    // Сначала пробуем основную модель
    match generate_fn(prompt, primary_model).await {
        Ok(response) => Ok(response),
        Err(e) if e.to_string().contains("not found") => {
            warn!("Основная модель '{}' не найдена, пробуем резервные модели", primary_model);
            
            // Популярные небольшие модели для fallback
            let fallback_models = [
                "llama3.2:3b", 
                "llama3.1:8b", 
                "qwen2.5:7b",
                "phi3:mini",
                "gemma2:2b"
            ];
            
            for model in &fallback_models {
                if check_model_availability(model, ollama_host).await? {
                    match generate_fn(prompt, model).await {
                        Ok(response) => {
                            info!("✅ Успешно использована резервная модель: {}", model);
                            return Ok(response);
                        },
                        Err(e) => {
                            warn!("Резервная модель {} не сработала: {}", model, e);
                            continue;
                        }
                    }
                }
            }
            
            Err(anyhow::anyhow!(
                "❌ Ни одна модель не доступна.\n\
                Установите модель командой: ollama pull llama3.2:3b"
            ))
        },
        Err(e) => Err(e),
    }
}

/// Парсит ответ Ollama с обработкой ошибок
pub fn parse_ollama_response(response_text: &str) -> Result<String> {
    // Логируем сырой ответ для отладки
    info!("Raw Ollama response: {}", response_text);
    
    // Парсим универсальную структуру
    let ollama_response: OllamaResponse = serde_json::from_str(response_text)
        .map_err(|e| anyhow::anyhow!("Failed to parse Ollama response: {}", e))?;
    
    // Проверяем ошибки и извлекаем контент
    match ollama_response.get_content() {
        Ok(content) => {
            info!("✅ Успешно сгенерировано {} символов", content.len());
            
            // Показываем информацию о производительности если доступна
            if let Some(perf_info) = ollama_response.get_performance_info() {
                info!("📊 Производительность: {}", perf_info);
            }
            
            Ok(content)
        },
        Err(error_msg) => {
            error!("❌ Ollama API error: {}", error_msg);
            
            // Специальная обработка для конкретных ошибок
            if error_msg.contains("not found") {
                return Err(anyhow::anyhow!(
                    "Модель не найдена. Проверьте установлена ли модель: `ollama pull <model_name>`"
                ));
            } else if error_msg.contains("not loaded") {
                return Err(anyhow::anyhow!(
                    "Модель не загружена. Дождитесь загрузки модели или перезапустите Ollama"
                ));
            } else if error_msg.contains("connection") || error_msg.contains("network") {
                return Err(anyhow::anyhow!(
                    "Сетевая ошибка при обращении к Ollama. Проверьте что сервер запущен"
                ));
            } else {
                return Err(anyhow::anyhow!("Ollama error: {}", error_msg));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_success_response() {
        let json = r#"{"model":"test","response":"Hello world","done":true}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.get_content().unwrap(), "Hello world");
        assert!(!response.has_error());
    }

    #[test] 
    fn test_parse_error_response() {
        let json = r#"{"error":"model not found"}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert!(response.has_error());
        assert_eq!(response.get_error().unwrap(), "model not found");
    }

    #[test]
    fn test_parse_chat_response() {
        let json = r#"{"model":"test","message":{"role":"assistant","content":"Hello"},"done":true}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.get_content().unwrap(), "Hello");
    }
}