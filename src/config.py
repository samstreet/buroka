"""
Configuration management for Market Analysis System
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, validator
from functools import lru_cache

class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "market_analysis"
    postgres_user: str = "trader"
    postgres_password: str = "secure_password"
    
    # InfluxDB
    influxdb_host: str = "localhost" 
    influxdb_port: int = 8086
    influxdb_database: str = "market_data"
    influxdb_org: str = "market_analysis"
    influxdb_token: str = "dev_token_12345"
    influxdb_bucket: str = "market_data_dev"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def influxdb_url(self) -> str:
        """Get InfluxDB connection URL."""
        return f"http://{self.influxdb_host}:{self.influxdb_port}"
    
    class Config:
        env_prefix = ""
        case_sensitive = False

class KafkaSettings(BaseSettings):
    """Kafka configuration settings."""
    
    bootstrap_servers: str = "localhost:9092"
    topic_prefix: str = "market_"
    consumer_group_prefix: str = "market_analysis"
    
    # Topic configurations
    raw_data_topic: str = "raw_data"
    processed_data_topic: str = "processed_data"
    patterns_topic: str = "patterns"
    alerts_topic: str = "alerts"
    technical_indicators_topic: str = "technical_indicators"
    sentiment_data_topic: str = "sentiment_data"
    ml_predictions_topic: str = "ml_predictions"
    
    @validator('bootstrap_servers')
    def validate_bootstrap_servers(cls, v):
        if not v:
            raise ValueError('Bootstrap servers cannot be empty')
        return v
    
    def get_topic_name(self, topic: str) -> str:
        """Get full topic name with prefix."""
        return f"{self.topic_prefix}{topic}"
    
    def get_consumer_group(self, group: str) -> str:
        """Get consumer group name with prefix."""
        return f"{self.consumer_group_prefix}_{group}"
    
    class Config:
        env_prefix = "KAFKA_"
        case_sensitive = False

class APISettings(BaseSettings):
    """API configuration settings."""
    
    debug: bool = True
    log_level: str = "INFO"
    api_port: int = 8000
    frontend_port: int = 3000
    
    # JWT settings
    jwt_secret_key: str = "your_super_secret_jwt_key_change_this_in_production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_time: int = 86400  # 24 hours
    
    # Rate limiting
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # 1 hour
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    class Config:
        env_prefix = ""
        case_sensitive = False

class ExternalAPISettings(BaseSettings):
    """External API configuration settings."""
    
    # Stock market APIs
    alpha_vantage_api_key: str = "demo"
    polygon_api_key: Optional[str] = None
    iex_cloud_api_key: Optional[str] = None
    
    # News APIs
    news_api_key: Optional[str] = None
    
    # Social media APIs
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    
    # Cryptocurrency APIs
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None
    coinbase_api_key: Optional[str] = None
    coinbase_secret_key: Optional[str] = None
    
    class Config:
        env_prefix = ""
        case_sensitive = False

class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    prometheus_port: int = 9090
    grafana_port: int = 3001
    grafana_admin_password: str = "admin"
    
    # Metrics collection
    metrics_enabled: bool = True
    metrics_endpoint: str = "/metrics"
    
    # Health check settings
    health_check_interval: int = 30
    
    class Config:
        env_prefix = ""
        case_sensitive = False

class MLSettings(BaseSettings):
    """Machine learning configuration settings."""
    
    model_update_interval: int = 3600  # 1 hour
    model_training_enabled: bool = True
    model_artifact_path: str = "./models"
    
    # Training parameters
    train_test_split: float = 0.8
    validation_split: float = 0.2
    random_state: int = 42
    
    class Config:
        env_prefix = "ML_"
        case_sensitive = False

class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = "development"
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    kafka: KafkaSettings = KafkaSettings()
    api: APISettings = APISettings()
    external_apis: ExternalAPISettings = ExternalAPISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    ml: MLSettings = MLSettings()
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ("development", "dev", "local")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("production", "prod")

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Convenience function to get specific setting groups
def get_db_settings() -> DatabaseSettings:
    """Get database settings."""
    return get_settings().database

def get_kafka_settings() -> KafkaSettings:
    """Get Kafka settings."""
    return get_settings().kafka

def get_api_settings() -> APISettings:
    """Get API settings."""
    return get_settings().api