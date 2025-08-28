# Market Analysis System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Market Analysis System in various environments, from local development to production deployments. The system uses Docker and Docker Compose for containerization and can be deployed on Kubernetes for production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Local Development Deployment](#local-development-deployment)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Environment Configuration](#environment-configuration)
- [Database Setup](#database-setup)
- [Service Health Checks](#service-health-checks)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 4 cores (2.0 GHz)
- **RAM**: 8 GB
- **Storage**: 50 GB available space
- **Network**: Stable internet connection for API data sources

**Recommended Requirements:**
- **CPU**: 8 cores (2.5 GHz)
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **Network**: High-bandwidth connection for real-time data

### Software Dependencies

**Required Software:**
- **Docker**: Version 20.10.0 or higher
- **Docker Compose**: Version 2.0.0 or higher
- **Git**: Version 2.20.0 or higher
- **Python**: Version 3.11 or higher (for development)
- **Node.js**: Version 16.0.0 or higher (for frontend, if applicable)

### Installation Commands

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose-plugin git python3.11 python3-pip nodejs npm

# CentOS/RHEL
sudo yum install docker docker-compose git python3.11 python3-pip nodejs npm
sudo systemctl start docker
sudo systemctl enable docker

# macOS (using Homebrew)
brew install docker docker-compose git python@3.11 node

# Windows (using Chocolatey)
choco install docker-desktop docker-compose git python nodejs
```

## Environment Setup

### Development Environment Variables

Create environment files in the `docker/env/` directory:

**File: `docker/env/development.env`**
```bash
# Application Settings
DEBUG=true
LOG_LEVEL=DEBUG
API_PORT=8000
RELOAD=true

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=market_analysis
POSTGRES_USER=trader
POSTGRES_PASSWORD=secure_password

# InfluxDB Configuration
INFLUXDB_HOST=influxdb
INFLUXDB_PORT=8086
INFLUXDB_TOKEN=dev_token_12345
INFLUXDB_ORG=market_analysis
INFLUXDB_BUCKET=market_data_dev

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
KAFKA_TOPIC_PREFIX=market_dev_

# External API Keys (obtain from providers)
ALPHA_VANTAGE_API_KEY=demo
POLYGON_API_KEY=demo
NEWS_API_KEY=demo

# JWT Configuration
JWT_SECRET_KEY=your_super_secret_jwt_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_TIME=1800

# Security Settings
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS_PER_HOUR=1000
ENABLE_AUDIT_LOGGING=true
HTTPS_ONLY=false

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin
```

### Production Environment Variables

**File: `docker/env/production.env`**
```bash
# Application Settings
DEBUG=false
LOG_LEVEL=INFO
API_PORT=8000
RELOAD=false

# Database Configuration (use strong passwords)
POSTGRES_HOST=postgres-prod
POSTGRES_PORT=5432
POSTGRES_DB=market_analysis_prod
POSTGRES_USER=trader_prod
POSTGRES_PASSWORD=STRONG_RANDOM_PASSWORD_HERE

# InfluxDB Configuration
INFLUXDB_HOST=influxdb-prod
INFLUXDB_PORT=8086
INFLUXDB_TOKEN=SECURE_PRODUCTION_TOKEN
INFLUXDB_ORG=market_analysis_prod
INFLUXDB_BUCKET=market_data_prod

# Redis Configuration
REDIS_HOST=redis-prod
REDIS_PORT=6379
REDIS_PASSWORD=REDIS_STRONG_PASSWORD
REDIS_DB=0

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka-cluster:9092
KAFKA_TOPIC_PREFIX=market_prod_

# External API Keys (production keys)
ALPHA_VANTAGE_API_KEY=YOUR_PROD_API_KEY
POLYGON_API_KEY=YOUR_PROD_API_KEY
NEWS_API_KEY=YOUR_PROD_API_KEY

# JWT Configuration (use strong secret)
JWT_SECRET_KEY=EXTREMELY_STRONG_RANDOM_SECRET_KEY
JWT_ALGORITHM=HS256
JWT_EXPIRATION_TIME=1800

# Security Settings
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS_PER_HOUR=5000
ENABLE_AUDIT_LOGGING=true
HTTPS_ONLY=true
SSL_CERT_PATH=/etc/ssl/certs/market-api.crt
SSL_KEY_PATH=/etc/ssl/private/market-api.key

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=SECURE_GRAFANA_PASSWORD
```

## Local Development Deployment

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd day-trader

# 2. Copy environment template
cp docker/env/development.env.template docker/env/development.env

# 3. Start all services
docker-compose up -d

# 4. Initialize the system (run once)
docker-compose --profile init up --build

# 5. Check service health
curl http://localhost:8000/api/v1/health
```

### Step-by-Step Development Setup

#### 1. Repository Setup
```bash
# Clone repository
git clone <repository-url>
cd day-trader

# Create necessary directories
mkdir -p logs docker/env data

# Set proper permissions
chmod +x scripts/*.py
chmod 755 docker/postgres/init*.sql
```

#### 2. Environment Configuration
```bash
# Create development environment file
cp docker/env/development.env.template docker/env/development.env

# Edit configuration as needed
nano docker/env/development.env
```

#### 3. Start Infrastructure Services
```bash
# Start databases and message queue
docker-compose up -d postgres influxdb redis kafka

# Wait for services to be healthy (check with docker ps)
docker-compose ps

# Verify connectivity
docker-compose --profile init run --rm db-init
```

#### 4. Start Application Services
```bash
# Start API and monitoring
docker-compose up -d api prometheus grafana

# Initialize Kafka topics
docker-compose --profile init run --rm kafka-init

# Seed development data (optional)
docker-compose --profile init run --rm seed-data
```

#### 5. Verification
```bash
# Check all services are running
docker-compose ps

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/health

# Access monitoring interfaces
open http://localhost:3001  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
open http://localhost:8080  # Kafka UI
```

### Development Workflow Commands

```bash
# View logs from all services
docker-compose logs -f

# View logs from specific service
docker-compose logs -f api

# Restart a service
docker-compose restart api

# Rebuild and restart service
docker-compose up -d --build api

# Stop all services
docker-compose down

# Remove all data (clean slate)
docker-compose down -v
```

## Docker Compose Deployment

### Service Architecture

The Docker Compose setup includes the following services:

#### Core Services
- **postgres**: PostgreSQL database for relational data
- **influxdb**: InfluxDB for time-series market data
- **redis**: Redis for caching and session storage
- **kafka**: Apache Kafka for message queuing

#### Application Services  
- **api**: FastAPI application server
- **frontend**: Next.js frontend (if implemented)

#### Monitoring Services
- **prometheus**: Metrics collection and monitoring
- **grafana**: Data visualization and dashboards
- **kafka-ui**: Kafka topic management interface

#### Initialization Services
- **kafka-init**: One-time Kafka topic creation
- **db-init**: Database connection testing
- **seed-data**: Development data seeding

### Service Dependencies

Services are configured with health checks and dependencies:

```yaml
depends_on:
  postgres:
    condition: service_healthy
  influxdb:
    condition: service_healthy
  redis:
    condition: service_healthy
  kafka:
    condition: service_healthy
```

### Network Configuration

All services use a custom bridge network (`market_network`) with:
- **Subnet**: 172.20.0.0/16
- **Internal Communication**: Services communicate by container name
- **External Access**: Selected ports exposed to host

### Volume Management

Data persistence through named volumes:
- `postgres_data`: PostgreSQL database files
- `influxdb_data`: InfluxDB time-series data
- `redis_data`: Redis cache data
- `kafka_data`: Kafka topic data
- `prometheus_data`: Prometheus metrics data
- `grafana_data`: Grafana dashboards and settings

## Environment Configuration

### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **Environment Files** (`.env` files)
3. **Configuration Classes** (`src/config.py`)
4. **Default Values** (lowest priority)

### Database Configuration

#### PostgreSQL Setup
```bash
# Connection parameters
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=market_analysis
POSTGRES_USER=trader
POSTGRES_PASSWORD=secure_password

# Connection pool settings
POSTGRES_MIN_CONNECTIONS=1
POSTGRES_MAX_CONNECTIONS=20
POSTGRES_CONNECTION_TIMEOUT=10
```

#### InfluxDB Setup
```bash
# InfluxDB 2.x configuration
INFLUXDB_HOST=influxdb
INFLUXDB_PORT=8086
INFLUXDB_TOKEN=dev_token_12345
INFLUXDB_ORG=market_analysis
INFLUXDB_BUCKET=market_data_dev

# Retention policies
INFLUXDB_RETENTION_PERIOD=30d
INFLUXDB_PRECISION=ms
```

#### Redis Configuration
```bash
# Redis connection
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=optional_password
REDIS_DB=0

# Connection pool settings
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5
REDIS_CONNECTION_TIMEOUT=10
```

### External API Configuration

#### Market Data Providers
```bash
# Alpha Vantage (free tier: 5 requests/minute, 500/day)
ALPHA_VANTAGE_API_KEY=your_api_key
ALPHA_VANTAGE_BASE_URL=https://www.alphavantage.co/query

# Polygon.io (paid service)
POLYGON_API_KEY=your_api_key
POLYGON_BASE_URL=https://api.polygon.io

# News API
NEWS_API_KEY=your_api_key
NEWS_API_BASE_URL=https://newsapi.org/v2
```

## Database Setup

### PostgreSQL Initialization

The PostgreSQL database is automatically initialized with:

**File: `docker/postgres/init-dev.sql`**
```sql
-- Create application database
CREATE DATABASE market_analysis;

-- Create user with appropriate permissions
CREATE USER trader WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE market_analysis TO trader;

-- Connect to the new database
\c market_analysis;

-- Create schema for market data
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS user_management;

-- Grant schema permissions
GRANT ALL ON SCHEMA market_data TO trader;
GRANT ALL ON SCHEMA analytics TO trader;
GRANT ALL ON SCHEMA user_management TO trader;

-- Create basic tables
CREATE TABLE IF NOT EXISTS market_data.symbols (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    exchange VARCHAR(50),
    sector VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_management.users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO market_data.symbols (symbol, name, exchange, sector) VALUES
    ('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology'),
    ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology'),
    ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology')
ON CONFLICT (symbol) DO NOTHING;
```

### InfluxDB Setup

InfluxDB 2.x is configured with:
- **Organization**: market_analysis
- **Bucket**: market_data_dev (development)
- **Token**: dev_token_12345 (development)
- **Retention**: 30 days (configurable)

### Database Migrations

```bash
# Run database migrations (when implemented)
docker-compose run --rm api python src/database/migrations/run_migrations.py

# Seed development data
docker-compose --profile init run --rm seed-data

# Test database connections
docker-compose --profile init run --rm db-init
```

## Service Health Checks

### Health Check Configuration

Each service includes health checks to ensure proper startup ordering:

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U trader -d market_analysis"]
  interval: 10s
  timeout: 5s
  retries: 5
```

### Manual Health Verification

```bash
# Check individual service health
docker-compose ps  # View service status

# Test PostgreSQL
docker exec -it market-postgres pg_isready -U trader -d market_analysis

# Test InfluxDB
docker exec -it market-influxdb influx ping

# Test Redis
docker exec -it market-redis redis-cli ping

# Test Kafka
docker exec -it market-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

### Application Health Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Comprehensive health check
curl http://localhost:8000/api/v1/health

# Component-specific health
curl http://localhost:8000/api/v1/storage/health
curl http://localhost:8000/api/v1/monitoring/health
```

## Production Deployment

### Pre-Deployment Checklist

#### Security Review
- [ ] Change all default passwords
- [ ] Generate strong JWT secrets
- [ ] Configure SSL/TLS certificates
- [ ] Review firewall rules
- [ ] Enable audit logging
- [ ] Configure rate limiting

#### Configuration Review
- [ ] Set DEBUG=false
- [ ] Configure production database credentials
- [ ] Set up external API keys
- [ ] Configure monitoring alerts
- [ ] Set up log aggregation
- [ ] Configure backup procedures

#### Infrastructure Review
- [ ] Provision adequate resources
- [ ] Set up load balancing
- [ ] Configure reverse proxy
- [ ] Set up container orchestration
- [ ] Configure persistent storage
- [ ] Set up disaster recovery

### Docker Swarm Deployment

```bash
# Initialize swarm mode
docker swarm init

# Create production secrets
echo "strong_postgres_password" | docker secret create postgres_password -
echo "secure_jwt_secret" | docker secret create jwt_secret -

# Deploy stack
docker stack deploy -c docker-compose.prod.yml market-analysis

# Monitor deployment
docker stack services market-analysis
docker stack ps market-analysis
```

### Kubernetes Deployment

**File: `k8s/namespace.yaml`**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: market-analysis
  labels:
    name: market-analysis
```

**File: `k8s/configmap.yaml`**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: market-config
  namespace: market-analysis
data:
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  POSTGRES_HOST: "postgres-service"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "market_analysis"
```

**Deployment Commands:**
```bash
# Create namespace and resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy services
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/api.yaml

# Check deployment status
kubectl get pods -n market-analysis
kubectl get services -n market-analysis
```

### Reverse Proxy Configuration

**Nginx Configuration:**
```nginx
upstream market_api {
    server localhost:8000;
}

server {
    listen 443 ssl http2;
    server_name api.market-analysis.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    location / {
        proxy_pass http://market_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring and Observability

### Prometheus Configuration

**File: `docker/prometheus/prometheus.dev.yml`**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'market-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres:5432']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Grafana Dashboards

Pre-configured dashboards include:
- **System Overview**: CPU, memory, network, disk usage
- **API Performance**: Request rates, latency, error rates
- **Database Performance**: Connection pools, query times
- **Business Metrics**: Market data ingestion rates, user activity

### Log Aggregation

```bash
# Centralized logging with ELK stack
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch:7.14.0

docker run -d \
  --name logstash \
  -p 5044:5044 \
  -v ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf \
  logstash:7.14.0

docker run -d \
  --name kibana \
  -p 5601:5601 \
  -e "ELASTICSEARCH_HOSTS=http://elasticsearch:9200" \
  kibana:7.14.0
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Service Won't Start

**Problem**: Container exits immediately
```bash
# Check container logs
docker-compose logs service-name

# Common issues:
# - Port already in use
# - Missing environment variables
# - Invalid configuration
# - Permission issues
```

**Solutions**:
```bash
# Check port usage
netstat -tlnp | grep :8000

# Verify environment variables
docker-compose config

# Fix permissions
sudo chown -R $USER:$USER logs/
chmod +x scripts/*.py
```

#### 2. Database Connection Issues

**Problem**: Cannot connect to PostgreSQL/InfluxDB
```bash
# Check database health
docker-compose ps postgres influxdb

# Test connectivity
docker exec -it market-postgres pg_isready -U trader
docker exec -it market-influxdb influx ping
```

**Solutions**:
```bash
# Restart database services
docker-compose restart postgres influxdb

# Check network connectivity
docker network ls
docker network inspect day-trader_market_network

# Verify credentials
cat docker/env/development.env | grep POSTGRES
```

#### 3. API Returns 500 Errors

**Problem**: Internal server errors in API responses
```bash
# Check API logs
docker-compose logs -f api

# Common causes:
# - Database connection failure
# - Missing environment variables
# - Import errors
# - Configuration issues
```

**Solutions**:
```bash
# Restart API service
docker-compose restart api

# Check service dependencies
curl http://localhost:8000/api/v1/health

# Verify all environment variables
docker-compose exec api env | grep -E "(POSTGRES|INFLUXDB|REDIS)"
```

#### 4. High Memory Usage

**Problem**: System running out of memory
```bash
# Check resource usage
docker stats

# Identify memory-hungry containers
docker system df
docker system prune
```

**Solutions**:
```bash
# Limit container resources
docker-compose up -d --compatibility  # Uses deploy.resources limits

# Clean up unused resources
docker system prune -a
docker volume prune
```

#### 5. Kafka Issues

**Problem**: Kafka won't start or topics not created
```bash
# Check Kafka logs
docker-compose logs kafka

# Verify Kafka is running
docker exec -it market-kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Solutions**:
```bash
# Restart Kafka
docker-compose restart kafka

# Manually create topics
docker-compose --profile init run --rm kafka-init

# Check Kafka UI
open http://localhost:8080
```

### Performance Optimization

#### Database Optimization
```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();

-- Create indexes for frequently queried columns
CREATE INDEX idx_market_data_symbol_timestamp ON market_data.ohlc_data(symbol, timestamp);
CREATE INDEX idx_users_email ON user_management.users(email);
```

#### API Performance
```bash
# Increase worker processes
UVICORN_WORKERS=4 docker-compose up -d api

# Enable response compression
ENABLE_GZIP=true docker-compose up -d api

# Tune connection pools
POSTGRES_MAX_CONNECTIONS=20 docker-compose up -d api
```

#### Caching Strategy
```bash
# Redis cache optimization
REDIS_MAX_MEMORY=512mb
REDIS_MAX_MEMORY_POLICY=allkeys-lru
REDIS_SAVE=""  # Disable persistence for cache-only use
```

### Backup and Recovery

#### Database Backup
```bash
# PostgreSQL backup
docker exec market-postgres pg_dump -U trader market_analysis > backup_$(date +%Y%m%d_%H%M%S).sql

# InfluxDB backup
docker exec market-influxdb influx backup /backup --bucket market_data_dev --token dev_token_12345

# Automated backup script
./scripts/backup_databases.sh
```

#### Volume Backup
```bash
# Create volume backups
docker run --rm -v day-trader_postgres_data:/data -v $(pwd)/backups:/backup ubuntu tar czf /backup/postgres_data_$(date +%Y%m%d).tar.gz -C /data .

# Restore volume from backup
docker run --rm -v day-trader_postgres_data:/data -v $(pwd)/backups:/backup ubuntu tar xzf /backup/postgres_data_20240115.tar.gz -C /data
```

### Scaling Considerations

#### Horizontal Scaling
```bash
# Scale API service
docker-compose up -d --scale api=3

# Load balancer configuration (nginx)
upstream market_api_cluster {
    least_conn;
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

#### Database Scaling
```bash
# PostgreSQL read replicas
# InfluxDB clustering
# Redis clustering
```

This deployment guide provides comprehensive instructions for setting up the Market Analysis System in various environments. For additional support or specific deployment scenarios, refer to the project documentation or contact the development team.