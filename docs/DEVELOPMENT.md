# Development Guide - Market Analysis System

## 🚀 Quick Start

The fastest way to get started:

```bash
make setup
```

This single command will:
- Build all Docker images
- Start all services  
- Initialize Kafka topics
- Seed test data
- Verify connections

## 📋 Phase 1 Complete - Docker Infrastructure

✅ **Completed Features:**
- Multi-stage Docker builds (development/production)
- Full Docker Compose orchestration
- PostgreSQL with schemas and test data
- InfluxDB v2 for time series data
- Redis for caching and sessions
- Kafka cluster with topic management
- Prometheus + Grafana monitoring
- Automated service initialization
- Health checks and logging
- Development workflow tools

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Development Environment                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐    │
│  │FastAPI  │  │Next.js  │  │Kafka UI │  │Grafana      │    │
│  │:8000    │  │:3000    │  │:8080    │  │:3001        │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                 Message Queue                           │  │
│  │  ┌──────────────┐    ┌─────────────────────────────┐    │  │
│  │  │  Zookeeper   │    │         Kafka               │    │  │
│  │  │    :2181     │◄──►│  market_dev_raw_data        │    │  │
│  │  └──────────────┘    │  market_dev_processed_data  │    │  │
│  │                      │  market_dev_patterns        │    │  │
│  │                      │  market_dev_alerts          │    │  │
│  │                      └─────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   Data Storage                          │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐    │  │
│  │  │PostgreSQL   │ │InfluxDB v2  │ │Redis            │    │  │
│  │  │:5432        │ │:8086        │ │:6379            │    │  │
│  │  │             │ │             │ │                 │    │  │
│  │  │• Users      │ │• Market Data│ │• Sessions       │    │  │
│  │  │• Patterns   │ │• Time Series│ │• Cache          │    │  │
│  │  │• Alerts     │ │• Metrics    │ │• Rate Limits    │    │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Development Workflow

### Daily Development
1. Start your development session:
   ```bash
   make dev-start
   ```

2. View logs to monitor activity:
   ```bash
   make logs
   ```

3. Make code changes (auto-reload enabled)

4. Test your changes:
   ```bash
   make test
   ```

5. Check service health:
   ```bash
   make health-check
   ```

### Common Development Tasks

**Database Operations:**
```bash
make seed-data          # Add test data
make reset-data         # Complete reset
make backup-dev         # Backup database
make shell-postgres     # Open database shell
```

**Kafka Management:**
```bash
make kafka-topics       # List topics
make init-kafka         # Initialize topics
make shell-kafka        # Access Kafka container
```

**Service Management:**
```bash
make dev-restart        # Restart all services
make logs-api           # View API logs only
make health-check       # Check all services
```

**Testing & Quality:**
```bash
make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
```

## 📊 Service Details

### PostgreSQL Database
- **Host:** localhost:5432
- **Database:** market_analysis
- **User:** trader / secure_password
- **Schemas:** 
  - `user_management` - Users, sessions, auth
  - `market_data` - Watchlists, symbols
  - `analytics` - Patterns, alerts

### InfluxDB v2
- **Host:** localhost:8086
- **Organization:** market_analysis
- **Bucket:** market_data_dev
- **Token:** dev_token_12345

### Redis
- **Host:** localhost:6379
- **Database:** 0
- **Password:** None (development)

### Kafka
- **Bootstrap Servers:** localhost:9092
- **Topics:**
  - `market_dev_raw_data` - Incoming market data
  - `market_dev_processed_data` - Cleaned data
  - `market_dev_patterns` - Detected patterns
  - `market_dev_alerts` - User alerts

## 🧪 Testing Strategy

### Test Structure
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Database + API tests  
└── e2e/           # End-to-end scenarios
```

### Running Tests
```bash
# All tests with coverage
make test

# Specific test types
pytest tests/unit/ -v
pytest tests/integration/ -v --db-url=$TEST_DB_URL

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

## 📈 Monitoring & Observability

### Grafana Dashboards
Access at http://localhost:3001 (admin/admin)

**Available Dashboards:**
- System Overview - CPU, memory, network
- API Performance - Request rates, latency
- Database Metrics - Connection pools, query times
- Kafka Monitoring - Topics, consumer lag

### Prometheus Metrics
Access at http://localhost:9090

**Key Metrics:**
- `api_request_duration_seconds` - API latency
- `kafka_consumer_lag` - Message processing lag
- `database_connections_active` - DB connection pool usage

### Logs
View aggregated logs:
```bash
make logs                # All services
make logs-api           # API only
make logs-postgres      # Database only
```

## 🔧 Configuration Management

### Environment Variables
Configuration is managed through:
- `.env` - Local overrides
- `docker/env/development.env` - Development defaults
- Container environment variables

### Key Settings
```bash
# Application
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development

# Database URLs
POSTGRES_URL=postgresql://trader:secure_password@postgres:5432/market_analysis
INFLUXDB_URL=http://influxdb:8086
REDIS_URL=redis://redis:6379/0

# Message Queue
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
KAFKA_TOPIC_PREFIX=market_dev_
```

## 🐛 Troubleshooting

### Common Issues

**Services won't start:**
```bash
make diagnose           # Check container status
make clean              # Clean everything
make setup              # Fresh setup
```

**Database connection errors:**
```bash
make health-check       # Test all connections
make shell-postgres     # Check database directly
docker logs market-postgres
```

**Kafka issues:**
```bash
make kafka-topics       # List topics
docker logs market-kafka
docker exec market-kafka kafka-topics --list --bootstrap-server localhost:9092
```

**API not responding:**
```bash
docker logs market-api
curl -v http://localhost:8000/health
make shell-api          # Debug inside container
```

### Performance Issues
1. Check container resources:
   ```bash
   docker stats
   ```

2. Monitor database queries:
   ```bash
   make shell-postgres
   SELECT * FROM pg_stat_activity;
   ```

3. Check Kafka consumer lag:
   ```bash
   # Visit Kafka UI at http://localhost:8080
   ```

## 🚀 Next Steps

**Phase 2: Pattern Recognition (Coming Next)**
- Technical indicators implementation
- Real-time pattern detection
- Statistical validation framework
- Confidence scoring system

**Preparation for Phase 2:**
```bash
# Current setup is ready for Phase 2 development
# Next: Implement technical indicators in src/core/indicators/
# Then: Build pattern detection in src/core/patterns/
```

## 📚 Additional Resources

- [PLAN.md](../PLAN.md) - Complete implementation roadmap
- [CLAUDE.md](../CLAUDE.md) - Claude development reference
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)

## 🤝 Contributing

1. Follow the development workflow above
2. Write tests for all new features
3. Update documentation
4. Ensure all health checks pass
5. Check code quality with linters

---

**Phase 1 Status: ✅ COMPLETE**

Ready to proceed to Phase 2: Pattern Recognition Engine!