# Market Analysis System - Claude Configuration

## Project Overview
Real-time market analysis platform for stock and cryptocurrency trading insights. Built with Python/FastAPI backend, Next.js frontend, containerized with Docker, and deployed on Kubernetes.

## Technology Stack
- **Backend**: Python 3.11+, FastAPI, asyncio
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, TradingView Charts
- **Message Queue**: Apache Kafka
- **Databases**: InfluxDB (time series), PostgreSQL (relational), Redis (cache)
- **ML/Analytics**: scikit-learn, TensorFlow, pandas, numpy
- **Infrastructure**: Docker, Kubernetes, Prometheus, Grafana

## Development Commands

### Docker Development
```bash
# Start all services
docker-compose up -d

# Start with rebuild
docker-compose up -d --build

# View logs
docker-compose logs -f [service_name]

# Stop all services
docker-compose down

# Reset volumes (clean slate)
docker-compose down -v
```

### Python Development
```bash
# Install dependencies
pip install -r requirements.txt
# or
poetry install

# Run tests
pytest tests/ -v --cov

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Lint code
flake8 src/ tests/
```

### Database Operations
```bash
# Connect to InfluxDB
docker exec -it influxdb influx

# Connect to PostgreSQL
docker exec -it postgres psql -U trader -d market_analysis

# Connect to Redis
docker exec -it redis redis-cli

# Run database migrations
python src/database/migrations/run_migrations.py
```

### Frontend Development
```bash
# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint and format
npm run lint
npm run format
```

## Project Structure
```
day-trader/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── routers/           # API route handlers
│   │   ├── middleware/        # Custom middleware
│   │   └── dependencies/      # Dependency injection
│   ├── core/                  # Core business logic
│   │   ├── patterns/          # Pattern detection algorithms
│   │   ├── indicators/        # Technical indicators
│   │   └── analysis/          # Market analysis engine
│   ├── data/                  # Data layer
│   │   ├── ingestion/         # Data ingestion services
│   │   ├── storage/           # Database adapters
│   │   └── models/            # Data models
│   ├── ml/                    # Machine learning models
│   │   ├── training/          # Model training scripts
│   │   ├── inference/         # Model inference
│   │   └── evaluation/        # Model evaluation
│   └── utils/                 # Utility functions
├── frontend/                  # Next.js application
│   ├── components/            # React components
│   ├── pages/                 # Next.js pages
│   ├── hooks/                 # Custom React hooks
│   └── utils/                 # Frontend utilities
├── tests/                     # Test files
├── docker/                    # Docker configurations
├── k8s/                       # Kubernetes manifests
├── docs/                      # Documentation
└── scripts/                   # Development scripts
```

## Development Workflow

### Phase 1: Docker Infrastructure (Current)
1. Set up Docker development environment
2. Configure all database containers
3. Create development workflow automation

### When Starting New Features
1. Create feature branch from `main`
2. Write tests first (TDD approach)
3. Implement feature following SOLID principles
4. Run full test suite
5. Update documentation if needed
6. Create pull request

## Testing Strategy

### Backend Testing
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# Performance tests
pytest tests/performance/ -v

# Test coverage report
pytest --cov=src --cov-report=html
```

### Frontend Testing
```bash
# Unit tests
npm run test:unit

# Component tests
npm run test:component

# E2E tests
npm run test:e2e

# Visual regression tests
npm run test:visual
```

## Key Development Guidelines

### Code Quality Standards
- Maintain 90%+ test coverage
- Follow PEP 8 for Python code
- Use TypeScript strict mode for frontend
- Implement proper error handling and logging
- Document all public APIs

### Performance Requirements
- API response times < 200ms
- Real-time updates < 50ms latency
- Handle 10,000+ requests/second in production
- Database queries optimized for sub-100ms response

### Security Requirements
- All external inputs validated and sanitized
- JWT tokens for authentication
- Rate limiting on all public endpoints
- Audit logging for sensitive operations
- No secrets in code or environment files

## Common Tasks

### Adding New Technical Indicators
1. Create indicator class in `src/core/indicators/`
2. Implement calculation logic with proper validation
3. Add comprehensive unit tests
4. Update API endpoints to expose indicator
5. Add frontend visualization if needed

### Adding New Pattern Detection
1. Create pattern detector in `src/core/patterns/`
2. Implement detection algorithm with confidence scoring
3. Add pattern to recognition engine
4. Create tests with historical data
5. Update pattern API endpoints

### Database Schema Changes
1. Create migration script in `src/database/migrations/`
2. Update data models
3. Run migration in development environment
4. Update tests to reflect schema changes
5. Document breaking changes

### Adding New Data Sources
1. Create data source adapter in `src/data/ingestion/`
2. Implement rate limiting and error handling
3. Add data validation and normalization
4. Create comprehensive tests
5. Update ingestion pipeline configuration

## Debugging and Troubleshooting

### Common Issues
```bash
# Check container status
docker ps -a

# View container logs
docker logs container_name

# Check database connections
docker exec -it postgres pg_isready

# Monitor Kafka topics
docker exec -it kafka kafka-console-consumer --topic market-data-raw --bootstrap-server localhost:9092

# Check Redis cache
docker exec -it redis redis-cli monitor
```

### Performance Monitoring
```bash
# View system metrics
docker stats

# Check API performance
curl -w "%{time_total}" http://localhost:8000/api/v1/health

# Monitor database performance
docker exec -it influxdb influx -execute "SHOW DIAGNOSTICS"
```

### Log Locations
- Application logs: `logs/app.log`
- Database logs: `logs/postgres.log`
- Kafka logs: `logs/kafka.log`
- Test logs: `tests/logs/`

## Environment Variables

### Required Environment Variables
```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=market_analysis
POSTGRES_USER=trader
POSTGRES_PASSWORD=secure_password

# InfluxDB Configuration
INFLUXDB_HOST=localhost
INFLUXDB_PORT=8086
INFLUXDB_DATABASE=market_data
INFLUXDB_USERNAME=admin
INFLUXDB_PASSWORD=admin_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=market_

# API Keys (obtain from respective providers)
ALPHA_VANTAGE_API_KEY=your_api_key_here
POLYGON_API_KEY=your_api_key_here
NEWS_API_KEY=your_api_key_here

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_TIME=86400

# Application Configuration
DEBUG=true
LOG_LEVEL=INFO
API_PORT=8000
FRONTEND_PORT=3000
```

### Development vs Production
- Development: Use `.env.dev` file
- Production: Use Kubernetes secrets and config maps
- Never commit actual API keys or secrets to repository

## Performance Benchmarks

### Target Metrics (Production)
- **API Latency**: 95th percentile < 200ms
- **Real-time Updates**: < 50ms from data ingestion to frontend
- **Throughput**: 10,000+ requests per second
- **Database Writes**: 1,000+ market data points per second
- **Pattern Detection**: < 5 seconds for full market analysis
- **Uptime**: 99.9% availability

### Load Testing Commands
```bash
# API load testing
artillery run tests/load/api-load-test.yml

# WebSocket load testing
artillery run tests/load/websocket-load-test.yml

# Database performance testing
python scripts/benchmark_database.py

# End-to-end performance testing
python scripts/benchmark_e2e.py
```

## Deployment

### Development Deployment
```bash
# Start development environment
make dev-start

# Run database migrations
make migrate

# Seed test data
make seed-data

# Run full test suite
make test-all
```

### Production Deployment
```bash
# Build production images
make build-prod

# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n market-analysis

# View application logs
kubectl logs -f deployment/market-analysis-api -n market-analysis
```

## Monitoring and Alerts

### Key Metrics to Monitor
- API response times and error rates
- Database connection pool utilization
- Kafka message lag and throughput
- Memory and CPU usage per container
- Pattern detection accuracy and performance
- WebSocket connection stability

### Alert Thresholds
- API 95th percentile latency > 500ms
- Error rate > 1% over 5 minutes
- Database connections > 80% of pool
- Kafka consumer lag > 10,000 messages
- Memory usage > 85% of container limit
- Pattern detection accuracy < 60%

### Grafana Dashboards
- System Overview: CPU, memory, network, disk usage
- API Performance: Request rates, latency, errors
- Database Performance: Query times, connection pools
- Business Metrics: Pattern detection rates, user activity

## Additional Resources

### Documentation
- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Database Schema](docs/database.md)
- [Deployment Guide](docs/deployment.md)

### External APIs
- [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)
- [Polygon.io API Docs](https://polygon.io/docs)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)

### Learning Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## Contributing Guidelines

### Code Review Checklist
- [ ] Tests pass and coverage maintained
- [ ] Code follows project style guidelines
- [ ] API changes are documented
- [ ] Performance impact assessed
- [ ] Security implications considered
- [ ] Breaking changes documented

### Git Workflow
- Use conventional commit messages
- Create feature branches for new development
- Squash commits before merging
- Tag releases with semantic versioning
- Maintain linear git history when possible

## Support and Troubleshooting

### Common Commands for Issues
```bash
# Reset development environment
make clean && make dev-start

# Check all service health
make health-check

# View real-time logs
make logs

# Run diagnostic tests
make diagnose

# Backup development data
make backup-dev

# Restore from backup
make restore-dev
```

### When to Scale Services
- **API Service**: Scale when CPU > 70% or response time > 300ms
- **Pattern Detection**: Scale when queue depth > 1000 messages
- **Database**: Scale reads when connection pool > 80% utilized
- **Frontend**: Scale when CDN cache hit rate < 90%

This configuration provides comprehensive guidance for development, testing, deployment, and maintenance of the market analysis system.