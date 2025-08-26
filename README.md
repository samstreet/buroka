# Market Analysis System

A real-time market analysis platform that ingests continuous data streams from stock markets and crypto exchanges, identifies trading patterns using statistical analysis and machine learning, and provides actionable trading insights.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Git (for version control)
- Make (for running development commands)

### Development Setup

#### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd day-trader

# Run complete setup (builds containers, initializes data, starts services)
make setup
```

#### Option 2: Manual Setup
1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd day-trader
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. **Start Development Environment**
   ```bash
   make dev-start
   ```

4. **Initialize Services**
   ```bash
   make init-kafka
   make seed-data
   ```

5. **Verify Setup**
   ```bash
   make health-check
   ```

### 📊 Access Points

Once the development environment is running:

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Kafka UI**: http://localhost:8080
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

### 🛠️ Development Commands

```bash
# Start all services
make dev-start

# Stop all services
make dev-stop

# View logs
make logs

# Check service health
make health-check

# Run tests
make test

# Clean up everything
make clean
```

### 📁 Project Structure

```
day-trader/
├── src/                    # Python source code
│   ├── api/               # FastAPI application
│   ├── core/              # Core business logic
│   ├── data/              # Data layer
│   ├── ml/                # Machine learning models
│   └── utils/             # Utility functions
├── docker/                # Docker configuration files
├── tests/                 # Test files
├── scripts/               # Development scripts
├── k8s/                   # Kubernetes manifests (future)
└── docs/                  # Documentation
```

### 🐳 Docker Services

The development environment includes:

- **API Service**: FastAPI application with hot reloading
- **PostgreSQL**: User data and configuration
- **InfluxDB**: Time series market data
- **Redis**: Caching and session storage
- **Kafka + Zookeeper**: Message streaming
- **Kafka UI**: Web interface for Kafka management
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### 🔧 Configuration

Key environment variables (see `.env.example`):

```bash
# Database connections
POSTGRES_HOST=postgres
INFLUXDB_HOST=influxdb
REDIS_HOST=redis
KAFKA_BOOTSTRAP_SERVERS=kafka:29092

# API Keys (get from providers)
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key
```

### 🧪 Testing

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration

# Run with coverage
docker exec market-api python -m pytest tests/ -v --cov
```

### 📈 Monitoring

Access monitoring tools:

- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kafka UI**: http://localhost:8080

### 🐛 Troubleshooting

**Services not starting?**
```bash
make diagnose
make logs
```

**Database connection issues?**
```bash
make health-check
docker exec market-postgres pg_isready -U trader -d market_analysis
```

**Reset everything:**
```bash
make clean
make dev-start
```

### 📋 Development Workflow

1. Start development environment: `make dev-start`
2. Make code changes (auto-reload enabled)
3. Run tests: `make test`
4. Check health: `make health-check`
5. View logs: `make logs`

### 🎯 Current Phase: Phase 1 - Docker Infrastructure

We're currently implementing Phase 1 of the [detailed plan](PLAN.md):

- ✅ Docker containerization setup
- ✅ Multi-stage Dockerfile
- ✅ Docker Compose configuration
- ✅ Development environment
- ✅ Health checks and monitoring
- ⏳ Database initialization
- ⏳ Kafka topic setup

### 📚 Additional Resources

- [Detailed Implementation Plan](PLAN.md)
- [Claude Development Guide](CLAUDE.md)
- [API Documentation](http://localhost:8000/docs) (when running)

### 🤝 Contributing

1. Follow the development workflow in [CLAUDE.md](CLAUDE.md)
2. Write tests for new features
3. Ensure all health checks pass
4. Update documentation as needed

---

**Next Steps**: Complete Phase 1 Docker setup and move to data ingestion pipeline development.