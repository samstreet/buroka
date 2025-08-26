# Market Analysis System - Docker Development Makefile

.PHONY: help dev-start dev-stop dev-restart dev-logs dev-build clean health-check test migrate seed-data backup-dev restore-dev

# Default target
help:
	@echo "Market Analysis System - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Docker Development:"
	@echo "  dev-start     - Start all development services"
	@echo "  dev-stop      - Stop all development services"
	@echo "  dev-restart   - Restart all development services"
	@echo "  dev-build     - Build development images"
	@echo "  dev-logs      - Show logs from all services"
	@echo ""
	@echo "Database:"
	@echo "  migrate       - Run database migrations"
	@echo "  seed-data     - Seed development database with test data"
	@echo "  backup-dev    - Backup development database"
	@echo "  restore-dev   - Restore development database"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         - Clean up containers, images, and volumes"
	@echo "  health-check  - Check health of all services"
	@echo "  test          - Run test suite"
	@echo ""
	@echo "Monitoring:"
	@echo "  logs          - Follow logs from all services"
	@echo "  diagnose      - Run diagnostic checks"

# Development environment management
dev-start:
	@echo "Starting Market Analysis development environment..."
	docker-compose up -d
	@echo "Services starting up... Run 'make health-check' to verify"
	@echo ""
	@echo "Access points:"
	@echo "  API:          http://localhost:8000"
	@echo "  API Docs:     http://localhost:8000/docs"
	@echo "  Kafka UI:     http://localhost:8080"
	@echo "  Grafana:      http://localhost:3001 (admin/admin)"
	@echo "  Prometheus:   http://localhost:9090"

dev-stop:
	@echo "Stopping Market Analysis development environment..."
	docker-compose down

dev-restart:
	@echo "Restarting Market Analysis development environment..."
	docker-compose restart

dev-build:
	@echo "Building development images..."
	docker-compose build --no-cache

test-build:
	@echo "Testing Docker build..."
	docker-compose -f docker-compose.test.yml build
	docker-compose -f docker-compose.test.yml run --rm api-test

dev-logs:
	docker-compose logs -f

# Individual service logs
logs-api:
	docker-compose logs -f api

logs-postgres:
	docker-compose logs -f postgres

logs-kafka:
	docker-compose logs -f kafka

logs-influxdb:
	docker-compose logs -f influxdb

# Health checks
health-check:
	@echo "Checking service health..."
	@echo "=========================="
	@echo ""
	@echo "PostgreSQL:"
	@docker exec market-postgres pg_isready -U trader -d market_analysis || echo "❌ PostgreSQL not ready"
	@echo ""
	@echo "InfluxDB:"
	@curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:8086/ping || echo "❌ InfluxDB not ready"
	@echo ""
	@echo "Redis:"
	@docker exec market-redis redis-cli ping || echo "❌ Redis not ready"
	@echo ""
	@echo "Kafka:"
	@docker exec market-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1 && echo "✅ Kafka ready" || echo "❌ Kafka not ready"
	@echo ""
	@echo "API:"
	@curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:8000/health || echo "❌ API not ready"

# Database operations
init-kafka:
	@echo "Initializing Kafka topics..."
	docker-compose --profile init run --rm kafka-init

test-connections:
	@echo "Testing all service connections..."
	docker-compose --profile init run --rm db-init

seed-data:
	@echo "Seeding development database..."
	docker-compose --profile init run --rm seed-data

reset-data:
	@echo "Resetting all development data..."
	docker-compose down -v
	docker-compose up -d postgres influxdb redis kafka
	@echo "Waiting for services to be ready..."
	@sleep 20
	make init-kafka
	make seed-data
	@echo "Data reset complete!"

migrate:
	@echo "Running database migrations..."
	docker exec market-api python -m alembic upgrade head

backup-dev:
	@echo "Backing up development database..."
	docker exec market-postgres pg_dump -U trader -d market_analysis > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup completed: backup_$(shell date +%Y%m%d_%H%M%S).sql"

restore-dev:
	@read -p "Enter backup file name: " backup_file; \
	echo "Restoring from $$backup_file..."; \
	docker exec -i market-postgres psql -U trader -d market_analysis < $$backup_file

# Testing
test:
	@echo "Running test suite..."
	docker exec market-api python -m pytest tests/ -v --cov

test-unit:
	@echo "Running unit tests..."
	docker exec market-api python -m pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	docker exec market-api python -m pytest tests/integration/ -v

# Cleanup
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all:
	@echo "WARNING: This will remove ALL Docker resources (not just this project)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker system prune -a -f --volumes; \
		echo "All Docker resources cleaned"; \
	else \
		echo "Cancelled"; \
	fi

# Development utilities
shell-api:
	@echo "Opening shell in API container..."
	docker exec -it market-api /bin/bash

shell-postgres:
	@echo "Opening PostgreSQL shell..."
	docker exec -it market-postgres psql -U trader -d market_analysis

shell-influx:
	@echo "Opening InfluxDB shell..."
	docker exec -it market-influxdb influx

shell-redis:
	@echo "Opening Redis shell..."
	docker exec -it market-redis redis-cli

# Monitoring and diagnostics
diagnose:
	@echo "Running diagnostics..."
	@echo "====================="
	@echo ""
	@echo "Docker containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "Docker networks:"
	@docker network ls | grep market
	@echo ""
	@echo "Docker volumes:"
	@docker volume ls | grep day-trader
	@echo ""
	@echo "Disk usage:"
	@docker system df

logs:
	@echo "Following logs from all services..."
	docker-compose logs -f --tail=100

# Production commands (for future use)
prod-build:
	@echo "Building production images..."
	docker-compose -f docker-compose.yml build --target production

prod-start:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.yml up -d

prod-stop:
	@echo "Stopping production environment..."
	docker-compose -f docker-compose.yml down

# Kafka management
kafka-topics:
	@echo "Listing Kafka topics..."
	docker exec market-kafka kafka-topics --list --bootstrap-server localhost:9092

kafka-create-topic:
	@read -p "Enter topic name: " topic; \
	docker exec market-kafka kafka-topics --create --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --topic $$topic

kafka-delete-topic:
	@read -p "Enter topic name to delete: " topic; \
	docker exec market-kafka kafka-topics --delete --bootstrap-server localhost:9092 --topic $$topic

# Performance testing
perf-test:
	@echo "Running performance tests..."
	docker exec market-api python scripts/performance_test.py

# Complete environment setup
setup:
	@echo "Setting up complete development environment..."
	python scripts/setup_dev_env.py