# Market Analysis System - Testing Strategy and Guidelines

## Overview

This document outlines the comprehensive testing strategy for the Market Analysis System, covering unit tests, integration tests, performance tests, and end-to-end testing. The testing approach follows Test-Driven Development (TDD) principles and ensures high code quality, reliability, and maintainability.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Testing Pyramid](#testing-pyramid)
- [Test Structure](#test-structure)
- [Testing Tools and Frameworks](#testing-tools-and-frameworks)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Performance Testing](#performance-testing)
- [Security Testing](#security-testing)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)
- [Testing Best Practices](#testing-best-practices)

## Testing Philosophy

### Core Principles

1. **Test-Driven Development (TDD)**
   - Write tests before implementation
   - Follow Red-Green-Refactor cycle
   - Ensure tests drive design decisions

2. **Comprehensive Coverage**
   - Maintain >90% code coverage
   - Test both happy path and edge cases
   - Include error handling scenarios

3. **Fast and Reliable**
   - Tests should run quickly (< 10 minutes for full suite)
   - Tests should be deterministic and not flaky
   - Isolated tests with no dependencies

4. **Maintainable Tests**
   - Clear test names and documentation
   - DRY principle applied to test code
   - Easy to understand and modify

## Testing Pyramid

The testing strategy follows the standard testing pyramid:

```
                    ┌─────────────────────┐
                    │   E2E Tests (5%)    │
                    │                     │
                    │ • Full system tests │
                    │ • User scenarios    │
                    │ • API workflows     │
                    └─────────────────────┘
                            │
              ┌─────────────────────────────────┐
              │    Integration Tests (25%)      │
              │                                 │
              │ • Service interactions          │
              │ • Database operations           │
              │ • External API mocks           │
              │ • Component integration         │
              └─────────────────────────────────┘
                            │
    ┌─────────────────────────────────────────────────────┐
    │              Unit Tests (70%)                       │
    │                                                     │
    │ • Individual functions and methods                  │
    │ • Business logic validation                         │
    │ • Data transformation                               │
    │ • Error handling                                    │
    │ • Edge cases and boundary conditions               │
    └─────────────────────────────────────────────────────┘
```

## Test Structure

### Directory Structure

```
tests/
├── unit/                          # Unit tests
│   ├── core/                      # Core business logic tests
│   │   ├── test_indicators.py     # Technical indicators tests
│   │   ├── test_patterns.py       # Pattern detection tests
│   │   └── test_analysis.py       # Analysis engine tests
│   ├── data/                      # Data layer tests
│   │   ├── test_ingestion_service.py
│   │   ├── test_storage_service.py
│   │   ├── test_validation.py
│   │   └── test_market_data_models.py
│   ├── api/                       # API layer tests
│   │   ├── test_auth.py
│   │   ├── test_market_data_routes.py
│   │   └── test_middleware.py
│   └── security/                  # Security tests
│       ├── test_audit_logging.py
│       ├── test_input_validation.py
│       └── test_security_middleware.py
├── integration/                   # Integration tests
│   ├── test_api_endpoints.py      # Full API integration
│   ├── test_database_operations.py
│   ├── test_external_apis.py
│   └── test_message_queue.py
├── e2e/                          # End-to-end tests
│   ├── test_user_workflows.py
│   ├── test_data_pipeline.py
│   └── test_system_monitoring.py
├── performance/                   # Performance tests
│   ├── test_api_load.py
│   ├── test_database_performance.py
│   └── test_concurrent_users.py
├── fixtures/                      # Test data and fixtures
│   ├── sample_market_data.json
│   ├── test_users.json
│   └── mock_responses/
├── conftest.py                   # Pytest configuration
├── requirements-test.txt         # Test dependencies
└── README.md                     # Testing documentation
```

## Testing Tools and Frameworks

### Core Testing Stack

#### pytest
Primary testing framework with powerful features:
- Fixture system for test setup/teardown
- Parametrized testing
- Plugin ecosystem
- Clear assertion messages

```python
# Example pytest usage
import pytest
from src.core.indicators.moving_averages import SimpleMovingAverage

class TestSimpleMovingAverage:
    @pytest.fixture
    def sample_prices(self):
        return [10, 12, 13, 10, 11, 14, 15, 16, 18, 17]
    
    @pytest.mark.parametrize("period,expected", [
        (3, [11.67, 11.67, 11.33, 11.67, 13.33, 15.0, 16.33, 17.0]),
        (5, [11.2, 12.0, 12.8, 13.6, 14.8, 16.0])
    ])
    def test_calculate_sma(self, sample_prices, period, expected):
        sma = SimpleMovingAverage(period=period)
        result = sma.calculate(sample_prices)
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected):
            assert abs(actual - exp) < 0.01
```

#### pytest-asyncio
For testing async code:
```python
import pytest
from src.data.ingestion.service import DataIngestionService

@pytest.mark.asyncio
async def test_async_data_ingestion():
    service = DataIngestionService()
    result = await service.ingest_symbol_data("AAPL", "daily")
    assert result["success"] is True
```

#### pytest-mock
For mocking dependencies:
```python
def test_external_api_call(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"symbol": "AAPL", "price": 150.0}
    mocker.patch('requests.get', return_value=mock_response)
    
    result = get_stock_price("AAPL")
    assert result == 150.0
```

### Additional Testing Tools

#### Factory Boy
For test data generation:
```python
import factory
from src.data.models.market_data import OHLCData

class OHLCDataFactory(factory.Factory):
    class Meta:
        model = OHLCData
    
    symbol = factory.Sequence(lambda n: f"STOCK{n}")
    timestamp = factory.Faker('date_time')
    open_price = factory.Faker('pydecimal', left_digits=3, right_digits=2, positive=True)
    high_price = factory.LazyAttribute(lambda obj: obj.open_price * 1.1)
    low_price = factory.LazyAttribute(lambda obj: obj.open_price * 0.9)
    close_price = factory.Faker('pydecimal', left_digits=3, right_digits=2, positive=True)
    volume = factory.Faker('pyint', min_value=1000, max_value=1000000)
```

#### Faker
For generating realistic test data:
```python
from faker import Faker

fake = Faker()

def test_user_creation():
    user_data = {
        "email": fake.email(),
        "username": fake.user_name(),
        "full_name": fake.name(),
        "password": fake.password(length=12)
    }
    # Test user creation logic
```

## Unit Testing

### Testing Guidelines

#### 1. Test Structure (AAA Pattern)
```python
def test_calculate_rsi():
    # Arrange - Set up test data and dependencies
    prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89]
    rsi_calculator = RSI(period=14)
    
    # Act - Execute the function being tested
    result = rsi_calculator.calculate(prices)
    
    # Assert - Verify the results
    assert len(result) > 0
    assert all(0 <= value <= 100 for value in result)
    assert abs(result[-1] - 70.53) < 0.1  # Expected RSI value
```

#### 2. Edge Cases and Error Handling
```python
class TestMarketDataValidation:
    def test_empty_price_list(self):
        validator = MarketDataValidator()
        with pytest.raises(ValidationError, match="Price list cannot be empty"):
            validator.validate_prices([])
    
    def test_negative_prices(self):
        validator = MarketDataValidator()
        with pytest.raises(ValidationError, match="Prices must be positive"):
            validator.validate_prices([10, -5, 15])
    
    def test_invalid_symbol_format(self):
        validator = MarketDataValidator()
        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validator.validate_symbol("")
```

#### 3. Parameterized Tests
```python
@pytest.mark.parametrize("symbol,expected_valid", [
    ("AAPL", True),
    ("MSFT", True),
    ("GOOGL", True),
    ("A", False),  # Too short
    ("TOOLONGNAME", False),  # Too long
    ("123", False),  # Numbers only
    ("", False),  # Empty
])
def test_symbol_validation(symbol, expected_valid):
    validator = SymbolValidator()
    result = validator.is_valid(symbol)
    assert result == expected_valid
```

### Testing Core Components

#### Technical Indicators Testing
```python
class TestTechnicalIndicators:
    @pytest.fixture
    def sample_ohlc_data(self):
        return [
            OHLCData(symbol="AAPL", open=100, high=105, low=95, close=102, volume=1000000),
            OHLCData(symbol="AAPL", open=102, high=108, low=98, close=104, volume=1200000),
            # ... more data
        ]
    
    def test_bollinger_bands_calculation(self, sample_ohlc_data):
        bb = BollingerBands(period=20, std_dev=2)
        result = bb.calculate(sample_ohlc_data)
        
        assert "upper_band" in result
        assert "middle_band" in result
        assert "lower_band" in result
        assert len(result["upper_band"]) == len(sample_ohlc_data) - 19
        
        # Verify upper band > middle band > lower band
        for i in range(len(result["upper_band"])):
            assert result["upper_band"][i] > result["middle_band"][i]
            assert result["middle_band"][i] > result["lower_band"][i]
```

#### Data Ingestion Testing
```python
class TestDataIngestionService:
    @pytest.fixture
    def mock_data_source(self, mocker):
        mock = mocker.Mock()
        mock.fetch_daily_data.return_value = {
            "symbol": "AAPL",
            "data": [{"date": "2024-01-15", "close": 150.0}]
        }
        return mock
    
    def test_successful_data_ingestion(self, mock_data_source):
        service = DataIngestionService(data_source=mock_data_source)
        result = service.ingest_symbol_data("AAPL", MarketDataType.DAILY)
        
        assert result["success"] is True
        assert result["records_count"] > 0
        mock_data_source.fetch_daily_data.assert_called_once_with("AAPL")
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_data_source):
        service = DataIngestionService(
            data_source=mock_data_source,
            rate_limit=1  # 1 request per second
        )
        
        start_time = time.time()
        await service.ingest_symbol_data("AAPL", MarketDataType.DAILY)
        await service.ingest_symbol_data("MSFT", MarketDataType.DAILY)
        end_time = time.time()
        
        # Should take at least 1 second due to rate limiting
        assert end_time - start_time >= 1.0
```

## Integration Testing

### Database Integration Tests

#### PostgreSQL Integration
```python
import pytest
from sqlalchemy import create_engine
from src.database.models import User, Symbol
from src.database.session import SessionLocal

@pytest.fixture(scope="function")
def db_session():
    # Create test database session
    engine = create_engine("postgresql://test_user:test_pass@localhost/test_db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.rollback()
    session.close()

class TestUserRepository:
    def test_create_user(self, db_session):
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password_hash": "hashed_password"
        }
        
        user = User(**user_data)
        db_session.add(user)
        db_session.commit()
        
        # Verify user was created
        retrieved_user = db_session.query(User).filter_by(username="testuser").first()
        assert retrieved_user is not None
        assert retrieved_user.email == "test@example.com"
```

#### InfluxDB Integration
```python
import pytest
from influxdb_client import InfluxDBClient
from src.data.storage.influxdb_sink import InfluxDBDataSink

@pytest.fixture
def influxdb_client():
    client = InfluxDBClient(
        url="http://localhost:8086",
        token="test_token",
        org="test_org"
    )
    
    yield client
    
    # Cleanup test data
    delete_api = client.delete_api()
    delete_api.delete(
        start="1970-01-01T00:00:00Z",
        stop="2030-01-01T00:00:00Z",
        predicate='_measurement="test_market_data"',
        bucket="test_bucket"
    )
    client.close()

class TestInfluxDBSink:
    def test_write_market_data(self, influxdb_client):
        sink = InfluxDBDataSink(client=influxdb_client, bucket="test_bucket")
        
        test_data = {
            "symbol": "AAPL",
            "timestamp": "2024-01-15T10:00:00Z",
            "close": 150.0,
            "volume": 1000000
        }
        
        result = sink.write_data(test_data)
        assert result is True
        
        # Verify data was written
        query = 'from(bucket:"test_bucket") |> range(start:-1h) |> filter(fn:(r) => r._measurement == "market_data")'
        tables = influxdb_client.query_api().query(query)
        assert len(tables) > 0
```

### API Integration Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture
def client():
    return TestClient(app)

class TestMarketDataAPI:
    def test_get_symbol_quote(self, client):
        response = client.get("/api/v1/market-data/AAPL/quote")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["symbol"] == "AAPL"
        assert "data" in data
        assert "timestamp" in data
        assert data["success"] is True
    
    def test_authentication_required(self, client):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
    
    def test_authenticated_request(self, client):
        # Login first
        login_data = {
            "email": "test@example.com",
            "password": "testpassword"
        }
        login_response = client.post("/api/v1/auth/login", json=login_data)
        assert login_response.status_code == 200
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make authenticated request
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
```

## End-to-End Testing

### Complete User Workflows

```python
import pytest
from playwright.sync_api import sync_playwright

class TestE2EUserWorkflows:
    @pytest.fixture(scope="class")
    def browser_context(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            yield context
            context.close()
            browser.close()
    
    def test_complete_user_registration_and_data_access(self, browser_context):
        page = browser_context.new_page()
        
        # Navigate to registration page
        page.goto("http://localhost:3000/register")
        
        # Fill registration form
        page.fill("input[name='email']", "testuser@example.com")
        page.fill("input[name='username']", "testuser")
        page.fill("input[name='password']", "SecurePassword123!")
        page.click("button[type='submit']")
        
        # Verify redirect to dashboard
        page.wait_for_url("**/dashboard")
        
        # Search for stock symbol
        page.fill("input[placeholder='Search symbols...']", "AAPL")
        page.press("input[placeholder='Search symbols...']", "Enter")
        
        # Verify search results
        page.wait_for_selector(".search-results")
        results = page.query_selector_all(".search-result")
        assert len(results) > 0
        
        # Click on first result
        results[0].click()
        
        # Verify stock data is displayed
        page.wait_for_selector(".stock-chart")
        page.wait_for_selector(".stock-indicators")
        
        # Verify indicators are calculated
        rsi_element = page.query_selector("[data-testid='rsi-value']")
        assert rsi_element is not None
        assert rsi_element.inner_text() != ""
```

### Data Pipeline Testing

```python
class TestDataPipelineE2E:
    @pytest.mark.asyncio
    async def test_complete_data_pipeline(self):
        """Test the complete flow from data ingestion to API response"""
        
        # 1. Trigger data ingestion
        ingestion_service = DataIngestionService()
        ingestion_result = await ingestion_service.ingest_symbol_data(
            symbol="AAPL", 
            data_type=MarketDataType.DAILY
        )
        assert ingestion_result["success"] is True
        
        # 2. Wait for data processing
        await asyncio.sleep(2)  # Allow time for processing
        
        # 3. Verify data is stored
        storage_service = DataStorageService()
        stored_data = await storage_service.query_data(
            symbol="AAPL",
            data_type=MarketDataType.DAILY,
            limit=1
        )
        assert len(stored_data) > 0
        
        # 4. Test API endpoint returns processed data
        client = TestClient(app)
        response = client.get("/api/v1/market-data/AAPL/daily")
        
        assert response.status_code == 200
        api_data = response.json()
        assert api_data["symbol"] == "AAPL"
        assert len(api_data["data"]["ohlc_data"]) > 0
        
        # 5. Verify indicator calculations
        indicators_response = client.get("/api/v1/indicators/AAPL/sma?period=20")
        assert indicators_response.status_code == 200
        
        indicators_data = indicators_response.json()
        assert "values" in indicators_data
        assert len(indicators_data["values"]) > 0
```

## Performance Testing

### Load Testing

```python
import asyncio
import aiohttp
import pytest
from concurrent.futures import ThreadPoolExecutor
import time

class TestAPIPerformance:
    @pytest.mark.performance
    async def test_api_load_concurrent_requests(self):
        """Test API performance under load"""
        
        async def make_request(session, url):
            start_time = time.time()
            async with session.get(url) as response:
                await response.text()
                end_time = time.time()
                return {
                    "status_code": response.status,
                    "response_time": end_time - start_time
                }
        
        # Test configuration
        concurrent_users = 50
        requests_per_user = 10
        total_requests = concurrent_users * requests_per_user
        
        # Create concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(total_requests):
                task = make_request(session, "http://localhost:8000/api/v1/market-data/AAPL/quote")
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
        
        # Analyze results
        successful_requests = sum(1 for r in results if r["status_code"] == 200)
        response_times = [r["response_time"] for r in results if r["status_code"] == 200]
        
        total_time = end_time - start_time
        requests_per_second = total_requests / total_time
        avg_response_time = sum(response_times) / len(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
        
        # Performance assertions
        assert successful_requests / total_requests >= 0.95  # 95% success rate
        assert requests_per_second >= 100  # At least 100 RPS
        assert avg_response_time < 0.5  # Average response time < 500ms
        assert p95_response_time < 1.0  # 95th percentile < 1 second
    
    @pytest.mark.performance
    def test_database_query_performance(self):
        """Test database query performance"""
        
        # Setup test data
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        with SessionLocal() as db:
            start_time = time.time()
            
            # Perform complex query
            results = db.query(MarketData).filter(
                MarketData.symbol.in_(test_symbols),
                MarketData.timestamp >= datetime.now() - timedelta(days=30)
            ).order_by(MarketData.timestamp.desc()).limit(1000).all()
            
            end_time = time.time()
            query_time = end_time - start_time
            
            # Performance assertions
            assert query_time < 0.1  # Query should complete in < 100ms
            assert len(results) > 0
```

### Memory and Resource Testing

```python
import psutil
import gc
from memory_profiler import profile

class TestResourceUsage:
    @profile
    def test_memory_usage_during_data_processing(self):
        """Test memory usage during large data processing"""
        
        # Generate large dataset
        large_dataset = []
        for i in range(10000):
            data_point = {
                "symbol": f"STOCK{i % 100}",
                "timestamp": datetime.now() - timedelta(minutes=i),
                "price": 100 + (i % 50),
                "volume": 1000 + (i % 10000)
            }
            large_dataset.append(data_point)
        
        # Process data
        processor = DataProcessor()
        processed_data = processor.process_batch(large_dataset)
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        assert memory_usage < 512  # Should use less than 512 MB
        assert len(processed_data) == len(large_dataset)
```

## Security Testing

### Authentication and Authorization Tests

```python
class TestSecurityFeatures:
    def test_jwt_token_validation(self):
        """Test JWT token security"""
        
        # Test with invalid token
        invalid_tokens = [
            "invalid.token.here",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid",
            "",
            "Bearer malformed_token"
        ]
        
        client = TestClient(app)
        
        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 401
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        
        client = TestClient(app)
        
        # Attempt SQL injection through symbol parameter
        malicious_symbols = [
            "AAPL'; DROP TABLE users; --",
            "AAPL' OR 1=1 --",
            "AAPL' UNION SELECT * FROM users --"
        ]
        
        for symbol in malicious_symbols:
            response = client.get(f"/api/v1/market-data/{symbol}/quote")
            # Should not cause server error, should return 400 or 404
            assert response.status_code in [400, 404]
    
    def test_rate_limiting_enforcement(self):
        """Test rate limiting functionality"""
        
        client = TestClient(app)
        
        # Make requests rapidly
        responses = []
        for i in range(110):  # Exceed rate limit of 100 requests
            response = client.get("/api/v1/health")
            responses.append(response.status_code)
        
        # Should eventually get rate limited (429)
        rate_limited_responses = sum(1 for status in responses if status == 429)
        assert rate_limited_responses > 0
    
    def test_input_sanitization(self):
        """Test input sanitization and validation"""
        
        client = TestClient(app)
        
        # Test XSS prevention
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src='x' onerror='alert(1)'>",
            "'; DROP TABLE users; --"
        ]
        
        for malicious_input in malicious_inputs:
            response = client.post("/api/v1/auth/register", json={
                "email": "test@example.com",
                "username": malicious_input,  # Malicious input in username
                "password": "ValidPassword123!",
                "full_name": "Test User"
            })
            
            # Should reject malicious input
            assert response.status_code == 400
```

## Running Tests

### Test Execution Commands

#### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/core/test_indicators.py

# Run specific test class
pytest tests/unit/core/test_indicators.py::TestSimpleMovingAverage

# Run specific test method
pytest tests/unit/core/test_indicators.py::TestSimpleMovingAverage::test_calculate_sma

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto
```

#### Test Categories
```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only end-to-end tests
pytest tests/e2e/

# Run performance tests (marked with @pytest.mark.performance)
pytest -m performance

# Run tests excluding slow tests
pytest -m "not slow"

# Run security tests
pytest -m security
```

#### Docker Test Environment
```bash
# Run tests in Docker container
docker-compose -f docker-compose.test.yml up --build

# Run specific test suite in Docker
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/unit/

# Run tests with coverage in Docker
docker-compose -f docker-compose.test.yml run --rm test-runner pytest --cov=src --cov-report=xml
```

### Test Configuration

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=90

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow running tests
    security: Security tests
    database: Tests requiring database
```

#### conftest.py
```python
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.base import Base
from src.main import app
from fastapi.testclient import TestClient

# Configure async test loop
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Database fixtures
@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///./test.db", echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(test_db_engine):
    """Create database session for tests."""
    TestingSessionLocal = sessionmaker(bind=test_db_engine)
    session = TestingSessionLocal()
    yield session
    session.rollback()
    session.close()

# API client fixture
@pytest.fixture(scope="function")
def client():
    """Create test client for API testing."""
    return TestClient(app)

# Authentication fixtures
@pytest.fixture
def authenticated_user(client):
    """Create authenticated user for tests."""
    user_data = {
        "email": "testuser@example.com",
        "username": "testuser",
        "password": "TestPassword123!",
        "full_name": "Test User"
    }
    
    # Register user
    client.post("/api/v1/auth/register", json=user_data)
    
    # Login to get token
    login_response = client.post("/api/v1/auth/login", json={
        "email": user_data["email"],
        "password": user_data["password"]
    })
    
    token = login_response.json()["access_token"]
    return {
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
        "user_data": user_data
    }
```

## Test Coverage

### Coverage Requirements

- **Overall Coverage**: Minimum 90%
- **Unit Test Coverage**: Minimum 95%
- **Integration Test Coverage**: Minimum 80%
- **Critical Path Coverage**: 100%

### Coverage Reporting

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report (for CI/CD)
pytest --cov=src --cov-report=xml

# Show missing lines in terminal
pytest --cov=src --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=90
```

### Coverage Analysis

```python
# .coveragerc configuration file
[run]
source = src/
omit = 
    src/*/tests/*
    src/*/test_*
    */venv/*
    */migrations/*
    */__pycache__/*
    */conftest.py
    src/main.py  # Exclude main entry point

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    @abstractmethod
    
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_USER: test_user
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        DATABASE_URL: postgresql://test_user:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
  
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/unit/ --tb=short
        language: system
        types: [python]
        pass_filenames: false
        always_run: true
```

## Testing Best Practices

### 1. Test Naming Conventions

```python
# Good test names - descriptive and specific
def test_calculate_sma_with_valid_prices_returns_correct_values():
    pass

def test_user_registration_with_duplicate_email_raises_validation_error():
    pass

def test_api_endpoint_returns_401_when_token_is_expired():
    pass

# Bad test names - vague and unclear
def test_sma():
    pass

def test_user():
    pass

def test_api():
    pass
```

### 2. Test Independence

```python
# Good - each test is independent
class TestUserService:
    def test_create_user_success(self):
        user_service = UserService()
        user = user_service.create_user("test@example.com", "password")
        assert user.email == "test@example.com"
    
    def test_create_user_duplicate_email(self):
        user_service = UserService()
        # Create first user
        user_service.create_user("test@example.com", "password1")
        
        # Attempt to create duplicate should fail
        with pytest.raises(DuplicateEmailError):
            user_service.create_user("test@example.com", "password2")

# Bad - tests depend on each other
class TestUserServiceBad:
    def test_create_user(self):
        # This test creates a user that other tests depend on
        self.user = UserService().create_user("test@example.com", "password")
    
    def test_get_user(self):
        # This test depends on the previous test
        user = UserService().get_user(self.user.id)
        assert user is not None
```

### 3. Use of Test Fixtures

```python
@pytest.fixture
def sample_market_data():
    """Provide consistent test data across tests."""
    return [
        {"symbol": "AAPL", "price": 150.0, "volume": 1000000, "date": "2024-01-15"},
        {"symbol": "MSFT", "price": 300.0, "volume": 800000, "date": "2024-01-15"},
        {"symbol": "GOOGL", "price": 2500.0, "volume": 500000, "date": "2024-01-15"},
    ]

@pytest.fixture
def mock_external_api(mocker):
    """Mock external API calls for consistent testing."""
    mock = mocker.patch('src.external.api_client.fetch_data')
    mock.return_value = {"status": "success", "data": [...]}
    return mock
```

### 4. Error Testing Patterns

```python
class TestErrorHandling:
    def test_specific_exception_with_message(self):
        """Test that specific exceptions are raised with correct messages."""
        validator = DataValidator()
        
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            validator.validate_symbol("")
    
    def test_multiple_error_conditions(self):
        """Test multiple error scenarios."""
        error_cases = [
            ("", "Symbol cannot be empty"),
            ("A", "Symbol too short"),
            ("VERYLONGSYMBOL", "Symbol too long"),
            ("123", "Symbol cannot be numeric only"),
        ]
        
        validator = DataValidator()
        
        for symbol, expected_message in error_cases:
            with pytest.raises(ValidationError, match=expected_message):
                validator.validate_symbol(symbol)
```

### 5. Async Testing Best Practices

```python
@pytest.mark.asyncio
async def test_async_data_processing():
    """Test async operations properly."""
    processor = AsyncDataProcessor()
    
    # Use async context managers
    async with processor.get_connection() as conn:
        result = await processor.process_data(conn, test_data)
    
    assert result.success is True

@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent async operations."""
    service = DataService()
    
    # Test multiple concurrent operations
    tasks = [
        service.process_symbol("AAPL"),
        service.process_symbol("MSFT"),
        service.process_symbol("GOOGL")
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    assert all(r.success for r in results)
```

This comprehensive testing guide ensures that the Market Analysis System maintains high quality, reliability, and performance through rigorous testing practices at all levels of the application.