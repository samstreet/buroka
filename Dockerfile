# Multi-stage Dockerfile for Market Analysis System

# Build stage for Python dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel for better package building
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Development stage
FROM python:3.11-slim as development

WORKDIR /app

# Install system dependencies for development
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN useradd --create-home --shell /bin/bash trader

# Copy Python packages from builder stage to trader's home
COPY --from=builder /root/.local /home/trader/.local

# Set up permissions and PATH for trader user
RUN chown -R trader:trader /home/trader/.local && \
    chmod -R 755 /home/trader/.local

# Switch to trader user
USER trader

# Update PATH to use trader's local bin
ENV PATH=/home/trader/.local/bin:$PATH

# Copy application code
COPY --chown=trader:trader . .

# Set PYTHONPATH to find our modules
ENV PYTHONPATH=/app

# Expose port for FastAPI
EXPOSE 8000

# Development command with hot reloading
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY requirements.txt .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trader && \
    chown -R trader:trader /app
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]