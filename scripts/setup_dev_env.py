#!/usr/bin/env python3
"""
Complete development environment setup script
"""

import os
import subprocess
import time
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description="", check=True, timeout=300):
    """Run a shell command with logging."""
    logger.info(f"🔧 {description}")
    logger.info(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            logger.info(f"   ✅ Success")
            if result.stdout.strip():
                logger.info(f"   Output: {result.stdout.strip()}")
        else:
            logger.error(f"   ❌ Failed (exit code: {result.returncode})")
            if result.stderr.strip():
                logger.error(f"   Error: {result.stderr.strip()}")
        
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"   ❌ Timeout after {timeout} seconds")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"   ❌ Failed: {e}")
        return None

def check_prerequisites():
    """Check if required tools are installed."""
    logger.info("🔍 Checking prerequisites...")
    
    required_tools = {
        "docker": "docker --version",
        "docker-compose": "docker-compose --version",
        "make": "make --version"
    }
    
    all_good = True
    
    for tool, command in required_tools.items():
        result = run_command(command, f"Checking {tool}", check=False)
        if result and result.returncode == 0:
            logger.info(f"   ✅ {tool} is available")
        else:
            logger.error(f"   ❌ {tool} is not available or not working")
            all_good = False
    
    return all_good

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            logger.info("📄 Creating .env file from template...")
            import shutil
            shutil.copy(env_example, env_file)
            logger.info("   ✅ .env file created")
            logger.info("   ⚠️  Please review and update .env file with your settings")
        else:
            logger.warning("   ⚠️  .env.example not found, creating basic .env")
            with open(env_file, 'w') as f:
                f.write("# Market Analysis System - Environment Variables\n")
                f.write("DEBUG=true\n")
                f.write("LOG_LEVEL=INFO\n")
            logger.info("   ✅ Basic .env file created")
    else:
        logger.info("📄 .env file already exists")

def setup_docker_environment():
    """Set up Docker development environment."""
    logger.info("🐳 Setting up Docker development environment...")
    
    # Stop any existing containers
    run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml down",
        "Stopping existing containers",
        check=False
    )
    
    # Clean up old volumes (optional)
    logger.info("🧹 Cleaning up old development data...")
    run_command(
        "docker volume prune -f",
        "Pruning unused volumes", 
        check=False
    )
    
    # Build images
    logger.info("🏗️  Building Docker images...")
    result = run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache",
        "Building all services",
        timeout=600  # 10 minutes
    )
    
    if not result or result.returncode != 0:
        logger.error("❌ Failed to build Docker images")
        return False
    
    # Start core services first
    logger.info("🚀 Starting core services...")
    result = run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres influxdb redis zookeeper kafka",
        "Starting database and message queue services",
        timeout=180
    )
    
    if not result or result.returncode != 0:
        logger.error("❌ Failed to start core services")
        return False
    
    # Wait for services to be ready
    logger.info("⏳ Waiting for services to be ready...")
    time.sleep(30)
    
    # Start remaining services
    logger.info("🚀 Starting remaining services...")
    result = run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d",
        "Starting all services"
    )
    
    if not result or result.returncode != 0:
        logger.error("❌ Failed to start all services")
        return False
    
    return True

def initialize_services():
    """Initialize services with data and topics."""
    logger.info("🔧 Initializing services...")
    
    # Wait a bit more for services to fully start
    logger.info("⏳ Waiting for services to fully initialize...")
    time.sleep(20)
    
    # Initialize Kafka topics
    result = run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm kafka-init",
        "Initializing Kafka topics"
    )
    
    if not result or result.returncode != 0:
        logger.error("❌ Failed to initialize Kafka topics")
        return False
    
    # Test connections
    result = run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm db-init",
        "Testing service connections"
    )
    
    if not result or result.returncode != 0:
        logger.warning("⚠️  Some service connections failed, but continuing...")
    
    # Seed development data
    result = run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm seed-data",
        "Seeding development database"
    )
    
    if not result or result.returncode != 0:
        logger.warning("⚠️  Failed to seed development data, but continuing...")
    
    return True

def verify_setup():
    """Verify the setup is working."""
    logger.info("🧪 Verifying setup...")
    
    # Check if containers are running
    result = run_command(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps",
        "Checking container status",
        check=False
    )
    
    # Test API endpoint
    time.sleep(10)  # Give API time to start
    result = run_command(
        "curl -f http://localhost:8000/health",
        "Testing API health endpoint",
        check=False
    )
    
    if result and result.returncode == 0:
        logger.info("   ✅ API is responding")
    else:
        logger.warning("   ⚠️  API may not be ready yet")
    
    return True

def show_access_points():
    """Show access points for the user."""
    logger.info("🎯 Development environment is ready!")
    logger.info("=" * 60)
    logger.info("📱 Access Points:")
    logger.info("   • API:              http://localhost:8000")
    logger.info("   • API Docs:         http://localhost:8000/docs")
    logger.info("   • API Health:       http://localhost:8000/health")
    logger.info("   • Kafka UI:         http://localhost:8080")
    logger.info("   • Grafana:          http://localhost:3001 (admin/admin)")
    logger.info("   • Prometheus:       http://localhost:9090")
    logger.info("")
    logger.info("🛠️  Common Commands:")
    logger.info("   • View logs:        make logs")
    logger.info("   • Stop services:    make dev-stop")
    logger.info("   • Restart:          make dev-restart")
    logger.info("   • Health check:     make health-check")
    logger.info("   • Reset data:       make reset-data")
    logger.info("")
    logger.info("📚 Documentation:")
    logger.info("   • README.md:        Project overview")
    logger.info("   • PLAN.md:          Implementation plan")
    logger.info("   • CLAUDE.md:        Development guide")
    logger.info("=" * 60)

def main():
    """Main setup function."""
    logger.info("🚀 Market Analysis System - Development Environment Setup")
    logger.info("=" * 60)
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("❌ Prerequisites not met. Please install required tools.")
            sys.exit(1)
        
        # Create environment file
        create_env_file()
        
        # Setup Docker environment
        if not setup_docker_environment():
            logger.error("❌ Docker environment setup failed")
            sys.exit(1)
        
        # Initialize services
        if not initialize_services():
            logger.error("❌ Service initialization failed")
            sys.exit(1)
        
        # Verify setup
        verify_setup()
        
        # Show access points
        show_access_points()
        
        logger.info("🎉 Setup completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()