#!/usr/bin/env python3
"""
Test script to verify Docker build works
"""

def test_imports():
    """Test that all required packages can be imported."""
    import sys
    print(f"Python version: {sys.version}")
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import asyncpg
        print(f"✅ AsyncPG imported successfully")
    except ImportError as e:
        print(f"❌ AsyncPG import failed: {e}")
        return False
    
    try:
        import redis
        print(f"✅ Redis imported successfully")
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False
    
    try:
        import kafka
        print("✅ Kafka-python imported successfully")
    except ImportError as e:
        print(f"❌ Kafka-python import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ Numpy {np.__version__}")
    except ImportError as e:
        print(f"❌ Numpy import failed: {e}")
        return False
    
    try:
        import httpx
        print("✅ HTTPX imported successfully")
    except ImportError as e:
        print(f"❌ HTTPX import failed: {e}")
        return False
    
    try:
        import pydantic
        print(f"✅ Pydantic {pydantic.__version__}")
    except ImportError as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
    
    try:
        from influxdb_client import InfluxDBClient
        print("✅ InfluxDB client imported successfully")
    except ImportError as e:
        print(f"❌ InfluxDB client import failed: {e}")
        return False
    
    print("\n🎉 All core packages imported successfully!")
    return True

def test_configuration():
    """Test configuration loading."""
    try:
        # This will test if our config module loads correctly
        from src.config import get_settings
        settings = get_settings()
        print(f"✅ Configuration loaded - Environment: {settings.environment}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Docker build...")
    print("=" * 50)
    
    success = True
    
    success &= test_imports()
    print()
    success &= test_configuration()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Docker build is working correctly.")
        exit(0)
    else:
        print("❌ Some tests failed. Check the error messages above.")
        exit(1)