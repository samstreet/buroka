#!/usr/bin/env python3
"""
Test all database and service connections
"""

import os
import asyncio
import asyncpg
import redis
from influxdb_client import InfluxDBClient
from kafka import KafkaProducer, KafkaConsumer
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_postgresql():
    """Test PostgreSQL connection."""
    try:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        database = os.getenv("POSTGRES_DB", "market_analysis")
        user = os.getenv("POSTGRES_USER", "trader")
        password = os.getenv("POSTGRES_PASSWORD", "secure_password")
        
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        logger.info("🐘 Testing PostgreSQL connection...")
        conn = await asyncpg.connect(connection_string)
        
        # Test query
        result = await conn.fetchval("SELECT version()")
        await conn.close()
        
        logger.info(f"✅ PostgreSQL connected successfully")
        logger.info(f"   Version: {result.split(',')[0]}")
        return True
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False

def test_redis():
    """Test Redis connection."""
    try:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        
        logger.info("🔴 Testing Redis connection...")
        
        if password:
            r = redis.Redis(host=host, port=port, password=password, decode_responses=True)
        else:
            r = redis.Redis(host=host, port=port, decode_responses=True)
        
        # Test operations
        r.ping()
        r.set("test_connection", "success")
        result = r.get("test_connection")
        r.delete("test_connection")
        
        info = r.info()
        
        logger.info(f"✅ Redis connected successfully")
        logger.info(f"   Version: {info['redis_version']}")
        logger.info(f"   Memory used: {info['used_memory_human']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False

def test_influxdb():
    """Test InfluxDB connection."""
    try:
        host = os.getenv("INFLUXDB_HOST", "localhost")
        port = int(os.getenv("INFLUXDB_PORT", "8086"))
        token = os.getenv("INFLUXDB_TOKEN", "dev_token_12345")
        org = os.getenv("INFLUXDB_ORG", "market_analysis")
        bucket = os.getenv("INFLUXDB_BUCKET", "market_data_dev")
        
        url = f"http://{host}:{port}"
        
        logger.info("📊 Testing InfluxDB connection...")
        
        client = InfluxDBClient(url=url, token=token, org=org)
        
        # Test connection
        health = client.health()
        
        # Test bucket access
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets()
        
        client.close()
        
        logger.info(f"✅ InfluxDB connected successfully")
        logger.info(f"   Status: {health.status}")
        logger.info(f"   Version: {health.version}")
        logger.info(f"   Available buckets: {len(buckets.buckets)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ InfluxDB connection failed: {e}")
        return False

def test_kafka():
    """Test Kafka connection."""
    try:
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        logger.info("📨 Testing Kafka connection...")
        
        # Test producer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: x.encode('utf-8')
        )
        
        # Test consumer (just create, don't consume)
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            consumer_timeout_ms=1000
        )
        
        # Get cluster metadata
        metadata = producer.list_topics()
        
        producer.close()
        consumer.close()
        
        logger.info(f"✅ Kafka connected successfully")
        logger.info(f"   Bootstrap servers: {bootstrap_servers}")
        logger.info(f"   Available topics: {len(metadata.topics)}")
        
        if metadata.topics:
            logger.info(f"   Topics: {', '.join(sorted(metadata.topics))}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Kafka connection failed: {e}")
        return False

async def run_all_tests():
    """Run all connection tests."""
    logger.info("🧪 Running database and service connection tests")
    logger.info("=" * 60)
    
    results = {}
    
    # Test PostgreSQL
    results['postgresql'] = await test_postgresql()
    
    # Test Redis
    results['redis'] = test_redis()
    
    # Test InfluxDB
    results['influxdb'] = test_influxdb()
    
    # Test Kafka
    results['kafka'] = test_kafka()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 Connection Test Summary:")
    
    all_passed = True
    for service, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {service.upper()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("🎉 All connection tests passed!")
        return 0
    else:
        logger.error("💥 Some connection tests failed!")
        return 1

async def main():
    """Main function."""
    logger.info(f"🚀 Market Analysis System - Connection Tests")
    logger.info(f"📅 {datetime.now().isoformat()}")
    logger.info("")
    
    # Show environment
    logger.info("🔧 Environment Configuration:")
    logger.info(f"   POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'localhost')}")
    logger.info(f"   INFLUXDB_HOST: {os.getenv('INFLUXDB_HOST', 'localhost')}")
    logger.info(f"   REDIS_HOST: {os.getenv('REDIS_HOST', 'localhost')}")
    logger.info(f"   KAFKA_BOOTSTRAP_SERVERS: {os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')}")
    logger.info("")
    
    exit_code = await run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())