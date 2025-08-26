#!/usr/bin/env python3
"""
Initialize Kafka topics for Market Analysis System
"""

import os
import time
from kafka import KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_kafka(bootstrap_servers: str, max_retries: int = 30, delay: int = 2):
    """Wait for Kafka to be ready."""
    for attempt in range(max_retries):
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=5000,
                connections_max_idle_ms=5000
            )
            # Try to get cluster metadata
            metadata = admin_client.describe_cluster()
            logger.info(f"Kafka is ready! Cluster ID: {metadata.cluster_id}")
            admin_client.close()
            return True
        except Exception as e:
            logger.info(f"Waiting for Kafka... (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(delay)
    
    raise Exception(f"Kafka not ready after {max_retries} attempts")

def create_topics(bootstrap_servers: str, topic_prefix: str = "market_"):
    """Create Kafka topics for the market analysis system."""
    
    # Topic configurations
    topics = [
        {
            "name": f"{topic_prefix}raw_data",
            "partitions": 3,
            "replication_factor": 1,
            "config": {
                "retention.ms": "604800000",  # 7 days
                "compression.type": "snappy",
                "cleanup.policy": "delete"
            },
            "description": "Raw market data from various sources"
        },
        {
            "name": f"{topic_prefix}processed_data",
            "partitions": 3,
            "replication_factor": 1,
            "config": {
                "retention.ms": "259200000",  # 3 days
                "compression.type": "snappy",
                "cleanup.policy": "delete"
            },
            "description": "Processed and normalized market data"
        },
        {
            "name": f"{topic_prefix}patterns",
            "partitions": 2,
            "replication_factor": 1,
            "config": {
                "retention.ms": "2592000000",  # 30 days
                "compression.type": "gzip",
                "cleanup.policy": "delete"
            },
            "description": "Detected trading patterns"
        },
        {
            "name": f"{topic_prefix}alerts",
            "partitions": 2,
            "replication_factor": 1,
            "config": {
                "retention.ms": "86400000",  # 1 day
                "compression.type": "gzip",
                "cleanup.policy": "delete"
            },
            "description": "User alerts and notifications"
        },
        {
            "name": f"{topic_prefix}technical_indicators",
            "partitions": 2,
            "replication_factor": 1,
            "config": {
                "retention.ms": "259200000",  # 3 days
                "compression.type": "snappy",
                "cleanup.policy": "delete"
            },
            "description": "Technical indicator calculations"
        },
        {
            "name": f"{topic_prefix}sentiment_data",
            "partitions": 2,
            "replication_factor": 1,
            "config": {
                "retention.ms": "172800000",  # 2 days
                "compression.type": "gzip",
                "cleanup.policy": "delete"
            },
            "description": "Market sentiment from news and social media"
        },
        {
            "name": f"{topic_prefix}ml_predictions",
            "partitions": 2,
            "replication_factor": 1,
            "config": {
                "retention.ms": "86400000",  # 1 day
                "compression.type": "gzip",
                "cleanup.policy": "delete"
            },
            "description": "Machine learning model predictions"
        }
    ]
    
    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        request_timeout_ms=10000
    )
    
    # Create topics
    new_topics = []
    for topic_config in topics:
        topic = NewTopic(
            name=topic_config["name"],
            num_partitions=topic_config["partitions"],
            replication_factor=topic_config["replication_factor"],
            topic_configs=topic_config["config"]
        )
        new_topics.append(topic)
    
    try:
        result = admin_client.create_topics(new_topics, validate_only=False)
        
        # Wait for topic creation to complete
        for topic_name, future in result.items():
            try:
                future.result()  # Block until topic is created
                logger.info(f"✅ Topic '{topic_name}' created successfully")
            except TopicAlreadyExistsError:
                logger.info(f"ℹ️  Topic '{topic_name}' already exists")
            except Exception as e:
                logger.error(f"❌ Failed to create topic '{topic_name}': {e}")
    
    except Exception as e:
        logger.error(f"❌ Failed to create topics: {e}")
        raise
    finally:
        admin_client.close()

def list_topics(bootstrap_servers: str):
    """List all topics in the Kafka cluster."""
    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        request_timeout_ms=10000
    )
    
    try:
        metadata = admin_client.list_topics()
        logger.info("📋 Current topics in cluster:")
        for topic in sorted(metadata):
            logger.info(f"   - {topic}")
        return metadata
    except Exception as e:
        logger.error(f"❌ Failed to list topics: {e}")
        raise
    finally:
        admin_client.close()

def main():
    """Main function to initialize Kafka topics."""
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_prefix = os.getenv("KAFKA_TOPIC_PREFIX", "market_")
    
    logger.info("🚀 Initializing Kafka topics for Market Analysis System")
    logger.info(f"📡 Bootstrap servers: {bootstrap_servers}")
    logger.info(f"🏷️  Topic prefix: {topic_prefix}")
    
    try:
        # Wait for Kafka to be ready
        wait_for_kafka(bootstrap_servers)
        
        # Create topics
        create_topics(bootstrap_servers, topic_prefix)
        
        # List all topics
        list_topics(bootstrap_servers)
        
        logger.info("✅ Kafka topics initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Kafka topics initialization failed: {e}")
        raise

if __name__ == "__main__":
    main()