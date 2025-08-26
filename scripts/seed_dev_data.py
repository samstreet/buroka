#!/usr/bin/env python3
"""
Seed development database with test data
"""

import os
import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_connection():
    """Create database connection."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "market_analysis")
    user = os.getenv("POSTGRES_USER", "trader")
    password = os.getenv("POSTGRES_PASSWORD", "secure_password")
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return await asyncpg.connect(connection_string)

async def seed_users(conn):
    """Seed test users."""
    logger.info("👥 Seeding test users...")
    
    users = [
        {
            "email": "alice@trader.com",
            "username": "alice_trader",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/JxHIo9K9EoD5I3KcO",  # password: test123
            "is_verified": True
        },
        {
            "email": "bob@trader.com", 
            "username": "bob_analyst",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/JxHIo9K9EoD5I3KcO",
            "is_verified": True
        },
        {
            "email": "charlie@trader.com",
            "username": "charlie_quant",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/JxHIo9K9EoD5I3KcO",
            "is_verified": True
        },
        {
            "email": "diana@trader.com",
            "username": "diana_portfolio",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/JxHIo9K9EoD5I3KcO",
            "is_verified": False  # Test unverified user
        }
    ]
    
    for user in users:
        try:
            await conn.execute("""
                INSERT INTO user_management.users (email, username, hashed_password, is_verified)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (email) DO UPDATE SET 
                    updated_at = NOW()
            """, user["email"], user["username"], user["hashed_password"], user["is_verified"])
            
            logger.info(f"   ✅ User {user['username']} created/updated")
        except Exception as e:
            logger.error(f"   ❌ Failed to create user {user['username']}: {e}")

async def seed_watchlists(conn):
    """Seed test watchlists."""
    logger.info("📋 Seeding watchlists...")
    
    # Get user IDs
    users = await conn.fetch("SELECT id, username FROM user_management.users")
    user_map = {user['username']: user['id'] for user in users}
    
    watchlists = [
        {
            "user": "alice_trader",
            "name": "Tech Giants",
            "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
            "is_default": True
        },
        {
            "user": "alice_trader",
            "name": "Electric Vehicles",
            "symbols": ["TSLA", "NIO", "RIVN", "LCID"],
            "is_default": False
        },
        {
            "user": "bob_analyst",
            "name": "Banking Sector",
            "symbols": ["JPM", "BAC", "WFC", "GS", "MS"],
            "is_default": True
        },
        {
            "user": "charlie_quant",
            "name": "Crypto Portfolio",
            "symbols": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD"],
            "is_default": True
        },
        {
            "user": "charlie_quant",
            "name": "High Beta Stocks",
            "symbols": ["NVDA", "AMD", "NFLX", "ZOOM", "ROKU"],
            "is_default": False
        },
        {
            "user": "diana_portfolio",
            "name": "Blue Chips",
            "symbols": ["SPY", "QQQ", "DIA", "IWM"],
            "is_default": True
        }
    ]
    
    for watchlist in watchlists:
        user_id = user_map.get(watchlist["user"])
        if user_id:
            try:
                await conn.execute("""
                    INSERT INTO market_data.watchlists (user_id, name, symbols, is_default)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, name) DO UPDATE SET 
                        symbols = $3, updated_at = NOW()
                """, user_id, watchlist["name"], watchlist["symbols"], watchlist["is_default"])
                
                logger.info(f"   ✅ Watchlist '{watchlist['name']}' for {watchlist['user']}")
            except Exception as e:
                logger.error(f"   ❌ Failed to create watchlist: {e}")

async def seed_pattern_definitions(conn):
    """Seed additional pattern definitions."""
    logger.info("🔍 Seeding pattern definitions...")
    
    patterns = [
        {
            "name": "Double Bottom",
            "description": "Classic reversal pattern with two lows at approximately same level",
            "pattern_type": "reversal",
            "parameters": {"min_distance": 10, "tolerance": 0.02}
        },
        {
            "name": "Head and Shoulders",
            "description": "Reversal pattern with three peaks, middle one highest",
            "pattern_type": "reversal",
            "parameters": {"shoulder_ratio": 0.8, "neckline_break": True}
        },
        {
            "name": "Ascending Triangle",
            "description": "Continuation pattern with horizontal resistance and rising support",
            "pattern_type": "continuation",
            "parameters": {"min_touches": 2, "trend_strength": 0.7}
        },
        {
            "name": "MACD Bullish Divergence",
            "description": "Price makes lower low while MACD makes higher low",
            "pattern_type": "divergence",
            "parameters": {"lookback": 20, "fast": 12, "slow": 26, "signal": 9}
        },
        {
            "name": "Bollinger Band Squeeze",
            "description": "Low volatility setup with tight Bollinger Bands",
            "pattern_type": "volatility",
            "parameters": {"period": 20, "std_dev": 2, "squeeze_threshold": 0.1}
        }
    ]
    
    for pattern in patterns:
        try:
            await conn.execute("""
                INSERT INTO analytics.pattern_definitions (name, description, pattern_type, parameters)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (name) DO UPDATE SET 
                    description = $2, parameters = $4, updated_at = NOW()
            """, pattern["name"], pattern["description"], pattern["pattern_type"], json.dumps(pattern["parameters"]))
            
            logger.info(f"   ✅ Pattern '{pattern['name']}' created/updated")
        except Exception as e:
            logger.error(f"   ❌ Failed to create pattern: {e}")

async def seed_detected_patterns(conn):
    """Seed sample detected patterns."""
    logger.info("📊 Seeding detected patterns...")
    
    # Get pattern definition IDs
    patterns = await conn.fetch("SELECT id, name FROM analytics.pattern_definitions")
    pattern_map = {p['name']: p['id'] for p in patterns}
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "BTC-USD", "ETH-USD"]
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    # Generate sample detected patterns for the last 7 days
    base_time = datetime.utcnow() - timedelta(days=7)
    
    detected_patterns = []
    for i in range(50):  # Create 50 sample patterns
        symbol = random.choice(symbols)
        pattern_name = random.choice(list(pattern_map.keys()))
        pattern_id = pattern_map[pattern_name]
        confidence = round(random.uniform(0.6, 0.95), 4)
        timeframe = random.choice(timeframes)
        detected_time = base_time + timedelta(
            days=random.randint(0, 6),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Generate realistic pattern data
        pattern_data = {
            "entry_price": round(random.uniform(100, 300), 2),
            "target_price": round(random.uniform(100, 350), 2),
            "stop_loss": round(random.uniform(80, 150), 2),
            "volume_confirmation": random.choice([True, False]),
            "strength": random.choice(["weak", "moderate", "strong"]),
            "risk_reward_ratio": round(random.uniform(1.5, 4.0), 2)
        }
        
        detected_patterns.append({
            "symbol": symbol,
            "pattern_definition_id": pattern_id,
            "confidence_score": confidence,
            "timeframe": timeframe,
            "detected_at": detected_time,
            "pattern_data": pattern_data,
            "status": random.choice(["active", "completed", "invalidated"])
        })
    
    for pattern in detected_patterns:
        try:
            await conn.execute("""
                INSERT INTO analytics.detected_patterns 
                (symbol, pattern_definition_id, confidence_score, timeframe, detected_at, pattern_data, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
            pattern["symbol"], 
            pattern["pattern_definition_id"], 
            pattern["confidence_score"],
            pattern["timeframe"], 
            pattern["detected_at"], 
            json.dumps(pattern["pattern_data"]),
            pattern["status"])
            
        except Exception as e:
            logger.error(f"   ❌ Failed to create detected pattern: {e}")
    
    logger.info(f"   ✅ Created {len(detected_patterns)} sample detected patterns")

async def seed_alerts(conn):
    """Seed sample alerts."""
    logger.info("🚨 Seeding alerts...")
    
    # Get user IDs
    users = await conn.fetch("SELECT id, username FROM user_management.users")
    user_map = {user['username']: user['id'] for user in users}
    
    alerts = [
        {
            "user": "alice_trader",
            "symbol": "AAPL",
            "alert_type": "price_threshold",
            "condition_data": {"operator": ">", "value": 200.00},
            "is_triggered": False
        },
        {
            "user": "alice_trader",
            "symbol": "TSLA",
            "alert_type": "pattern_detected",
            "condition_data": {"pattern_type": "trend_reversal", "min_confidence": 0.8},
            "is_triggered": True,
            "triggered_at": datetime.utcnow() - timedelta(hours=2)
        },
        {
            "user": "bob_analyst",
            "symbol": "JPM",
            "alert_type": "volume_spike",
            "condition_data": {"multiplier": 2.0, "timeframe": "1h"},
            "is_triggered": False
        },
        {
            "user": "charlie_quant",
            "symbol": "BTC-USD",
            "alert_type": "volatility_spike",
            "condition_data": {"threshold": 0.05, "window": "4h"},
            "is_triggered": True,
            "triggered_at": datetime.utcnow() - timedelta(minutes=30)
        },
        {
            "user": "diana_portfolio",
            "symbol": "SPY",
            "alert_type": "moving_average_cross",
            "condition_data": {"fast_ma": 10, "slow_ma": 20, "direction": "bullish"},
            "is_triggered": False
        }
    ]
    
    for alert in alerts:
        user_id = user_map.get(alert["user"])
        if user_id:
            try:
                await conn.execute("""
                    INSERT INTO analytics.alerts 
                    (user_id, symbol, alert_type, condition_data, is_triggered, triggered_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                user_id, 
                alert["symbol"], 
                alert["alert_type"],
                json.dumps(alert["condition_data"]),
                alert["is_triggered"],
                alert.get("triggered_at"))
                
                logger.info(f"   ✅ Alert for {alert['user']} on {alert['symbol']}")
            except Exception as e:
                logger.error(f"   ❌ Failed to create alert: {e}")

async def main():
    """Main seeding function."""
    logger.info("🌱 Seeding development database with test data")
    logger.info("=" * 50)
    
    try:
        conn = await create_connection()
        
        # Run seeding functions
        await seed_users(conn)
        await seed_watchlists(conn)
        await seed_pattern_definitions(conn)
        await seed_detected_patterns(conn)
        await seed_alerts(conn)
        
        await conn.close()
        
        logger.info("=" * 50)
        logger.info("✅ Development data seeding completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Seeding failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())