#!/usr/bin/env python3
"""
Docker health check script for API container
"""

import httpx
import sys
import asyncio
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.WARNING)  # Reduce noise in health checks
logger = logging.getLogger(__name__)

async def check_api_health() -> Dict[str, Any]:
    """Check API health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/health")
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy" if data.get("status") == "healthy" else "unhealthy",
                    "response_time": response.elapsed.total_seconds(),
                    "data": data
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time": response.elapsed.total_seconds() if response.elapsed else None
                }
                
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time": None
        }

async def main():
    """Main health check function."""
    result = await check_api_health()
    
    if result["status"] == "healthy":
        print("✅ API is healthy")
        sys.exit(0)
    else:
        print(f"❌ API is unhealthy: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())