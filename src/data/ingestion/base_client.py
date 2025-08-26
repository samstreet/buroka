"""
Base API client with rate limiting and error handling.
"""

import asyncio
import aiohttp
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ..models.market_data import APIRateLimitInfo, APIError, MarketDataResponse


@dataclass
class RateLimitBucket:
    """Rate limiting bucket for API calls."""
    max_calls: int
    time_window: int  # seconds
    calls_made: List[float] = field(default_factory=list)
    
    def can_make_call(self) -> bool:
        """Check if a call can be made within rate limits."""
        current_time = time.time()
        # Remove calls outside the time window
        self.calls_made = [call_time for call_time in self.calls_made 
                          if current_time - call_time < self.time_window]
        return len(self.calls_made) < self.max_calls
    
    def record_call(self):
        """Record a new API call."""
        self.calls_made.append(time.time())
    
    def get_wait_time(self) -> float:
        """Get time to wait before next call can be made."""
        if not self.calls_made or len(self.calls_made) < self.max_calls:
            return 0.0
        
        oldest_call = min(self.calls_made)
        return max(0, self.time_window - (time.time() - oldest_call))
    
    def get_rate_limit_info(self) -> APIRateLimitInfo:
        """Get current rate limit status."""
        current_time = time.time()
        self.calls_made = [call_time for call_time in self.calls_made 
                          if current_time - call_time < self.time_window]
        
        calls_remaining = max(0, self.max_calls - len(self.calls_made))
        reset_time = None
        if self.calls_made:
            oldest_call = min(self.calls_made)
            reset_time = datetime.fromtimestamp(oldest_call + self.time_window)
        
        return APIRateLimitInfo(
            calls_made=len(self.calls_made),
            calls_remaining=calls_remaining,
            reset_time=reset_time,
            limit_period=self.time_window
        )


class CircuitBreaker:
    """Circuit breaker pattern implementation for API resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                self.failure_count = 0
            else:
                raise APIError("Circuit breaker is open - too many failures")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except self.expected_exception as e:
            self.on_failure()
            raise e
    
    async def acall(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open" 
                self.failure_count = 0
            else:
                raise APIError("Circuit breaker is open - too many failures")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except self.expected_exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class BaseAPIClient(ABC):
    """Base class for market data API clients."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        rate_limits: Dict[str, RateLimitBucket],
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.rate_limits = rate_limits
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breaker = CircuitBreaker()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        rate_limit_key: str = "default"
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and error handling."""
        await self._ensure_session()
        
        # Check rate limits
        if rate_limit_key in self.rate_limits:
            rate_limit = self.rate_limits[rate_limit_key]
            if not rate_limit.can_make_call():
                wait_time = rate_limit.get_wait_time()
                self.logger.warning(f"Rate limit exceeded, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Prepare request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        params = self._prepare_params(params)
        
        # Make request with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.circuit_breaker.acall(
                    self._execute_request, url, headers, params
                )
                
                # Record successful API call
                if rate_limit_key in self.rate_limits:
                    self.rate_limits[rate_limit_key].record_call()
                
                return response
                
            except aiohttp.ClientError as e:
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise APIError(f"Request failed after {self.max_retries} retries: {e}")
            except APIError:
                raise  # Re-raise API errors without retry
    
    async def _execute_request(self, url: str, headers: Dict, params: Dict) -> Dict[str, Any]:
        """Execute the actual HTTP request."""
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 429:  # Rate limit exceeded
                raise APIError(f"Rate limit exceeded: {response.status}", response.status)
            elif response.status == 401:
                raise APIError(f"Authentication failed: {response.status}", response.status)
            elif response.status == 403:
                raise APIError(f"Access forbidden: {response.status}", response.status)
            elif response.status >= 500:
                raise APIError(f"Server error: {response.status}", response.status)
            elif response.status >= 400:
                error_data = await response.json() if response.content_type == 'application/json' else {}
                raise APIError(f"Client error: {response.status}", response.status, error_data)
            
            response.raise_for_status()
            return await response.json()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests."""
        return {
            "User-Agent": "MarketAnalysisSystem/1.0",
            "Accept": "application/json"
        }
    
    def _prepare_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Prepare request parameters."""
        prepared_params = {"apikey": self.api_key}
        prepared_params.update({k: str(v) for k, v in params.items() if v is not None})
        return prepared_params
    
    def get_rate_limit_info(self, rate_limit_key: str = "default") -> Optional[APIRateLimitInfo]:
        """Get current rate limit information."""
        if rate_limit_key in self.rate_limits:
            return self.rate_limits[rate_limit_key].get_rate_limit_info()
        return None
    
    # Abstract methods that must be implemented by concrete clients
    @abstractmethod
    async def get_intraday_data(self, symbol: str, interval: str) -> MarketDataResponse:
        """Get intraday market data for a symbol."""
        pass
    
    @abstractmethod
    async def get_daily_data(self, symbol: str, outputsize: str = "compact") -> MarketDataResponse:
        """Get daily market data for a symbol."""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> MarketDataResponse:
        """Get real-time quote for a symbol."""
        pass
    
    @abstractmethod
    async def search_symbols(self, keywords: str) -> MarketDataResponse:
        """Search for symbols matching keywords."""
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format."""
        pass