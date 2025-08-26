"""
API client factory for creating configured market data clients.
"""

import logging
from typing import Optional, Dict, Any, Union
from enum import Enum

from ...config import get_settings
from .base_client import BaseAPIClient
from .alpha_vantage_client import AlphaVantageClient
from ..models.market_data import APIError


class DataProvider(str, Enum):
    """Supported market data providers."""
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX = "iex"


class APIClientFactory:
    """Factory class for creating configured API clients."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
    
    def create_client(
        self, 
        provider: Union[DataProvider, str],
        **kwargs
    ) -> BaseAPIClient:
        """Create and configure an API client for the specified provider."""
        if isinstance(provider, str):
            try:
                provider = DataProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        
        if provider == DataProvider.ALPHA_VANTAGE:
            return self._create_alpha_vantage_client(**kwargs)
        elif provider == DataProvider.POLYGON:
            return self._create_polygon_client(**kwargs)
        elif provider == DataProvider.IEX:
            return self._create_iex_client(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _create_alpha_vantage_client(self, **kwargs) -> AlphaVantageClient:
        """Create Alpha Vantage client with configuration."""
        api_key = self.settings.external_apis.alpha_vantage_api_key
        
        if not api_key or api_key == "demo":
            if self.settings.is_development:
                self.logger.warning(
                    "Using demo API key for Alpha Vantage in development. "
                    "Set ALPHA_VANTAGE_API_KEY environment variable for production."
                )
            else:
                raise APIError(
                    "Alpha Vantage API key not configured. "
                    "Set ALPHA_VANTAGE_API_KEY environment variable."
                )
        
        client_config = {
            "api_key": api_key,
            "timeout": kwargs.get("timeout", 30),
            "max_retries": kwargs.get("max_retries", 3),
            "backoff_factor": kwargs.get("backoff_factor", 1.0)
        }
        
        return AlphaVantageClient(**client_config)
    
    def _create_polygon_client(self, **kwargs) -> BaseAPIClient:
        """Create Polygon client with configuration."""
        # Placeholder for Polygon client implementation
        api_key = self.settings.external_apis.polygon_api_key
        
        if not api_key:
            raise APIError(
                "Polygon API key not configured. "
                "Set POLYGON_API_KEY environment variable."
            )
        
        # TODO: Implement PolygonClient when needed
        raise NotImplementedError("Polygon client not yet implemented")
    
    def _create_iex_client(self, **kwargs) -> BaseAPIClient:
        """Create IEX Cloud client with configuration."""
        # Placeholder for IEX client implementation
        api_key = self.settings.external_apis.iex_cloud_api_key
        
        if not api_key:
            raise APIError(
                "IEX Cloud API key not configured. "
                "Set IEX_CLOUD_API_KEY environment variable."
            )
        
        # TODO: Implement IEXClient when needed
        raise NotImplementedError("IEX client not yet implemented")
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get list of available providers and their configuration status."""
        providers = {}
        
        # Check Alpha Vantage
        av_key = self.settings.external_apis.alpha_vantage_api_key
        providers[DataProvider.ALPHA_VANTAGE] = bool(av_key and av_key != "demo")
        
        # Check Polygon
        poly_key = self.settings.external_apis.polygon_api_key
        providers[DataProvider.POLYGON] = bool(poly_key)
        
        # Check IEX
        iex_key = self.settings.external_apis.iex_cloud_api_key
        providers[DataProvider.IEX] = bool(iex_key)
        
        return providers
    
    def get_default_provider(self) -> DataProvider:
        """Get the default provider based on available configuration."""
        available = self.get_available_providers()
        
        # Priority order: Alpha Vantage, Polygon, IEX
        for provider in [DataProvider.ALPHA_VANTAGE, DataProvider.POLYGON, DataProvider.IEX]:
            if available.get(provider, False):
                return provider
        
        # Fallback to Alpha Vantage with demo key in development
        if self.settings.is_development:
            return DataProvider.ALPHA_VANTAGE
        
        raise APIError("No configured data providers available")


# Convenience functions for common use cases
def get_default_client(**kwargs) -> BaseAPIClient:
    """Get a client for the default configured provider."""
    factory = APIClientFactory()
    default_provider = factory.get_default_provider()
    return factory.create_client(default_provider, **kwargs)


def get_alpha_vantage_client(**kwargs) -> AlphaVantageClient:
    """Get an Alpha Vantage client."""
    factory = APIClientFactory()
    return factory.create_client(DataProvider.ALPHA_VANTAGE, **kwargs)


async def test_client_connection(provider: Union[DataProvider, str]) -> Dict[str, Any]:
    """Test connection to a data provider."""
    factory = APIClientFactory()
    
    try:
        client = factory.create_client(provider)
        
        # Test with a simple symbol search or quote
        async with client:
            response = await client.search_symbols("AAPL")
            
        return {
            "provider": str(provider),
            "success": response.success,
            "error": response.error_message,
            "rate_limit_info": client.get_rate_limit_info()
        }
        
    except Exception as e:
        return {
            "provider": str(provider),
            "success": False,
            "error": str(e),
            "rate_limit_info": None
        }