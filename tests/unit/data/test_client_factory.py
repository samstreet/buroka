"""
Tests for API client factory.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.data.ingestion.client_factory import (
    APIClientFactory, DataProvider, get_default_client,
    get_alpha_vantage_client
)
from src.data.ingestion.alpha_vantage_client import AlphaVantageClient
from src.data.models.market_data import APIError


class TestAPIClientFactory:
    """Test APIClientFactory class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = APIClientFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        assert self.factory.settings is not None
        assert self.factory.logger is not None
    
    def test_create_alpha_vantage_client(self):
        """Test creating Alpha Vantage client."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "test_key"
            mock_settings.return_value.is_development = False
            
            client = self.factory.create_client(DataProvider.ALPHA_VANTAGE)
            
            assert isinstance(client, AlphaVantageClient)
            assert client.api_key == "test_key"
    
    def test_create_alpha_vantage_client_demo_key_development(self):
        """Test creating Alpha Vantage client with demo key in development."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "demo"
            mock_settings.return_value.is_development = True
            
            # Should work in development with demo key
            client = self.factory.create_client(DataProvider.ALPHA_VANTAGE)
            assert isinstance(client, AlphaVantageClient)
            assert client.api_key == "demo"
    
    def test_create_alpha_vantage_client_demo_key_production(self):
        """Test creating Alpha Vantage client with demo key in production."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "demo"
            mock_settings.return_value.is_development = False
            
            # Should raise error in production with demo key
            with pytest.raises(APIError) as exc_info:
                self.factory.create_client(DataProvider.ALPHA_VANTAGE)
            
            assert "not configured" in str(exc_info.value)
    
    def test_create_client_string_provider(self):
        """Test creating client with string provider."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "test_key"
            mock_settings.return_value.is_development = False
            
            client = self.factory.create_client("alpha_vantage")
            assert isinstance(client, AlphaVantageClient)
    
    def test_create_client_invalid_provider(self):
        """Test creating client with invalid provider."""
        with pytest.raises(ValueError) as exc_info:
            self.factory.create_client("invalid_provider")
        
        assert "Unsupported provider" in str(exc_info.value)
    
    def test_create_polygon_client_not_implemented(self):
        """Test creating Polygon client (not yet implemented)."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.polygon_api_key = "test_key"
            
            with pytest.raises(NotImplementedError):
                self.factory.create_client(DataProvider.POLYGON)
    
    def test_create_iex_client_not_implemented(self):
        """Test creating IEX client (not yet implemented)."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.iex_cloud_api_key = "test_key"
            
            with pytest.raises(NotImplementedError):
                self.factory.create_client(DataProvider.IEX)
    
    def test_create_polygon_client_no_key(self):
        """Test creating Polygon client without API key."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.polygon_api_key = None
            
            with pytest.raises(APIError) as exc_info:
                self.factory.create_client(DataProvider.POLYGON)
            
            assert "not configured" in str(exc_info.value)
    
    def test_create_iex_client_no_key(self):
        """Test creating IEX client without API key."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.iex_cloud_api_key = None
            
            with pytest.raises(APIError) as exc_info:
                self.factory.create_client(DataProvider.IEX)
            
            assert "not configured" in str(exc_info.value)
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "real_key"
            mock_settings.return_value.external_apis.polygon_api_key = "polygon_key"
            mock_settings.return_value.external_apis.iex_cloud_api_key = None
            
            providers = self.factory.get_available_providers()
            
            assert providers[DataProvider.ALPHA_VANTAGE] is True
            assert providers[DataProvider.POLYGON] is True
            assert providers[DataProvider.IEX] is False
    
    def test_get_available_providers_demo_key(self):
        """Test available providers with demo key."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "demo"
            mock_settings.return_value.external_apis.polygon_api_key = None
            mock_settings.return_value.external_apis.iex_cloud_api_key = None
            
            providers = self.factory.get_available_providers()
            
            # Demo key should not be considered as available
            assert providers[DataProvider.ALPHA_VANTAGE] is False
    
    def test_get_default_provider_with_available(self):
        """Test getting default provider when providers are available."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "real_key"
            mock_settings.return_value.external_apis.polygon_api_key = None
            mock_settings.return_value.external_apis.iex_cloud_api_key = None
            
            default = self.factory.get_default_provider()
            assert default == DataProvider.ALPHA_VANTAGE
    
    def test_get_default_provider_development_fallback(self):
        """Test getting default provider with development fallback."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "demo"
            mock_settings.return_value.external_apis.polygon_api_key = None
            mock_settings.return_value.external_apis.iex_cloud_api_key = None
            mock_settings.return_value.is_development = True
            
            default = self.factory.get_default_provider()
            assert default == DataProvider.ALPHA_VANTAGE  # Should fallback to AV in dev
    
    def test_get_default_provider_no_providers(self):
        """Test getting default provider when no providers are available."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = None
            mock_settings.return_value.external_apis.polygon_api_key = None
            mock_settings.return_value.external_apis.iex_cloud_api_key = None
            mock_settings.return_value.is_development = False
            
            with pytest.raises(APIError) as exc_info:
                self.factory.get_default_provider()
            
            assert "No configured data providers" in str(exc_info.value)
    
    def test_create_client_with_custom_config(self):
        """Test creating client with custom configuration."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "test_key"
            mock_settings.return_value.is_development = False
            
            client = self.factory.create_client(
                DataProvider.ALPHA_VANTAGE,
                timeout=60,
                max_retries=5
            )
            
            assert isinstance(client, AlphaVantageClient)
            assert client.timeout == 60
            assert client.max_retries == 5


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_default_client(self):
        """Test get_default_client function."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "test_key"
            mock_settings.return_value.is_development = False
            
            client = get_default_client()
            assert isinstance(client, AlphaVantageClient)
    
    def test_get_alpha_vantage_client(self):
        """Test get_alpha_vantage_client function."""
        with patch('src.data.ingestion.client_factory.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "test_key"
            mock_settings.return_value.is_development = False
            
            client = get_alpha_vantage_client()
            assert isinstance(client, AlphaVantageClient)
    
    @pytest.mark.asyncio
    async def test_test_client_connection_success(self):
        """Test successful client connection test."""
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.error_message = None
        
        with patch('src.data.ingestion.client_factory.APIClientFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_client = MagicMock()
            mock_client.__aenter__ = MagicMock(return_value=mock_client)
            mock_client.__aexit__ = MagicMock(return_value=None)
            mock_client.search_symbols = MagicMock(return_value=mock_response)
            mock_client.get_rate_limit_info = MagicMock(return_value=None)
            
            mock_factory.create_client = MagicMock(return_value=mock_client)
            mock_factory_class.return_value = mock_factory
            
            from src.data.ingestion.client_factory import test_client_connection
            result = await test_client_connection(DataProvider.ALPHA_VANTAGE)
            
            assert result["success"] is True
            assert result["provider"] == str(DataProvider.ALPHA_VANTAGE)
            assert result["error"] is None
    
    @pytest.mark.asyncio
    async def test_test_client_connection_failure(self):
        """Test failed client connection test."""
        with patch('src.data.ingestion.client_factory.APIClientFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create_client.side_effect = APIError("Connection failed")
            mock_factory_class.return_value = mock_factory
            
            from src.data.ingestion.client_factory import test_client_connection
            result = await test_client_connection(DataProvider.ALPHA_VANTAGE)
            
            assert result["success"] is False
            assert "Connection failed" in result["error"]
            assert result["rate_limit_info"] is None


class TestDataProvider:
    """Test DataProvider enum."""
    
    def test_data_provider_values(self):
        """Test DataProvider enum values."""
        assert DataProvider.ALPHA_VANTAGE == "alpha_vantage"
        assert DataProvider.POLYGON == "polygon"
        assert DataProvider.IEX == "iex"
    
    def test_data_provider_from_string(self):
        """Test creating DataProvider from string."""
        assert DataProvider("alpha_vantage") == DataProvider.ALPHA_VANTAGE
        assert DataProvider("polygon") == DataProvider.POLYGON
        assert DataProvider("iex") == DataProvider.IEX
    
    def test_data_provider_invalid_string(self):
        """Test creating DataProvider from invalid string."""
        with pytest.raises(ValueError):
            DataProvider("invalid_provider")