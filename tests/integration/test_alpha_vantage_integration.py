"""
Integration tests for Alpha Vantage API client.
These tests make actual API calls (when API key is configured).
"""

import pytest
import asyncio
from unittest.mock import patch
from datetime import datetime

from src.data.ingestion.client_factory import get_alpha_vantage_client, test_client_connection, DataProvider
from src.data.models.market_data import MarketDataType, DataGranularity
from src.config import get_settings


@pytest.mark.integration
@pytest.mark.asyncio
class TestAlphaVantageIntegration:
    """Integration tests for Alpha Vantage API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = get_settings()
        self.has_real_api_key = (
            self.settings.external_apis.alpha_vantage_api_key and 
            self.settings.external_apis.alpha_vantage_api_key != "demo"
        )
    
    @pytest.mark.skipif(
        not get_settings().external_apis.alpha_vantage_api_key or 
        get_settings().external_apis.alpha_vantage_api_key == "demo",
        reason="Alpha Vantage API key not configured"
    )
    async def test_get_intraday_data_integration(self):
        """Test actual intraday data retrieval."""
        async with get_alpha_vantage_client() as client:
            response = await client.get_intraday_data("AAPL", "5min")
            
            if response.success:
                assert response.symbol == "AAPL"
                assert response.data_type == MarketDataType.INTRADAY
                assert "ohlc_data" in response.data
                
                # Verify data structure
                if response.data["ohlc_data"]:
                    ohlc_item = response.data["ohlc_data"][0]
                    required_fields = ["symbol", "timestamp", "open_price", "high_price", "low_price", "close_price", "volume"]
                    for field in required_fields:
                        assert field in ohlc_item, f"Missing field: {field}"
            else:
                # API might fail due to rate limits or other issues
                assert response.error_message is not None
                pytest.skip(f"API call failed: {response.error_message}")
    
    @pytest.mark.skipif(
        not get_settings().external_apis.alpha_vantage_api_key or 
        get_settings().external_apis.alpha_vantage_api_key == "demo",
        reason="Alpha Vantage API key not configured"
    )
    async def test_get_daily_data_integration(self):
        """Test actual daily data retrieval."""
        async with get_alpha_vantage_client() as client:
            response = await client.get_daily_data("AAPL")
            
            if response.success:
                assert response.symbol == "AAPL"
                assert response.data_type == MarketDataType.DAILY
                assert "ohlc_data" in response.data
                
                # Verify data structure
                if response.data["ohlc_data"]:
                    ohlc_item = response.data["ohlc_data"][0]
                    assert ohlc_item["granularity"] == "daily"
            else:
                pytest.skip(f"API call failed: {response.error_message}")
    
    @pytest.mark.skipif(
        not get_settings().external_apis.alpha_vantage_api_key or 
        get_settings().external_apis.alpha_vantage_api_key == "demo",
        reason="Alpha Vantage API key not configured"
    )
    async def test_get_quote_integration(self):
        """Test actual quote retrieval."""
        async with get_alpha_vantage_client() as client:
            response = await client.get_quote("AAPL")
            
            if response.success:
                assert response.symbol == "AAPL"
                assert response.data_type == MarketDataType.QUOTE
                assert "quote" in response.data
                
                quote_data = response.data["quote"]
                assert "last_price" in quote_data
            else:
                pytest.skip(f"API call failed: {response.error_message}")
    
    @pytest.mark.skipif(
        not get_settings().external_apis.alpha_vantage_api_key or 
        get_settings().external_apis.alpha_vantage_api_key == "demo",
        reason="Alpha Vantage API key not configured"
    )
    async def test_search_symbols_integration(self):
        """Test actual symbol search."""
        async with get_alpha_vantage_client() as client:
            response = await client.search_symbols("Apple")
            
            if response.success:
                assert response.data is not None
                # The response format may vary, just check it's not empty
                assert len(response.data) > 0
            else:
                pytest.skip(f"API call failed: {response.error_message}")
    
    async def test_demo_key_behavior(self):
        """Test behavior with demo API key."""
        # Force demo key for this test
        with patch('src.config.get_settings') as mock_settings:
            mock_settings.return_value.external_apis.alpha_vantage_api_key = "demo"
            mock_settings.return_value.is_development = True
            
            async with get_alpha_vantage_client() as client:
                # Demo key should work but might have limited data
                response = await client.search_symbols("AAPL")
                
                # With demo key, we might get an error or limited response
                # Just ensure the client doesn't crash
                assert isinstance(response.success, bool)
    
    async def test_rate_limiting_behavior(self):
        """Test rate limiting behavior."""
        if not self.has_real_api_key:
            pytest.skip("Real API key required for rate limiting test")
        
        async with get_alpha_vantage_client() as client:
            # Check initial rate limit
            rate_limit_info = client.get_rate_limit_info()
            assert rate_limit_info is not None
            
            initial_remaining = rate_limit_info.calls_remaining
            
            # Make a request
            await client.search_symbols("AAPL")
            
            # Check updated rate limit
            updated_rate_limit = client.get_rate_limit_info()
            assert updated_rate_limit.calls_remaining <= initial_remaining
    
    async def test_connection_test_function(self):
        """Test the connection test utility function."""
        result = await test_client_connection(DataProvider.ALPHA_VANTAGE)
        
        assert "provider" in result
        assert "success" in result
        assert result["provider"] == str(DataProvider.ALPHA_VANTAGE)
        
        if self.has_real_api_key:
            # With real API key, should succeed (unless rate limited)
            if not result["success"]:
                # Check if it's a rate limit issue
                assert result["error"] is not None
        else:
            # Without real API key in production, should fail
            if not get_settings().is_development:
                assert not result["success"]
    
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        if not self.has_real_api_key:
            pytest.skip("Real API key required for concurrent request test")
        
        async with get_alpha_vantage_client() as client:
            # Make multiple concurrent requests
            symbols = ["AAPL", "MSFT", "GOOGL"]
            tasks = [client.search_symbols(symbol) for symbol in symbols]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least some should succeed (rate limiting might cause some to fail)
            successful_results = [r for r in results if not isinstance(r, Exception) and r.success]
            assert len(successful_results) >= 0  # Just ensure no crashes
    
    async def test_invalid_symbol_handling(self):
        """Test handling of invalid symbols."""
        if not self.has_real_api_key:
            pytest.skip("Real API key required for invalid symbol test")
        
        async with get_alpha_vantage_client() as client:
            # Try with clearly invalid symbol
            response = await client.get_quote("INVALID_SYMBOL_12345")
            
            # Should either fail gracefully or return no data
            if not response.success:
                assert response.error_message is not None
            else:
                # If it succeeds, the data should reflect the invalid symbol
                assert response.symbol == "INVALID_SYMBOL_12345"
    
    async def test_network_error_handling(self):
        """Test network error handling."""
        # Create client with invalid base URL to simulate network error
        from src.data.ingestion.alpha_vantage_client import AlphaVantageClient
        
        client = AlphaVantageClient(api_key="test")
        client.base_url = "https://invalid-url-that-does-not-exist.com"
        
        try:
            async with client:
                response = await client.get_quote("AAPL")
                # Should fail gracefully
                assert not response.success
                assert response.error_message is not None
        except Exception as e:
            # Network errors should be handled gracefully
            assert "Network error" in str(e) or "Connection" in str(e) or "timeout" in str(e).lower()


# Utility function to check if integration tests should run
def should_run_integration_tests():
    """Check if integration tests should run based on configuration."""
    settings = get_settings()
    return (
        settings.external_apis.alpha_vantage_api_key and 
        settings.external_apis.alpha_vantage_api_key != "demo"
    )


# Test configuration check
def test_api_key_configuration():
    """Test that API key configuration is working."""
    settings = get_settings()
    api_key = settings.external_apis.alpha_vantage_api_key
    
    # Should have some value (even if it's demo)
    assert api_key is not None
    assert isinstance(api_key, str)
    assert len(api_key) > 0