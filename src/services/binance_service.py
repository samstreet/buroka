"""
Binance API Service for Crypto Market Data

This service handles all interactions with the Binance API including:
- Real-time price data via WebSocket
- Historical kline/candlestick data
- 24hr ticker statistics  
- Symbol information
- Trading pair data
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import websockets
from dataclasses import dataclass, asdict
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class CryptoTicker:
    """Crypto ticker data structure"""
    symbol: str
    price: float
    price_change: float
    price_change_percent: float
    high_price: float
    low_price: float
    open_price: float
    volume: float
    quote_volume: float
    open_time: datetime
    close_time: datetime
    count: int  # Trade count

@dataclass
class CryptoKline:
    """Crypto kline/candlestick data structure"""
    symbol: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float

@dataclass
class CryptoSymbolInfo:
    """Crypto symbol information"""
    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    base_asset_precision: int
    quote_asset_precision: int
    filters: List[Dict[str, Any]]

class BinanceService:
    """Service for interacting with Binance API"""
    
    BASE_URL = "https://api.binance.com"
    WS_BASE_URL = "wss://fstream.binance.com"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, List[Callable]] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        
        # Close all websocket connections
        for ws in self.websocket_connections.values():
            await ws.close()
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including all trading pairs"""
        try:
            async with self.session.get(f"{self.BASE_URL}/api/v3/exchangeInfo") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to get exchange info: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return {}
    
    async def get_active_crypto_symbols(self) -> List[CryptoSymbolInfo]:
        """Get all active cryptocurrency trading pairs"""
        exchange_info = await self.get_exchange_info()
        symbols = []
        
        for symbol_data in exchange_info.get('symbols', []):
            if symbol_data.get('status') == 'TRADING':
                symbol_info = CryptoSymbolInfo(
                    symbol=symbol_data['symbol'],
                    base_asset=symbol_data['baseAsset'],
                    quote_asset=symbol_data['quoteAsset'],
                    status=symbol_data['status'],
                    base_asset_precision=symbol_data['baseAssetPrecision'],
                    quote_asset_precision=symbol_data['quotePrecision'],
                    filters=symbol_data['filters']
                )
                symbols.append(symbol_info)
        
        return symbols
    
    async def get_popular_crypto_pairs(self) -> List[str]:
        """Get popular cryptocurrency trading pairs"""
        popular_bases = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOGE', 'DOT', 'AVAX', 'MATIC', 'LTC']
        popular_quotes = ['USDT', 'BUSD', 'BTC', 'ETH']
        
        symbols = await self.get_active_crypto_symbols()
        popular_pairs = []
        
        for symbol in symbols:
            # USDT pairs for major coins
            if (symbol.base_asset in popular_bases and symbol.quote_asset == 'USDT'):
                popular_pairs.append(symbol.symbol)
            # BTC pairs for altcoins
            elif (symbol.base_asset in popular_bases and symbol.quote_asset == 'BTC'):
                popular_pairs.append(symbol.symbol)
        
        # Always include the top pairs
        must_include = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
                       'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LTCUSDT']
        
        for pair in must_include:
            if pair not in popular_pairs:
                popular_pairs.append(pair)
        
        return popular_pairs[:50]  # Return top 50
    
    async def get_24hr_ticker(self, symbol: Optional[str] = None) -> List[CryptoTicker]:
        """Get 24hr ticker price change statistics"""
        try:
            url = f"{self.BASE_URL}/api/v3/ticker/24hr"
            if symbol:
                url += f"?symbol={symbol}"
                
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle single symbol response
                    if symbol:
                        data = [data]
                    
                    tickers = []
                    for ticker_data in data:
                        ticker = CryptoTicker(
                            symbol=ticker_data['symbol'],
                            price=float(ticker_data['lastPrice']),
                            price_change=float(ticker_data['priceChange']),
                            price_change_percent=float(ticker_data['priceChangePercent']),
                            high_price=float(ticker_data['highPrice']),
                            low_price=float(ticker_data['lowPrice']),
                            open_price=float(ticker_data['openPrice']),
                            volume=float(ticker_data['volume']),
                            quote_volume=float(ticker_data['quoteVolume']),
                            open_time=datetime.fromtimestamp(ticker_data['openTime'] / 1000),
                            close_time=datetime.fromtimestamp(ticker_data['closeTime'] / 1000),
                            count=int(ticker_data['count'])
                        )
                        tickers.append(ticker)
                    
                    return tickers
                else:
                    logger.error(f"Failed to get 24hr ticker: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting 24hr ticker: {e}")
            return []
    
    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 1000, 
                        start_time: Optional[datetime] = None, 
                        end_time: Optional[datetime] = None) -> List[CryptoKline]:
        """Get kline/candlestick data for a symbol"""
        try:
            url = f"{self.BASE_URL}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    klines = []
                    for kline_data in data:
                        kline = CryptoKline(
                            symbol=symbol,
                            open_time=datetime.fromtimestamp(kline_data[0] / 1000),
                            close_time=datetime.fromtimestamp(kline_data[6] / 1000),
                            open_price=float(kline_data[1]),
                            high_price=float(kline_data[2]),
                            low_price=float(kline_data[3]),
                            close_price=float(kline_data[4]),
                            volume=float(kline_data[5]),
                            quote_asset_volume=float(kline_data[7]),
                            number_of_trades=int(kline_data[8]),
                            taker_buy_base_asset_volume=float(kline_data[9]),
                            taker_buy_quote_asset_volume=float(kline_data[10])
                        )
                        klines.append(kline)
                    
                    return klines
                else:
                    logger.error(f"Failed to get klines for {symbol}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    async def subscribe_to_ticker_stream(self, symbols: List[str], callback: Callable[[Dict], None]):
        """Subscribe to ticker price streams via WebSocket"""
        if not symbols:
            return
        
        # Convert symbols to lowercase for stream names
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        stream_name = "/".join(streams)
        
        ws_url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                self.websocket_connections['ticker'] = websocket
                
                logger.info(f"Connected to Binance ticker stream for {len(symbols)} symbols")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Handle individual ticker update
                        if 'e' in data and data['e'] == '24hrTicker':
                            ticker_update = {
                                'symbol': data['s'],
                                'price': float(data['c']),
                                'price_change': float(data['P']),
                                'price_change_percent': float(data['P']),
                                'high_price': float(data['h']),
                                'low_price': float(data['l']),
                                'volume': float(data['v']),
                                'quote_volume': float(data['q']),
                                'timestamp': datetime.fromtimestamp(data['E'] / 1000)
                            }
                            await callback(ticker_update)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing ticker update: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
    
    async def subscribe_to_kline_stream(self, symbol: str, interval: str, callback: Callable[[Dict], None]):
        """Subscribe to kline/candlestick data stream"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        ws_url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                self.websocket_connections[f'kline_{symbol}'] = websocket
                
                logger.info(f"Connected to Binance kline stream for {symbol} ({interval})")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if 'k' in data:
                            kline_data = data['k']
                            kline_update = {
                                'symbol': kline_data['s'],
                                'open_time': datetime.fromtimestamp(kline_data['t'] / 1000),
                                'close_time': datetime.fromtimestamp(kline_data['T'] / 1000),
                                'open_price': float(kline_data['o']),
                                'high_price': float(kline_data['h']),
                                'low_price': float(kline_data['l']),
                                'close_price': float(kline_data['c']),
                                'volume': float(kline_data['v']),
                                'is_closed': kline_data['x'],  # Whether this kline is closed
                                'timestamp': datetime.fromtimestamp(data['E'] / 1000)
                            }
                            await callback(kline_update)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse kline WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing kline update: {e}")
                        
        except Exception as e:
            logger.error(f"Kline WebSocket connection error: {e}")
    
    def get_binance_intervals(self) -> List[str]:
        """Get supported Binance kline intervals"""
        return [
            '1m', '3m', '5m', '15m', '30m',  # Minutes
            '1h', '2h', '4h', '6h', '8h', '12h',  # Hours
            '1d', '3d',  # Days
            '1w',  # Week
            '1M'   # Month
        ]
    
    async def get_top_volume_pairs(self, quote_asset: str = 'USDT', limit: int = 20) -> List[str]:
        """Get top volume trading pairs for a quote asset"""
        try:
            tickers = await self.get_24hr_ticker()
            
            # Filter by quote asset and sort by volume
            filtered_tickers = [
                ticker for ticker in tickers 
                if ticker.symbol.endswith(quote_asset)
            ]
            
            # Sort by quote volume (trading volume in quote asset)
            filtered_tickers.sort(key=lambda x: x.quote_volume, reverse=True)
            
            return [ticker.symbol for ticker in filtered_tickers[:limit]]
            
        except Exception as e:
            logger.error(f"Error getting top volume pairs: {e}")
            return []
    
    async def get_price_change_leaders(self, min_volume: float = 1000000, limit: int = 20) -> Dict[str, List[str]]:
        """Get biggest gainers and losers with minimum volume threshold"""
        try:
            tickers = await self.get_24hr_ticker()
            
            # Filter by minimum volume
            filtered_tickers = [
                ticker for ticker in tickers 
                if ticker.quote_volume >= min_volume and ticker.symbol.endswith('USDT')
            ]
            
            # Sort by price change percentage
            gainers = sorted(filtered_tickers, key=lambda x: x.price_change_percent, reverse=True)
            losers = sorted(filtered_tickers, key=lambda x: x.price_change_percent)
            
            return {
                'gainers': [ticker.symbol for ticker in gainers[:limit]],
                'losers': [ticker.symbol for ticker in losers[:limit]]
            }
            
        except Exception as e:
            logger.error(f"Error getting price change leaders: {e}")
            return {'gainers': [], 'losers': []}

# Global service instance
binance_service = BinanceService()

async def get_binance_service() -> BinanceService:
    """Get the global Binance service instance"""
    return binance_service