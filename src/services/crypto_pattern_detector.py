"""
Cryptocurrency-specific pattern detection service.
Handles unique characteristics of crypto markets including 24/7 trading,
different volatility patterns, and crypto-specific formations.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

from src.services.pattern_detection_service import PatternDetector, DetectorPriority
from src.data.models.pattern_models import (
    PatternType, PatternDirection, Timeframe, PatternDetectionResult
)

logger = logging.getLogger(__name__)


@dataclass
class CryptoPatternConfig:
    """Configuration for crypto-specific pattern detection."""
    min_volume_threshold: float = 1000000  # Minimum volume in USDT
    volatility_threshold: float = 0.15     # 15% threshold for high volatility
    pump_detection_threshold: float = 0.20  # 20% price increase threshold
    dump_detection_threshold: float = -0.15  # 15% price decrease threshold
    whale_volume_multiplier: float = 5.0   # 5x normal volume for whale detection
    support_resistance_tolerance: float = 0.02  # 2% tolerance for support/resistance


class CryptoVolumePatternDetector(PatternDetector):
    """Detects crypto-specific volume patterns like whale movements."""
    
    def __init__(self, config: CryptoPatternConfig = CryptoPatternConfig()):
        super().__init__("CryptoVolumePatternDetector", PatternType.VOLUME, DetectorPriority.HIGH)
        self.config = config
        
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        if not self.validate_data(data):
            return []
            
        results = []
        
        # Detect whale movements (unusual volume spikes)
        whale_patterns = await self._detect_whale_movements(data, symbol, timeframe)
        results.extend(whale_patterns)
        
        # Detect accumulation/distribution patterns
        accumulation_patterns = await self._detect_accumulation_distribution(data, symbol, timeframe)
        results.extend(accumulation_patterns)
        
        # Detect volume breakouts
        breakout_patterns = await self._detect_volume_breakouts(data, symbol, timeframe)
        results.extend(breakout_patterns)
        
        return results
    
    async def _detect_whale_movements(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        # Calculate volume moving average
        volume_ma = data['volume'].rolling(window=20).mean()
        
        for i in range(20, len(data)):
            current_volume = data['volume'].iloc[i]
            avg_volume = volume_ma.iloc[i]
            
            # Check for whale movement (volume spike)
            if current_volume > avg_volume * self.config.whale_volume_multiplier:
                price_change = (data['close'].iloc[i] - data['open'].iloc[i]) / data['open'].iloc[i]
                
                direction = PatternDirection.BULLISH if price_change > 0 else PatternDirection.BEARISH
                confidence = min(0.95, (current_volume / avg_volume) / self.config.whale_volume_multiplier)
                
                result = PatternDetectionResult(
                    pattern_type=self.pattern_type,
                    pattern_name="Whale Movement",
                    direction=direction,
                    confidence=confidence,
                    timeframe=timeframe,
                    start_time=data.index[i],
                    end_time=data.index[i],
                    symbol=symbol,
                    entry_price=data['close'].iloc[i],
                    strength="strong" if confidence > 0.8 else "moderate",
                    metadata={
                        'volume_multiple': current_volume / avg_volume,
                        'price_change_percent': price_change * 100,
                        'volume_usdt': current_volume * data['close'].iloc[i]
                    }
                )
                results.append(result)
                
        return results
    
    async def _detect_accumulation_distribution(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        # Calculate Accumulation/Distribution Line
        money_flow_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        money_flow_volume = money_flow_multiplier * data['volume']
        ad_line = money_flow_volume.cumsum()
        
        # Look for divergences
        price_ma = data['close'].rolling(window=10).mean()
        ad_ma = ad_line.rolling(window=10).mean()
        
        for i in range(20, len(data) - 5):
            # Check for bullish divergence (price falling, A/D rising)
            price_trend = price_ma.iloc[i] - price_ma.iloc[i-10]
            ad_trend = ad_ma.iloc[i] - ad_ma.iloc[i-10]
            
            if price_trend < 0 and ad_trend > 0:  # Bullish divergence
                confidence = min(0.9, abs(ad_trend) / abs(price_trend) * 0.1)
                
                result = PatternDetectionResult(
                    pattern_type=self.pattern_type,
                    pattern_name="Accumulation Divergence",
                    direction=PatternDirection.BULLISH,
                    confidence=confidence,
                    timeframe=timeframe,
                    start_time=data.index[i-10],
                    end_time=data.index[i],
                    symbol=symbol,
                    entry_price=data['close'].iloc[i],
                    target_price=data['close'].iloc[i] * 1.05,  # 5% target
                    stop_loss=data['close'].iloc[i] * 0.97,     # 3% stop loss
                    metadata={
                        'price_trend': price_trend,
                        'ad_trend': ad_trend,
                        'divergence_strength': abs(ad_trend) / abs(price_trend)
                    }
                )
                results.append(result)
                
            elif price_trend > 0 and ad_trend < 0:  # Bearish divergence
                confidence = min(0.9, abs(ad_trend) / abs(price_trend) * 0.1)
                
                result = PatternDetectionResult(
                    pattern_type=self.pattern_type,
                    pattern_name="Distribution Divergence",
                    direction=PatternDirection.BEARISH,
                    confidence=confidence,
                    timeframe=timeframe,
                    start_time=data.index[i-10],
                    end_time=data.index[i],
                    symbol=symbol,
                    entry_price=data['close'].iloc[i],
                    target_price=data['close'].iloc[i] * 0.95,  # 5% target
                    stop_loss=data['close'].iloc[i] * 1.03,     # 3% stop loss
                    metadata={
                        'price_trend': price_trend,
                        'ad_trend': ad_trend,
                        'divergence_strength': abs(ad_trend) / abs(price_trend)
                    }
                )
                results.append(result)
                
        return results
    
    async def _detect_volume_breakouts(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        # Calculate price breakouts with volume confirmation
        high_ma = data['high'].rolling(window=20).max()
        low_ma = data['low'].rolling(window=20).min()
        volume_ma = data['volume'].rolling(window=20).mean()
        
        for i in range(20, len(data)):
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            current_volume = data['volume'].iloc[i]
            
            resistance_level = high_ma.iloc[i-1]
            support_level = low_ma.iloc[i-1]
            avg_volume = volume_ma.iloc[i-1]
            
            # Bullish breakout above resistance with volume
            if (current_high > resistance_level and 
                current_volume > avg_volume * 1.5 and
                data['close'].iloc[i] > resistance_level):
                
                confidence = min(0.9, (current_volume / avg_volume) * 0.2)
                
                result = PatternDetectionResult(
                    pattern_type=self.pattern_type,
                    pattern_name="Volume Breakout (Bullish)",
                    direction=PatternDirection.BULLISH,
                    confidence=confidence,
                    timeframe=timeframe,
                    start_time=data.index[i],
                    end_time=data.index[i],
                    symbol=symbol,
                    entry_price=data['close'].iloc[i],
                    target_price=resistance_level + (resistance_level - support_level) * 0.5,
                    stop_loss=resistance_level * 0.98,
                    metadata={
                        'resistance_level': resistance_level,
                        'volume_multiple': current_volume / avg_volume,
                        'breakout_strength': (current_high - resistance_level) / resistance_level
                    }
                )
                results.append(result)
                
            # Bearish breakdown below support with volume
            elif (current_low < support_level and 
                  current_volume > avg_volume * 1.5 and
                  data['close'].iloc[i] < support_level):
                
                confidence = min(0.9, (current_volume / avg_volume) * 0.2)
                
                result = PatternDetectionResult(
                    pattern_type=self.pattern_type,
                    pattern_name="Volume Breakdown (Bearish)",
                    direction=PatternDirection.BEARISH,
                    confidence=confidence,
                    timeframe=timeframe,
                    start_time=data.index[i],
                    end_time=data.index[i],
                    symbol=symbol,
                    entry_price=data['close'].iloc[i],
                    target_price=support_level - (resistance_level - support_level) * 0.5,
                    stop_loss=support_level * 1.02,
                    metadata={
                        'support_level': support_level,
                        'volume_multiple': current_volume / avg_volume,
                        'breakdown_strength': (support_level - current_low) / support_level
                    }
                )
                results.append(result)
                
        return results


class CryptoPumpDumpDetector(PatternDetector):
    """Detects pump and dump patterns specific to crypto markets."""
    
    def __init__(self, config: CryptoPatternConfig = CryptoPatternConfig()):
        super().__init__("CryptoPumpDumpDetector", PatternType.PRICE, DetectorPriority.CRITICAL)
        self.config = config
        
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        if not self.validate_data(data):
            return []
            
        results = []
        
        # Detect pump patterns
        pump_patterns = await self._detect_pumps(data, symbol, timeframe)
        results.extend(pump_patterns)
        
        # Detect dump patterns
        dump_patterns = await self._detect_dumps(data, symbol, timeframe)
        results.extend(dump_patterns)
        
        # Detect pump and dump sequences
        pump_dump_patterns = await self._detect_pump_dump_sequences(data, symbol, timeframe)
        results.extend(pump_dump_patterns)
        
        return results
    
    async def _detect_pumps(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        for i in range(10, len(data)):
            # Look for rapid price increases
            price_change = (data['close'].iloc[i] - data['open'].iloc[i-5]) / data['open'].iloc[i-5]
            volume_spike = data['volume'].iloc[i-5:i+1].mean() / data['volume'].iloc[i-20:i-5].mean()
            
            if (price_change > self.config.pump_detection_threshold and 
                volume_spike > 2.0):  # Volume spike accompanies price pump
                
                confidence = min(0.95, price_change / self.config.pump_detection_threshold * 0.5 + 
                               min(0.4, volume_spike / 5.0))
                
                result = PatternDetectionResult(
                    pattern_type=self.pattern_type,
                    pattern_name="Crypto Pump",
                    direction=PatternDirection.BEARISH,  # Pumps usually lead to dumps
                    confidence=confidence,
                    timeframe=timeframe,
                    start_time=data.index[i-5],
                    end_time=data.index[i],
                    symbol=symbol,
                    entry_price=data['close'].iloc[i],
                    target_price=data['close'].iloc[i] * 0.85,  # Expect 15% correction
                    stop_loss=data['close'].iloc[i] * 1.05,     # 5% stop loss (in case of continued pump)
                    strength="strong" if confidence > 0.8 else "moderate",
                    metadata={
                        'price_change_percent': price_change * 100,
                        'volume_spike_multiple': volume_spike,
                        'pump_duration_bars': 6,
                        'warning': 'Potential pump detected - high risk of reversal'
                    }
                )
                results.append(result)
                
        return results
    
    async def _detect_dumps(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        for i in range(10, len(data)):
            # Look for rapid price decreases
            price_change = (data['close'].iloc[i] - data['open'].iloc[i-3]) / data['open'].iloc[i-3]
            volume_spike = data['volume'].iloc[i-3:i+1].mean() / data['volume'].iloc[i-15:i-3].mean()
            
            if (price_change < self.config.dump_detection_threshold and 
                volume_spike > 1.5):  # Volume spike accompanies price dump
                
                confidence = min(0.95, abs(price_change) / abs(self.config.dump_detection_threshold) * 0.5 + 
                               min(0.4, volume_spike / 4.0))
                
                result = PatternDetectionResult(
                    pattern_type=self.pattern_type,
                    pattern_name="Crypto Dump",
                    direction=PatternDirection.BULLISH,  # After dump, potential bounce
                    confidence=confidence,
                    timeframe=timeframe,
                    start_time=data.index[i-3],
                    end_time=data.index[i],
                    symbol=symbol,
                    entry_price=data['close'].iloc[i],
                    target_price=data['close'].iloc[i] * 1.10,  # Expect 10% bounce
                    stop_loss=data['close'].iloc[i] * 0.95,     # 5% stop loss
                    strength="strong" if confidence > 0.8 else "moderate",
                    metadata={
                        'price_change_percent': price_change * 100,
                        'volume_spike_multiple': volume_spike,
                        'dump_duration_bars': 4,
                        'opportunity': 'Potential oversold bounce opportunity'
                    }
                )
                results.append(result)
                
        return results
    
    async def _detect_pump_dump_sequences(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        # Look for pump followed by dump pattern
        for i in range(20, len(data) - 10):
            # First, identify pump phase (5-10 bars)
            pump_start = i - 10
            pump_end = i - 5
            pump_change = (data['close'].iloc[pump_end] - data['close'].iloc[pump_start]) / data['close'].iloc[pump_start]
            
            # Then, identify dump phase (next 5-10 bars)
            dump_start = i - 5
            dump_end = i
            dump_change = (data['close'].iloc[dump_end] - data['close'].iloc[dump_start]) / data['close'].iloc[dump_start]
            
            if (pump_change > 0.15 and dump_change < -0.10):  # 15% pump followed by 10% dump
                volume_during_pump = data['volume'].iloc[pump_start:pump_end].mean()
                volume_during_dump = data['volume'].iloc[dump_start:dump_end].mean()
                normal_volume = data['volume'].iloc[i-30:pump_start].mean()
                
                if (volume_during_pump > normal_volume * 2.0 and 
                    volume_during_dump > normal_volume * 1.5):
                    
                    confidence = min(0.95, 
                                   (pump_change / 0.15) * 0.3 + 
                                   (abs(dump_change) / 0.10) * 0.3 + 
                                   min(0.4, (volume_during_pump + volume_during_dump) / (normal_volume * 4)))
                    
                    result = PatternDetectionResult(
                        pattern_type=self.pattern_type,
                        pattern_name="Pump and Dump Sequence",
                        direction=PatternDirection.BEARISH,  # Generally bearish after P&D
                        confidence=confidence,
                        timeframe=timeframe,
                        start_time=data.index[pump_start],
                        end_time=data.index[dump_end],
                        symbol=symbol,
                        entry_price=data['close'].iloc[dump_end],
                        target_price=data['close'].iloc[pump_start],  # Return to pre-pump level
                        stop_loss=data['close'].iloc[dump_end] * 1.05,
                        strength="strong",
                        metadata={
                            'pump_change_percent': pump_change * 100,
                            'dump_change_percent': dump_change * 100,
                            'pump_volume_multiple': volume_during_pump / normal_volume,
                            'dump_volume_multiple': volume_during_dump / normal_volume,
                            'sequence_duration_bars': dump_end - pump_start + 1,
                            'warning': 'Manipulated price action detected - high risk'
                        }
                    )
                    results.append(result)
                    
        return results


class CryptoSupportResistanceDetector(PatternDetector):
    """Detects support and resistance levels specific to crypto markets."""
    
    def __init__(self, config: CryptoPatternConfig = CryptoPatternConfig()):
        super().__init__("CryptoSupportResistanceDetector", PatternType.STRUCTURE, DetectorPriority.MEDIUM)
        self.config = config
        
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        if not self.validate_data(data):
            return []
            
        results = []
        
        # Detect psychological levels (round numbers)
        psychological_patterns = await self._detect_psychological_levels(data, symbol, timeframe)
        results.extend(psychological_patterns)
        
        # Detect dynamic support/resistance
        dynamic_patterns = await self._detect_dynamic_levels(data, symbol, timeframe)
        results.extend(dynamic_patterns)
        
        return results
    
    async def _detect_psychological_levels(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        current_price = data['close'].iloc[-1]
        
        # Define psychological levels (round numbers)
        if current_price < 1:
            # For small coins, use different intervals
            intervals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        elif current_price < 10:
            intervals = [1, 5, 10]
        elif current_price < 100:
            intervals = [10, 25, 50, 75, 100]
        else:
            intervals = [100, 250, 500, 750, 1000, 2500, 5000, 10000]
        
        for level in intervals:
            distance = abs(current_price - level) / current_price
            
            if distance < self.config.support_resistance_tolerance:
                # Check how price reacted to this level historically
                touches = 0
                bounces = 0
                
                for i in range(len(data)):
                    if abs(data['low'].iloc[i] - level) / level < self.config.support_resistance_tolerance:
                        touches += 1
                        # Check if price bounced up from this level
                        if i < len(data) - 5 and data['close'].iloc[i+3] > data['close'].iloc[i] * 1.02:
                            bounces += 1
                    elif abs(data['high'].iloc[i] - level) / level < self.config.support_resistance_tolerance:
                        touches += 1
                        # Check if price rejected down from this level
                        if i < len(data) - 5 and data['close'].iloc[i+3] < data['close'].iloc[i] * 0.98:
                            bounces += 1
                
                if touches >= 3:  # Level has been tested multiple times
                    confidence = min(0.9, (bounces / touches) * 0.7 + min(0.2, touches / 10))
                    
                    level_type = "Support" if current_price > level else "Resistance"
                    direction = PatternDirection.BULLISH if level_type == "Support" else PatternDirection.BEARISH
                    
                    result = PatternDetectionResult(
                        pattern_type=self.pattern_type,
                        pattern_name=f"Psychological {level_type}",
                        direction=direction,
                        confidence=confidence,
                        timeframe=timeframe,
                        start_time=data.index[0],
                        end_time=data.index[-1],
                        symbol=symbol,
                        entry_price=current_price,
                        target_price=level * 1.05 if level_type == "Resistance" else level * 0.95,
                        stop_loss=level * 0.98 if level_type == "Support" else level * 1.02,
                        metadata={
                            'level_price': level,
                            'touches': touches,
                            'bounces': bounces,
                            'bounce_rate': bounces / touches if touches > 0 else 0,
                            'distance_from_level': distance,
                            'level_strength': touches
                        }
                    )
                    results.append(result)
                    
        return results
    
    async def _detect_dynamic_levels(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        results = []
        
        # Use moving averages as dynamic support/resistance
        periods = [20, 50, 100, 200]
        
        for period in periods:
            if len(data) < period:
                continue
                
            ma = data['close'].rolling(window=period).mean()
            current_price = data['close'].iloc[-1]
            current_ma = ma.iloc[-1]
            
            # Check how price interacts with this MA
            distance = abs(current_price - current_ma) / current_ma
            
            if distance < 0.05:  # Within 5% of MA
                # Count recent touches and bounces
                touches = 0
                bounces = 0
                
                for i in range(max(0, len(data) - 50), len(data) - 1):
                    if abs(data['close'].iloc[i] - ma.iloc[i]) / ma.iloc[i] < 0.03:
                        touches += 1
                        
                        # Check direction after touch
                        if i < len(data) - 3:
                            price_direction = data['close'].iloc[i+2] - data['close'].iloc[i]
                            ma_direction = ma.iloc[i+2] - ma.iloc[i] if i < len(data) - 2 else 0
                            
                            if price_direction * ma_direction > 0:  # Same direction = bounce
                                bounces += 1
                
                if touches >= 3:
                    confidence = min(0.85, (bounces / touches) * 0.6 + min(0.25, touches / 10))
                    
                    level_type = "Support" if current_price > current_ma else "Resistance"
                    direction = PatternDirection.BULLISH if level_type == "Support" else PatternDirection.BEARISH
                    
                    result = PatternDetectionResult(
                        pattern_type=self.pattern_type,
                        pattern_name=f"MA{period} Dynamic {level_type}",
                        direction=direction,
                        confidence=confidence,
                        timeframe=timeframe,
                        start_time=data.index[-period] if len(data) >= period else data.index[0],
                        end_time=data.index[-1],
                        symbol=symbol,
                        entry_price=current_price,
                        target_price=current_ma * 1.03 if level_type == "Resistance" else current_ma * 0.97,
                        stop_loss=current_ma * 0.97 if level_type == "Support" else current_ma * 1.03,
                        metadata={
                            'ma_period': period,
                            'ma_level': current_ma,
                            'touches': touches,
                            'bounces': bounces,
                            'bounce_rate': bounces / touches if touches > 0 else 0,
                            'distance_from_ma': distance,
                            'ma_slope': (ma.iloc[-1] - ma.iloc[-10]) / ma.iloc[-10] if len(data) >= 10 else 0
                        }
                    )
                    results.append(result)
                    
        return results


# Factory function to create crypto-specific pattern detection service
def create_crypto_pattern_service(repository, redis_client=None, config: CryptoPatternConfig = None):
    """Create a pattern detection service optimized for cryptocurrency markets."""
    from src.services.pattern_detection_service import PatternDetectionService
    
    if config is None:
        config = CryptoPatternConfig()
    
    service = PatternDetectionService(repository, redis_client)
    
    # Register crypto-specific detectors
    service.register_detector(CryptoVolumePatternDetector(config))
    service.register_detector(CryptoPumpDumpDetector(config))
    service.register_detector(CryptoSupportResistanceDetector(config))
    
    return service