"""
Volume-Price relationship analysis for market data.
Implements divergence detection, volume patterns, and signal confirmation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats, signal
from collections import defaultdict
import warnings

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DivergenceSignal:
    """Represents a price-volume divergence signal."""
    divergence_type: str  # 'bullish' or 'bearish'
    start_idx: int
    end_idx: int
    start_date: datetime
    end_date: datetime
    price_change_pct: float
    volume_change_pct: float
    strength: float  # 0-1 scale
    confirmed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VolumePattern:
    """Represents an unusual volume pattern."""
    pattern_type: str
    pattern_name: str
    start_idx: int
    end_idx: int
    start_date: datetime
    end_date: datetime
    avg_volume: float
    pattern_volume: float
    significance: float  # Statistical significance
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VolumeProfile:
    """Volume profile data structure."""
    price_levels: np.ndarray
    volume_at_price: np.ndarray
    poc: float  # Point of Control (price with highest volume)
    value_area_high: float
    value_area_low: float
    total_volume: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PriceVolumeDivergence:
    """
    Detects divergences between price and volume movements.
    """
    
    def __init__(self, lookback_period: int = 20, min_divergence: float = 0.1):
        """
        Initialize divergence detector.
        
        Args:
            lookback_period: Period for divergence calculation
            min_divergence: Minimum divergence threshold (10% default)
        """
        self.lookback_period = lookback_period
        self.min_divergence = min_divergence
    
    def detect_divergences(self, 
                          prices: pd.Series,
                          volumes: pd.Series) -> List[DivergenceSignal]:
        """
        Detect price-volume divergences.
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            List of divergence signals
        """
        if len(prices) != len(volumes):
            raise ValueError("Price and volume series must have same length")
        
        if len(prices) < self.lookback_period * 2:
            return []
        
        divergences = []
        
        # Calculate rolling correlations
        rolling_corr = self._calculate_rolling_correlation(prices, volumes)
        
        # Find peaks and troughs in price
        price_peaks = signal.argrelextrema(prices.values, np.greater, order=5)[0]
        price_troughs = signal.argrelextrema(prices.values, np.less, order=5)[0]
        
        # Check for divergences at peaks
        for peak_idx in price_peaks:
            if peak_idx >= self.lookback_period:
                divergence = self._check_peak_divergence(
                    peak_idx, prices, volumes, rolling_corr
                )
                if divergence:
                    divergences.append(divergence)
        
        # Check for divergences at troughs
        for trough_idx in price_troughs:
            if trough_idx >= self.lookback_period:
                divergence = self._check_trough_divergence(
                    trough_idx, prices, volumes, rolling_corr
                )
                if divergence:
                    divergences.append(divergence)
        
        # Check for continuous divergences
        continuous_divs = self._detect_continuous_divergence(prices, volumes)
        divergences.extend(continuous_divs)
        
        return divergences
    
    def _calculate_rolling_correlation(self, prices: pd.Series, 
                                      volumes: pd.Series) -> pd.Series:
        """Calculate rolling correlation between price and volume."""
        price_returns = prices.pct_change()
        volume_changes = volumes.pct_change()
        
        rolling_corr = price_returns.rolling(
            window=self.lookback_period
        ).corr(volume_changes)
        
        return rolling_corr
    
    def _check_peak_divergence(self, idx: int, prices: pd.Series,
                               volumes: pd.Series, rolling_corr: pd.Series) -> Optional[DivergenceSignal]:
        """Check for bearish divergence at price peak."""
        if idx < self.lookback_period or idx >= len(prices) - 5:
            return None
        
        # Look for previous peak
        prev_peaks = []
        for i in range(max(0, idx - self.lookback_period * 2), idx - 5):
            if i > 0 and prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                prev_peaks.append(i)
        
        if not prev_peaks:
            return None
        
        # Use the most recent previous peak
        prev_peak_idx = prev_peaks[-1]
        
        # Calculate price and volume changes
        price_change = (prices.iloc[idx] - prices.iloc[prev_peak_idx]) / prices.iloc[prev_peak_idx]
        
        # Calculate average volume between peaks
        volume_period_current = volumes.iloc[max(idx-5, 0):idx+1].mean()
        volume_period_prev = volumes.iloc[max(prev_peak_idx-5, 0):prev_peak_idx+1].mean()
        volume_change = (volume_period_current - volume_period_prev) / volume_period_prev
        
        # Bearish divergence: higher high in price, lower volume
        if price_change > 0 and volume_change < -self.min_divergence:
            # Calculate strength based on magnitude of divergence
            strength = min(1.0, abs(volume_change) / (abs(price_change) + 0.001))
            
            # Check correlation for confirmation
            avg_corr = rolling_corr.iloc[prev_peak_idx:idx].mean()
            confirmed = avg_corr < 0  # Negative correlation confirms divergence
            
            return DivergenceSignal(
                divergence_type="bearish",
                start_idx=prev_peak_idx,
                end_idx=idx,
                start_date=prices.index[prev_peak_idx],
                end_date=prices.index[idx],
                price_change_pct=price_change * 100,
                volume_change_pct=volume_change * 100,
                strength=strength,
                confirmed=confirmed,
                metadata={
                    "correlation": avg_corr,
                    "peak_price": prices.iloc[idx],
                    "prev_peak_price": prices.iloc[prev_peak_idx]
                }
            )
        
        return None
    
    def _check_trough_divergence(self, idx: int, prices: pd.Series,
                                volumes: pd.Series, rolling_corr: pd.Series) -> Optional[DivergenceSignal]:
        """Check for bullish divergence at price trough."""
        if idx < self.lookback_period or idx >= len(prices) - 5:
            return None
        
        # Look for previous trough
        prev_troughs = []
        for i in range(max(0, idx - self.lookback_period * 2), idx - 5):
            if i > 0 and prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                prev_troughs.append(i)
        
        if not prev_troughs:
            return None
        
        prev_trough_idx = prev_troughs[-1]
        
        # Calculate changes
        price_change = (prices.iloc[idx] - prices.iloc[prev_trough_idx]) / prices.iloc[prev_trough_idx]
        
        volume_period_current = volumes.iloc[max(idx-5, 0):idx+1].mean()
        volume_period_prev = volumes.iloc[max(prev_trough_idx-5, 0):prev_trough_idx+1].mean()
        volume_change = (volume_period_current - volume_period_prev) / volume_period_prev
        
        # Bullish divergence: lower low in price, higher volume (accumulation)
        if price_change < 0 and volume_change > self.min_divergence:
            strength = min(1.0, abs(volume_change) / (abs(price_change) + 0.001))
            
            avg_corr = rolling_corr.iloc[prev_trough_idx:idx].mean()
            confirmed = avg_corr < 0
            
            return DivergenceSignal(
                divergence_type="bullish",
                start_idx=prev_trough_idx,
                end_idx=idx,
                start_date=prices.index[prev_trough_idx],
                end_date=prices.index[idx],
                price_change_pct=price_change * 100,
                volume_change_pct=volume_change * 100,
                strength=strength,
                confirmed=confirmed,
                metadata={
                    "correlation": avg_corr,
                    "trough_price": prices.iloc[idx],
                    "prev_trough_price": prices.iloc[prev_trough_idx]
                }
            )
        
        return None
    
    def _detect_continuous_divergence(self, prices: pd.Series,
                                     volumes: pd.Series) -> List[DivergenceSignal]:
        """Detect continuous divergences over periods."""
        divergences = []
        
        # Calculate trends
        price_trend = prices.rolling(window=self.lookback_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        volume_trend = volumes.rolling(window=self.lookback_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        # Normalize trends
        price_trend_norm = price_trend / prices.rolling(window=self.lookback_period).mean()
        volume_trend_norm = volume_trend / volumes.rolling(window=self.lookback_period).mean()
        
        # Find periods of divergence
        i = self.lookback_period
        while i < len(prices) - self.lookback_period:
            # Check for significant opposite trends
            if (abs(price_trend_norm.iloc[i]) > 0.01 and 
                abs(volume_trend_norm.iloc[i]) > 0.01):
                
                # Bearish: price up, volume down
                if (price_trend_norm.iloc[i] > 0.01 and 
                    volume_trend_norm.iloc[i] < -0.01):
                    
                    # Find extent of divergence
                    end_idx = i
                    while (end_idx < len(prices) - 1 and
                           price_trend_norm.iloc[end_idx] > 0 and
                           volume_trend_norm.iloc[end_idx] < 0):
                        end_idx += 1
                    
                    divergence = DivergenceSignal(
                        divergence_type="bearish",
                        start_idx=i - self.lookback_period,
                        end_idx=end_idx,
                        start_date=prices.index[i - self.lookback_period],
                        end_date=prices.index[end_idx],
                        price_change_pct=((prices.iloc[end_idx] - prices.iloc[i - self.lookback_period]) / 
                                        prices.iloc[i - self.lookback_period] * 100),
                        volume_change_pct=((volumes.iloc[end_idx] - volumes.iloc[i - self.lookback_period]) / 
                                         volumes.iloc[i - self.lookback_period] * 100),
                        strength=min(1.0, abs(volume_trend_norm.iloc[i]) / abs(price_trend_norm.iloc[i])),
                        confirmed=True,
                        metadata={"trend_based": True}
                    )
                    divergences.append(divergence)
                    i = end_idx
                
                # Bullish: price down, volume up
                elif (price_trend_norm.iloc[i] < -0.01 and 
                      volume_trend_norm.iloc[i] > 0.01):
                    
                    end_idx = i
                    while (end_idx < len(prices) - 1 and
                           price_trend_norm.iloc[end_idx] < 0 and
                           volume_trend_norm.iloc[end_idx] > 0):
                        end_idx += 1
                    
                    divergence = DivergenceSignal(
                        divergence_type="bullish",
                        start_idx=i - self.lookback_period,
                        end_idx=end_idx,
                        start_date=prices.index[i - self.lookback_period],
                        end_date=prices.index[end_idx],
                        price_change_pct=((prices.iloc[end_idx] - prices.iloc[i - self.lookback_period]) / 
                                        prices.iloc[i - self.lookback_period] * 100),
                        volume_change_pct=((volumes.iloc[end_idx] - volumes.iloc[i - self.lookback_period]) / 
                                         volumes.iloc[i - self.lookback_period] * 100),
                        strength=min(1.0, abs(volume_trend_norm.iloc[i]) / abs(price_trend_norm.iloc[i])),
                        confirmed=True,
                        metadata={"trend_based": True}
                    )
                    divergences.append(divergence)
                    i = end_idx
            
            i += 1
        
        return divergences


class UnusualVolumePatterns:
    """
    Identifies unusual volume patterns that may signal important market events.
    """
    
    def __init__(self, lookback_period: int = 20, significance_level: float = 0.05):
        """
        Initialize unusual volume pattern detector.
        
        Args:
            lookback_period: Period for calculating normal volume
            significance_level: Statistical significance threshold
        """
        self.lookback_period = lookback_period
        self.significance_level = significance_level
    
    def detect_patterns(self,
                       prices: pd.Series,
                       volumes: pd.Series,
                       high: Optional[pd.Series] = None,
                       low: Optional[pd.Series] = None) -> List[VolumePattern]:
        """
        Detect unusual volume patterns.
        
        Args:
            prices: Close prices
            volumes: Volume data
            high: Optional high prices
            low: Optional low prices
            
        Returns:
            List of detected volume patterns
        """
        patterns = []
        
        # Detect various pattern types
        patterns.extend(self._detect_volume_climax(prices, volumes))
        patterns.extend(self._detect_volume_dry_up(prices, volumes))
        patterns.extend(self._detect_volume_breakout(prices, volumes))
        patterns.extend(self._detect_churning(prices, volumes, high, low))
        patterns.extend(self._detect_pocket_pivots(prices, volumes))
        
        return patterns
    
    def _detect_volume_climax(self, prices: pd.Series, 
                             volumes: pd.Series) -> List[VolumePattern]:
        """Detect volume climax patterns (extremely high volume)."""
        patterns = []
        
        # Calculate volume statistics
        volume_mean = volumes.rolling(window=self.lookback_period).mean()
        volume_std = volumes.rolling(window=self.lookback_period).std()
        
        # Z-score for volume
        z_scores = (volumes - volume_mean) / volume_std
        
        for i in range(self.lookback_period, len(volumes)):
            # Volume climax: z-score > 3
            if z_scores.iloc[i] > 3:
                # Determine if buying or selling climax
                price_change = prices.iloc[i] - prices.iloc[i-1] if i > 0 else 0
                
                if price_change > 0:
                    pattern_name = "buying_climax"
                elif price_change < 0:
                    pattern_name = "selling_climax"
                else:
                    pattern_name = "volume_climax"
                
                pattern = VolumePattern(
                    pattern_type="climax",
                    pattern_name=pattern_name,
                    start_idx=i,
                    end_idx=i,
                    start_date=volumes.index[i],
                    end_date=volumes.index[i],
                    avg_volume=volume_mean.iloc[i],
                    pattern_volume=volumes.iloc[i],
                    significance=1 - stats.norm.cdf(z_scores.iloc[i]),
                    metadata={
                        "z_score": z_scores.iloc[i],
                        "volume_ratio": volumes.iloc[i] / volume_mean.iloc[i],
                        "price_change": price_change
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_volume_dry_up(self, prices: pd.Series,
                             volumes: pd.Series) -> List[VolumePattern]:
        """Detect volume dry-up patterns (extremely low volume)."""
        patterns = []
        
        volume_mean = volumes.rolling(window=self.lookback_period).mean()
        volume_std = volumes.rolling(window=self.lookback_period).std()
        
        for i in range(self.lookback_period, len(volumes)):
            # Volume dry-up: volume < 50% of average
            if volumes.iloc[i] < volume_mean.iloc[i] * 0.5:
                # Check price action during dry-up
                if i >= 5:
                    price_volatility = prices.iloc[i-5:i+1].std() / prices.iloc[i-5:i+1].mean()
                    
                    pattern = VolumePattern(
                        pattern_type="dry_up",
                        pattern_name="volume_dry_up",
                        start_idx=i,
                        end_idx=i,
                        start_date=volumes.index[i],
                        end_date=volumes.index[i],
                        avg_volume=volume_mean.iloc[i],
                        pattern_volume=volumes.iloc[i],
                        significance=volumes.iloc[i] / volume_mean.iloc[i],
                        metadata={
                            "volume_ratio": volumes.iloc[i] / volume_mean.iloc[i],
                            "price_volatility": price_volatility,
                            "potential_breakout": price_volatility < 0.01  # Low volatility before breakout
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_volume_breakout(self, prices: pd.Series,
                               volumes: pd.Series) -> List[VolumePattern]:
        """Detect volume breakout patterns."""
        patterns = []
        
        volume_mean = volumes.rolling(window=self.lookback_period).mean()
        price_high = prices.rolling(window=self.lookback_period).max()
        price_low = prices.rolling(window=self.lookback_period).min()
        
        for i in range(self.lookback_period, len(prices)):
            # Check for price breakout with volume confirmation
            if volumes.iloc[i] > volume_mean.iloc[i] * 1.5:
                if prices.iloc[i] > price_high.iloc[i-1]:
                    # Upside breakout
                    pattern = VolumePattern(
                        pattern_type="breakout",
                        pattern_name="volume_breakout_up",
                        start_idx=i,
                        end_idx=i,
                        start_date=prices.index[i],
                        end_date=prices.index[i],
                        avg_volume=volume_mean.iloc[i],
                        pattern_volume=volumes.iloc[i],
                        significance=min(1.0, volumes.iloc[i] / volume_mean.iloc[i] / 2),
                        metadata={
                            "breakout_level": price_high.iloc[i-1],
                            "volume_surge": volumes.iloc[i] / volume_mean.iloc[i],
                            "price_move": (prices.iloc[i] - price_high.iloc[i-1]) / price_high.iloc[i-1]
                        }
                    )
                    patterns.append(pattern)
                
                elif prices.iloc[i] < price_low.iloc[i-1]:
                    # Downside breakout
                    pattern = VolumePattern(
                        pattern_type="breakout",
                        pattern_name="volume_breakout_down",
                        start_idx=i,
                        end_idx=i,
                        start_date=prices.index[i],
                        end_date=prices.index[i],
                        avg_volume=volume_mean.iloc[i],
                        pattern_volume=volumes.iloc[i],
                        significance=min(1.0, volumes.iloc[i] / volume_mean.iloc[i] / 2),
                        metadata={
                            "breakout_level": price_low.iloc[i-1],
                            "volume_surge": volumes.iloc[i] / volume_mean.iloc[i],
                            "price_move": (price_low.iloc[i-1] - prices.iloc[i]) / price_low.iloc[i-1]
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_churning(self, prices: pd.Series, volumes: pd.Series,
                        high: Optional[pd.Series], low: Optional[pd.Series]) -> List[VolumePattern]:
        """Detect churning (high volume with little price movement)."""
        patterns = []
        
        if high is None or low is None:
            return patterns
        
        volume_mean = volumes.rolling(window=self.lookback_period).mean()
        
        for i in range(self.lookback_period, len(prices)):
            # High volume
            if volumes.iloc[i] > volume_mean.iloc[i] * 1.5:
                # Calculate price range
                daily_range = (high.iloc[i] - low.iloc[i]) / prices.iloc[i]
                
                # Small range despite high volume = churning
                if daily_range < 0.01:  # Less than 1% range
                    pattern = VolumePattern(
                        pattern_type="churning",
                        pattern_name="volume_churning",
                        start_idx=i,
                        end_idx=i,
                        start_date=prices.index[i],
                        end_date=prices.index[i],
                        avg_volume=volume_mean.iloc[i],
                        pattern_volume=volumes.iloc[i],
                        significance=volumes.iloc[i] / volume_mean.iloc[i] * (1 - daily_range),
                        metadata={
                            "volume_ratio": volumes.iloc[i] / volume_mean.iloc[i],
                            "daily_range": daily_range,
                            "potential_distribution": prices.iloc[i] > prices.rolling(50).mean().iloc[i] if i >= 50 else False
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_pocket_pivots(self, prices: pd.Series,
                             volumes: pd.Series) -> List[VolumePattern]:
        """Detect pocket pivot patterns (O'Neil methodology)."""
        patterns = []
        
        for i in range(10, len(prices)):
            # Get down days in last 10 days
            down_days = []
            for j in range(i-10, i):
                if prices.iloc[j] < prices.iloc[j-1]:
                    down_days.append(j)
            
            if not down_days:
                continue
            
            # Get maximum down day volume
            max_down_volume = max(volumes.iloc[d] for d in down_days)
            
            # Pocket pivot: up day with volume > max down volume
            if (prices.iloc[i] > prices.iloc[i-1] and 
                volumes.iloc[i] > max_down_volume):
                
                # Additional criteria: stock should be near highs
                recent_high = prices.iloc[max(0, i-50):i].max() if i >= 50 else prices.iloc[:i].max()
                near_high = prices.iloc[i] >= recent_high * 0.95
                
                pattern = VolumePattern(
                    pattern_type="pocket_pivot",
                    pattern_name="pocket_pivot",
                    start_idx=i,
                    end_idx=i,
                    start_date=prices.index[i],
                    end_date=prices.index[i],
                    avg_volume=volumes.iloc[i-10:i].mean(),
                    pattern_volume=volumes.iloc[i],
                    significance=0.8 if near_high else 0.5,
                    metadata={
                        "max_down_volume": max_down_volume,
                        "volume_ratio": volumes.iloc[i] / max_down_volume,
                        "near_high": near_high,
                        "price_change": (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
                    }
                )
                patterns.append(pattern)
        
        return patterns


class AccumulationDistributionPatterns:
    """
    Identifies accumulation and distribution patterns in volume-price data.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize accumulation/distribution pattern detector.
        
        Args:
            period: Analysis period
        """
        self.period = period
    
    def detect_patterns(self,
                       open_p: pd.Series,
                       high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       volume: pd.Series) -> Dict[str, Any]:
        """
        Detect accumulation and distribution patterns.
        
        Returns:
            Dictionary with pattern analysis
        """
        patterns = {
            'wyckoff_phases': self._detect_wyckoff_phases(high, low, close, volume),
            'volume_by_price': self._analyze_volume_by_price(close, volume),
            'smart_money_flow': self._calculate_smart_money_flow(open_p, high, low, close, volume),
            'accumulation_days': self._identify_accumulation_days(close, volume),
            'distribution_days': self._identify_distribution_days(close, volume)
        }
        
        return patterns
    
    def _detect_wyckoff_phases(self, high: pd.Series, low: pd.Series,
                              close: pd.Series, volume: pd.Series) -> List[Dict]:
        """Detect Wyckoff accumulation/distribution phases."""
        phases = []
        
        # Simplified Wyckoff phase detection
        for i in range(self.period * 2, len(close)):
            window_close = close.iloc[i-self.period*2:i]
            window_volume = volume.iloc[i-self.period*2:i]
            window_high = high.iloc[i-self.period*2:i]
            window_low = low.iloc[i-self.period*2:i]
            
            # Calculate trading range
            range_high = window_high.max()
            range_low = window_low.min()
            range_size = range_high - range_low
            
            # Check if in trading range
            if range_size / close.iloc[i] < 0.15:  # Less than 15% range
                # Analyze volume pattern
                first_half_vol = window_volume.iloc[:self.period].mean()
                second_half_vol = window_volume.iloc[self.period:].mean()
                
                # Accumulation: decreasing volume in range
                if second_half_vol < first_half_vol * 0.8:
                    if close.iloc[i] > window_close.mean():
                        phases.append({
                            'phase': 'accumulation',
                            'stage': 'spring' if close.iloc[i] < range_low * 1.02 else 'markup',
                            'index': i,
                            'date': close.index[i],
                            'range_high': range_high,
                            'range_low': range_low,
                            'volume_pattern': 'decreasing'
                        })
                
                # Distribution: increasing volume at range high
                elif second_half_vol > first_half_vol * 1.2:
                    if close.iloc[i] < window_close.mean():
                        phases.append({
                            'phase': 'distribution',
                            'stage': 'upthrust' if close.iloc[i] > range_high * 0.98 else 'markdown',
                            'index': i,
                            'date': close.index[i],
                            'range_high': range_high,
                            'range_low': range_low,
                            'volume_pattern': 'increasing'
                        })
        
        return phases
    
    def _analyze_volume_by_price(self, close: pd.Series, 
                                volume: pd.Series) -> Dict[str, Any]:
        """Analyze volume distribution by price level."""
        # Create price bins
        price_min = close.min()
        price_max = close.max()
        n_bins = 20
        price_bins = np.linspace(price_min, price_max, n_bins + 1)
        
        # Accumulate volume by price level
        volume_by_price = np.zeros(n_bins)
        
        for i in range(len(close)):
            bin_idx = np.digitize(close.iloc[i], price_bins) - 1
            if 0 <= bin_idx < n_bins:
                volume_by_price[bin_idx] += volume.iloc[i]
        
        # Find key levels
        total_volume = volume_by_price.sum()
        cumulative_volume = np.cumsum(volume_by_price)
        
        # Point of Control (highest volume price)
        poc_idx = np.argmax(volume_by_price)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        value_area_volume = total_volume * 0.7
        value_area_start = np.where(cumulative_volume >= total_volume * 0.15)[0][0]
        value_area_end = np.where(cumulative_volume >= total_volume * 0.85)[0][0]
        
        return {
            'price_bins': price_bins,
            'volume_by_price': volume_by_price,
            'poc': poc_price,
            'value_area_high': price_bins[min(value_area_end + 1, len(price_bins) - 1)],
            'value_area_low': price_bins[value_area_start],
            'total_volume': total_volume
        }
    
    def _calculate_smart_money_flow(self, open_p: pd.Series, high: pd.Series,
                                   low: pd.Series, close: pd.Series,
                                   volume: pd.Series) -> pd.Series:
        """Calculate Smart Money Flow Index."""
        # First 30 minutes vs rest of day (simplified)
        money_flow = []
        
        for i in range(len(close)):
            # Typical price
            typical_price = (high.iloc[i] + low.iloc[i] + close.iloc[i]) / 3
            
            # Raw money flow
            raw_mf = typical_price * volume.iloc[i]
            
            # Smart money flow (emphasize closing action)
            if i > 0:
                if close.iloc[i] > close.iloc[i-1]:
                    smart_mf = raw_mf
                else:
                    smart_mf = -raw_mf
            else:
                smart_mf = 0
            
            money_flow.append(smart_mf)
        
        # Calculate SMF index
        smf_series = pd.Series(money_flow, index=close.index)
        smf_index = smf_series.rolling(window=self.period).sum()
        
        return smf_index
    
    def _identify_accumulation_days(self, close: pd.Series,
                                   volume: pd.Series) -> List[int]:
        """Identify accumulation days."""
        accumulation_days = []
        
        avg_volume = volume.rolling(window=50).mean()
        
        for i in range(1, len(close)):
            # Accumulation: up day on higher volume
            if (close.iloc[i] > close.iloc[i-1] * 1.002 and  # Up at least 0.2%
                volume.iloc[i] > avg_volume.iloc[i] * 1.1):   # Volume 10% above average
                accumulation_days.append(i)
        
        return accumulation_days
    
    def _identify_distribution_days(self, close: pd.Series,
                                   volume: pd.Series) -> List[int]:
        """Identify distribution days."""
        distribution_days = []
        
        avg_volume = volume.rolling(window=50).mean()
        
        for i in range(1, len(close)):
            # Distribution: down day on higher volume
            if (close.iloc[i] < close.iloc[i-1] * 0.998 and  # Down at least 0.2%
                volume.iloc[i] > avg_volume.iloc[i] * 1.1):   # Volume 10% above average
                distribution_days.append(i)
        
        return distribution_days


class VolumeConfirmation:
    """
    Provides volume confirmation for price patterns and signals.
    """
    
    def __init__(self, min_volume_increase: float = 1.5):
        """
        Initialize volume confirmation analyzer.
        
        Args:
            min_volume_increase: Minimum volume increase for confirmation
        """
        self.min_volume_increase = min_volume_increase
    
    def confirm_pattern(self,
                       pattern_type: str,
                       pattern_indices: List[int],
                       prices: pd.Series,
                       volumes: pd.Series) -> Dict[str, Any]:
        """
        Confirm a price pattern with volume analysis.
        
        Args:
            pattern_type: Type of pattern to confirm
            pattern_indices: Indices involved in the pattern
            prices: Price series
            volumes: Volume series
            
        Returns:
            Confirmation analysis
        """
        if not pattern_indices:
            return {'confirmed': False, 'reason': 'No pattern indices provided'}
        
        # Get average volume before pattern
        start_idx = min(pattern_indices)
        if start_idx < 20:
            baseline_volume = volumes.iloc[:start_idx].mean() if start_idx > 0 else volumes.iloc[0]
        else:
            baseline_volume = volumes.iloc[start_idx-20:start_idx].mean()
        
        # Get pattern volume
        pattern_volume = volumes.iloc[pattern_indices].mean()
        
        # Calculate volume ratio
        volume_ratio = pattern_volume / baseline_volume if baseline_volume > 0 else 0
        
        # Pattern-specific confirmation
        if pattern_type == 'breakout':
            confirmed = volume_ratio >= self.min_volume_increase
            confidence = min(1.0, volume_ratio / 2)
        elif pattern_type == 'reversal':
            # Reversal needs increasing volume
            volume_trend = np.polyfit(range(len(pattern_indices)), 
                                    volumes.iloc[pattern_indices].values, 1)[0]
            confirmed = volume_ratio > 1.2 and volume_trend > 0
            confidence = min(1.0, volume_ratio / 1.5)
        elif pattern_type == 'continuation':
            # Continuation can have lower volume
            confirmed = volume_ratio > 0.8
            confidence = min(1.0, volume_ratio)
        else:
            confirmed = volume_ratio >= 1.0
            confidence = min(1.0, volume_ratio)
        
        return {
            'confirmed': confirmed,
            'confidence': confidence,
            'volume_ratio': volume_ratio,
            'baseline_volume': baseline_volume,
            'pattern_volume': pattern_volume,
            'volume_trend': 'increasing' if pattern_indices and 
                          volumes.iloc[pattern_indices[-1]] > volumes.iloc[pattern_indices[0]] 
                          else 'decreasing'
        }


class VolumeProfileAnalysis:
    """
    Performs volume profile analysis for market structure identification.
    """
    
    def __init__(self, n_bins: int = 30):
        """
        Initialize volume profile analyzer.
        
        Args:
            n_bins: Number of price bins for profile
        """
        self.n_bins = n_bins
    
    def create_profile(self,
                      high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series,
                      period: Optional[int] = None) -> VolumeProfile:
        """
        Create volume profile for the given period.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: Optional period (uses all data if None)
            
        Returns:
            VolumeProfile object
        """
        if period:
            high = high.iloc[-period:]
            low = low.iloc[-period:]
            close = close.iloc[-period:]
            volume = volume.iloc[-period:]
        
        # Determine price range
        price_min = low.min()
        price_max = high.max()
        price_levels = np.linspace(price_min, price_max, self.n_bins + 1)
        
        # Calculate volume at each price level
        volume_at_price = np.zeros(self.n_bins)
        
        for i in range(len(close)):
            # Distribute volume across the day's range
            day_low = low.iloc[i]
            day_high = high.iloc[i]
            day_volume = volume.iloc[i]
            
            # Find bins that this day's range covers
            low_bin = np.searchsorted(price_levels, day_low, side='left')
            high_bin = np.searchsorted(price_levels, day_high, side='right')
            
            # Distribute volume evenly across bins (simplified)
            if high_bin > low_bin:
                volume_per_bin = day_volume / (high_bin - low_bin)
                for bin_idx in range(max(0, low_bin), min(self.n_bins, high_bin)):
                    volume_at_price[bin_idx] += volume_per_bin
        
        # Calculate key metrics
        total_volume = volume_at_price.sum()
        
        # Point of Control (POC)
        poc_idx = np.argmax(volume_at_price)
        poc = (price_levels[poc_idx] + price_levels[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        sorted_indices = np.argsort(volume_at_price)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_at_price[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= total_volume * 0.7:
                break
        
        value_area_indices = sorted(value_area_indices)
        value_area_low = price_levels[value_area_indices[0]]
        value_area_high = price_levels[min(value_area_indices[-1] + 1, len(price_levels) - 1)]
        
        return VolumeProfile(
            price_levels=price_levels,
            volume_at_price=volume_at_price,
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            total_volume=total_volume,
            metadata={
                'n_bins': self.n_bins,
                'price_range': (price_min, price_max),
                'value_area_volume': cumulative_volume,
                'value_area_percentage': (cumulative_volume / total_volume) * 100 if total_volume > 0 else 0
            }
        )
    
    def identify_support_resistance(self, profile: VolumeProfile) -> List[Dict[str, Any]]:
        """Identify support and resistance levels from volume profile."""
        levels = []
        
        # High volume nodes (HVN) - potential support/resistance
        threshold = np.percentile(profile.volume_at_price, 70)
        
        for i in range(len(profile.volume_at_price)):
            if profile.volume_at_price[i] > threshold:
                price_level = (profile.price_levels[i] + profile.price_levels[i + 1]) / 2
                
                levels.append({
                    'type': 'hvn',
                    'price': price_level,
                    'volume': profile.volume_at_price[i],
                    'strength': profile.volume_at_price[i] / profile.total_volume
                })
        
        # Low volume nodes (LVN) - potential breakout levels
        low_threshold = np.percentile(profile.volume_at_price, 30)
        
        for i in range(len(profile.volume_at_price)):
            if profile.volume_at_price[i] < low_threshold and profile.volume_at_price[i] > 0:
                price_level = (profile.price_levels[i] + profile.price_levels[i + 1]) / 2
                
                levels.append({
                    'type': 'lvn',
                    'price': price_level,
                    'volume': profile.volume_at_price[i],
                    'strength': 1 - (profile.volume_at_price[i] / profile.total_volume)
                })
        
        # Add POC and Value Area as key levels
        levels.append({
            'type': 'poc',
            'price': profile.poc,
            'volume': profile.volume_at_price[np.argmax(profile.volume_at_price)],
            'strength': 1.0
        })
        
        levels.append({
            'type': 'value_area_high',
            'price': profile.value_area_high,
            'volume': None,
            'strength': 0.8
        })
        
        levels.append({
            'type': 'value_area_low',
            'price': profile.value_area_low,
            'volume': None,
            'strength': 0.8
        })
        
        return sorted(levels, key=lambda x: x['price'])


class VolumeSignalStrength:
    """
    Calculates signal strength based on volume characteristics.
    """
    
    def __init__(self):
        self.weights = {
            'volume_surge': 0.3,
            'volume_trend': 0.2,
            'relative_volume': 0.2,
            'volume_consistency': 0.15,
            'price_volume_correlation': 0.15
        }
    
    def calculate_strength(self,
                          signal_idx: int,
                          prices: pd.Series,
                          volumes: pd.Series,
                          lookback: int = 20) -> Dict[str, Any]:
        """
        Calculate signal strength based on volume.
        
        Args:
            signal_idx: Index of the signal
            prices: Price series
            volumes: Volume series
            lookback: Lookback period for analysis
            
        Returns:
            Signal strength analysis
        """
        if signal_idx < lookback:
            lookback = signal_idx
        
        if lookback < 2:
            return {'strength': 0, 'components': {}}
        
        # Calculate components
        components = {}
        
        # 1. Volume surge
        avg_volume = volumes.iloc[signal_idx-lookback:signal_idx].mean()
        current_volume = volumes.iloc[signal_idx]
        components['volume_surge'] = min(1.0, current_volume / avg_volume / 2) if avg_volume > 0 else 0
        
        # 2. Volume trend
        volume_trend = np.polyfit(range(lookback), 
                                 volumes.iloc[signal_idx-lookback:signal_idx].values, 1)[0]
        volume_trend_norm = volume_trend / avg_volume if avg_volume > 0 else 0
        components['volume_trend'] = min(1.0, abs(volume_trend_norm) * 10)
        
        # 3. Relative volume
        if signal_idx >= 50:
            long_avg_volume = volumes.iloc[signal_idx-50:signal_idx].mean()
            components['relative_volume'] = min(1.0, current_volume / long_avg_volume) if long_avg_volume > 0 else 0
        else:
            components['relative_volume'] = components['volume_surge']
        
        # 4. Volume consistency
        volume_std = volumes.iloc[signal_idx-lookback:signal_idx].std()
        volume_cv = volume_std / avg_volume if avg_volume > 0 else 1
        components['volume_consistency'] = max(0, 1 - volume_cv)
        
        # 5. Price-volume correlation
        price_changes = prices.iloc[signal_idx-lookback:signal_idx].pct_change()
        volume_changes = volumes.iloc[signal_idx-lookback:signal_idx].pct_change()
        
        if len(price_changes.dropna()) > 2 and len(volume_changes.dropna()) > 2:
            correlation = price_changes.corr(volume_changes)
            components['price_volume_correlation'] = abs(correlation) if not pd.isna(correlation) else 0
        else:
            components['price_volume_correlation'] = 0
        
        # Calculate weighted strength
        total_strength = sum(components[key] * self.weights[key] for key in components)
        
        return {
            'strength': total_strength,
            'components': components,
            'signal_quality': self._classify_signal_quality(total_strength)
        }
    
    def _classify_signal_quality(self, strength: float) -> str:
        """Classify signal quality based on strength."""
        if strength >= 0.8:
            return 'excellent'
        elif strength >= 0.6:
            return 'good'
        elif strength >= 0.4:
            return 'moderate'
        elif strength >= 0.2:
            return 'weak'
        else:
            return 'very_weak'