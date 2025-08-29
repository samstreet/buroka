"""
Chart pattern recognition for technical analysis.
Implements candlestick patterns, breakout patterns, and classic chart formations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import signal, stats
import warnings

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PatternResult:
    """Result from pattern detection."""
    pattern_type: str
    pattern_name: str
    start_idx: int
    end_idx: int
    start_date: datetime
    end_date: datetime
    confidence: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternPerformance:
    """Track pattern performance."""
    pattern_name: str
    total_occurrences: int
    successful_predictions: int
    failed_predictions: int
    success_rate: float
    avg_return: float
    max_return: float
    min_return: float
    avg_duration_bars: float


class CandlestickPatterns:
    """
    Recognizes candlestick patterns in OHLC data.
    """
    
    def __init__(self, min_body_size: float = 0.001):
        """
        Initialize candlestick pattern detector.
        
        Args:
            min_body_size: Minimum body size as percentage of price
        """
        self.min_body_size = min_body_size
        self.patterns_detected = []
    
    def detect_all_patterns(self, 
                           open_prices: pd.Series,
                           high_prices: pd.Series,
                           low_prices: pd.Series,
                           close_prices: pd.Series) -> List[PatternResult]:
        """
        Detect all candlestick patterns.
        
        Args:
            open_prices: Open prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Close prices
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Single candle patterns
        patterns.extend(self._detect_doji(open_prices, high_prices, low_prices, close_prices))
        patterns.extend(self._detect_hammer(open_prices, high_prices, low_prices, close_prices))
        patterns.extend(self._detect_shooting_star(open_prices, high_prices, low_prices, close_prices))
        patterns.extend(self._detect_spinning_top(open_prices, high_prices, low_prices, close_prices))
        
        # Two candle patterns
        patterns.extend(self._detect_engulfing(open_prices, high_prices, low_prices, close_prices))
        patterns.extend(self._detect_harami(open_prices, high_prices, low_prices, close_prices))
        patterns.extend(self._detect_piercing_line(open_prices, high_prices, low_prices, close_prices))
        
        # Three candle patterns
        patterns.extend(self._detect_morning_star(open_prices, high_prices, low_prices, close_prices))
        patterns.extend(self._detect_evening_star(open_prices, high_prices, low_prices, close_prices))
        patterns.extend(self._detect_three_soldiers(open_prices, high_prices, low_prices, close_prices))
        
        self.patterns_detected = patterns
        return patterns
    
    def _detect_doji(self, open_p: pd.Series, high: pd.Series, 
                    low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect doji patterns."""
        patterns = []
        
        for i in range(len(close)):
            # Doji: open and close are very close
            body = abs(close.iloc[i] - open_p.iloc[i])
            avg_price = (open_p.iloc[i] + close.iloc[i]) / 2
            
            if body / avg_price < self.min_body_size:
                # Check for specific doji types
                upper_shadow = high.iloc[i] - max(open_p.iloc[i], close.iloc[i])
                lower_shadow = min(open_p.iloc[i], close.iloc[i]) - low.iloc[i]
                total_range = high.iloc[i] - low.iloc[i]
                
                doji_type = "standard"
                confidence = 0.7
                
                if total_range > 0:
                    if upper_shadow / total_range > 0.7:
                        doji_type = "gravestone"
                        confidence = 0.8
                    elif lower_shadow / total_range > 0.7:
                        doji_type = "dragonfly"
                        confidence = 0.8
                    elif abs(upper_shadow - lower_shadow) / total_range < 0.1:
                        doji_type = "long-legged"
                        confidence = 0.75
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name=f"{doji_type}_doji",
                    start_idx=i,
                    end_idx=i,
                    start_date=close.index[i],
                    end_date=close.index[i],
                    confidence=confidence,
                    direction="neutral",
                    entry_price=close.iloc[i],
                    target_price=None,
                    stop_loss=None,
                    metadata={
                        "body_size": body,
                        "upper_shadow": upper_shadow,
                        "lower_shadow": lower_shadow
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_hammer(self, open_p: pd.Series, high: pd.Series,
                      low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect hammer and inverted hammer patterns."""
        patterns = []
        
        for i in range(1, len(close)):
            body = abs(close.iloc[i] - open_p.iloc[i])
            upper_shadow = high.iloc[i] - max(open_p.iloc[i], close.iloc[i])
            lower_shadow = min(open_p.iloc[i], close.iloc[i]) - low.iloc[i]
            
            # Hammer: small body at top, long lower shadow
            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                # Check if in downtrend
                if i >= 5:
                    prev_trend = close.iloc[i-5:i].mean() > close.iloc[i]
                    if prev_trend:
                        pattern = PatternResult(
                            pattern_type="candlestick",
                            pattern_name="hammer",
                            start_idx=i,
                            end_idx=i,
                            start_date=close.index[i],
                            end_date=close.index[i],
                            confidence=0.7,
                            direction="bullish",
                            entry_price=close.iloc[i],
                            target_price=close.iloc[i] * 1.02,
                            stop_loss=low.iloc[i] * 0.99,
                            metadata={"lower_shadow_ratio": lower_shadow / body}
                        )
                        patterns.append(pattern)
            
            # Inverted hammer: small body at bottom, long upper shadow
            elif upper_shadow > body * 2 and lower_shadow < body * 0.5:
                if i >= 5:
                    prev_trend = close.iloc[i-5:i].mean() > close.iloc[i]
                    if prev_trend:
                        pattern = PatternResult(
                            pattern_type="candlestick",
                            pattern_name="inverted_hammer",
                            start_idx=i,
                            end_idx=i,
                            start_date=close.index[i],
                            end_date=close.index[i],
                            confidence=0.65,
                            direction="bullish",
                            entry_price=close.iloc[i],
                            target_price=close.iloc[i] * 1.02,
                            stop_loss=low.iloc[i] * 0.99,
                            metadata={"upper_shadow_ratio": upper_shadow / body}
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_shooting_star(self, open_p: pd.Series, high: pd.Series,
                             low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect shooting star pattern."""
        patterns = []
        
        for i in range(1, len(close)):
            body = abs(close.iloc[i] - open_p.iloc[i])
            upper_shadow = high.iloc[i] - max(open_p.iloc[i], close.iloc[i])
            lower_shadow = min(open_p.iloc[i], close.iloc[i]) - low.iloc[i]
            
            # Shooting star: small body at bottom, long upper shadow
            if upper_shadow > body * 2 and lower_shadow < body * 0.5:
                # Check if in uptrend
                if i >= 5:
                    prev_trend = close.iloc[i-5:i].mean() < close.iloc[i]
                    if prev_trend:
                        pattern = PatternResult(
                            pattern_type="candlestick",
                            pattern_name="shooting_star",
                            start_idx=i,
                            end_idx=i,
                            start_date=close.index[i],
                            end_date=close.index[i],
                            confidence=0.7,
                            direction="bearish",
                            entry_price=close.iloc[i],
                            target_price=close.iloc[i] * 0.98,
                            stop_loss=high.iloc[i] * 1.01,
                            metadata={"upper_shadow_ratio": upper_shadow / body if body > 0 else 0}
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_spinning_top(self, open_p: pd.Series, high: pd.Series,
                            low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect spinning top pattern."""
        patterns = []
        
        for i in range(len(close)):
            body = abs(close.iloc[i] - open_p.iloc[i])
            upper_shadow = high.iloc[i] - max(open_p.iloc[i], close.iloc[i])
            lower_shadow = min(open_p.iloc[i], close.iloc[i]) - low.iloc[i]
            total_range = high.iloc[i] - low.iloc[i]
            
            if total_range > 0:
                body_ratio = body / total_range
                
                # Spinning top: small body with similar shadows
                if body_ratio < 0.3 and abs(upper_shadow - lower_shadow) / total_range < 0.2:
                    pattern = PatternResult(
                        pattern_type="candlestick",
                        pattern_name="spinning_top",
                        start_idx=i,
                        end_idx=i,
                        start_date=close.index[i],
                        end_date=close.index[i],
                        confidence=0.6,
                        direction="neutral",
                        entry_price=close.iloc[i],
                        target_price=None,
                        stop_loss=None,
                        metadata={"body_ratio": body_ratio}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_engulfing(self, open_p: pd.Series, high: pd.Series,
                         low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect bullish and bearish engulfing patterns."""
        patterns = []
        
        for i in range(1, len(close)):
            prev_body = close.iloc[i-1] - open_p.iloc[i-1]
            curr_body = close.iloc[i] - open_p.iloc[i]
            
            # Bullish engulfing
            if (prev_body < 0 and curr_body > 0 and
                open_p.iloc[i] < close.iloc[i-1] and
                close.iloc[i] > open_p.iloc[i-1]):
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name="bullish_engulfing",
                    start_idx=i-1,
                    end_idx=i,
                    start_date=close.index[i-1],
                    end_date=close.index[i],
                    confidence=0.75,
                    direction="bullish",
                    entry_price=close.iloc[i],
                    target_price=close.iloc[i] * 1.03,
                    stop_loss=low.iloc[i] * 0.98,
                    metadata={"engulfing_ratio": abs(curr_body) / abs(prev_body) if prev_body != 0 else 0}
                )
                patterns.append(pattern)
            
            # Bearish engulfing
            elif (prev_body > 0 and curr_body < 0 and
                  open_p.iloc[i] > close.iloc[i-1] and
                  close.iloc[i] < open_p.iloc[i-1]):
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name="bearish_engulfing",
                    start_idx=i-1,
                    end_idx=i,
                    start_date=close.index[i-1],
                    end_date=close.index[i],
                    confidence=0.75,
                    direction="bearish",
                    entry_price=close.iloc[i],
                    target_price=close.iloc[i] * 0.97,
                    stop_loss=high.iloc[i] * 1.02,
                    metadata={"engulfing_ratio": abs(curr_body) / abs(prev_body) if prev_body != 0 else 0}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_harami(self, open_p: pd.Series, high: pd.Series,
                      low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect harami patterns."""
        patterns = []
        
        for i in range(1, len(close)):
            prev_body = abs(close.iloc[i-1] - open_p.iloc[i-1])
            curr_body = abs(close.iloc[i] - open_p.iloc[i])
            
            # Current candle body is within previous candle body
            if (curr_body < prev_body * 0.5 and
                min(open_p.iloc[i], close.iloc[i]) > min(open_p.iloc[i-1], close.iloc[i-1]) and
                max(open_p.iloc[i], close.iloc[i]) < max(open_p.iloc[i-1], close.iloc[i-1])):
                
                # Determine direction based on previous trend
                if close.iloc[i-1] > open_p.iloc[i-1]:
                    direction = "bearish"  # Bearish harami after uptrend
                else:
                    direction = "bullish"  # Bullish harami after downtrend
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name=f"{direction}_harami",
                    start_idx=i-1,
                    end_idx=i,
                    start_date=close.index[i-1],
                    end_date=close.index[i],
                    confidence=0.65,
                    direction=direction,
                    entry_price=close.iloc[i],
                    target_price=close.iloc[i] * (1.02 if direction == "bullish" else 0.98),
                    stop_loss=close.iloc[i] * (0.98 if direction == "bullish" else 1.02),
                    metadata={"body_ratio": curr_body / prev_body if prev_body > 0 else 0}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_piercing_line(self, open_p: pd.Series, high: pd.Series,
                             low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect piercing line and dark cloud cover patterns."""
        patterns = []
        
        for i in range(1, len(close)):
            prev_body = close.iloc[i-1] - open_p.iloc[i-1]
            curr_body = close.iloc[i] - open_p.iloc[i]
            
            # Piercing line (bullish)
            if (prev_body < 0 and curr_body > 0 and
                open_p.iloc[i] < low.iloc[i-1] and
                close.iloc[i] > (open_p.iloc[i-1] + close.iloc[i-1]) / 2 and
                close.iloc[i] < open_p.iloc[i-1]):
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name="piercing_line",
                    start_idx=i-1,
                    end_idx=i,
                    start_date=close.index[i-1],
                    end_date=close.index[i],
                    confidence=0.7,
                    direction="bullish",
                    entry_price=close.iloc[i],
                    target_price=close.iloc[i] * 1.03,
                    stop_loss=low.iloc[i] * 0.98,
                    metadata={"penetration": (close.iloc[i] - close.iloc[i-1]) / abs(prev_body) if prev_body != 0 else 0}
                )
                patterns.append(pattern)
            
            # Dark cloud cover (bearish)
            elif (prev_body > 0 and curr_body < 0 and
                  open_p.iloc[i] > high.iloc[i-1] and
                  close.iloc[i] < (open_p.iloc[i-1] + close.iloc[i-1]) / 2 and
                  close.iloc[i] > open_p.iloc[i-1]):
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name="dark_cloud_cover",
                    start_idx=i-1,
                    end_idx=i,
                    start_date=close.index[i-1],
                    end_date=close.index[i],
                    confidence=0.7,
                    direction="bearish",
                    entry_price=close.iloc[i],
                    target_price=close.iloc[i] * 0.97,
                    stop_loss=high.iloc[i] * 1.02,
                    metadata={"penetration": (open_p.iloc[i-1] - close.iloc[i]) / abs(prev_body) if prev_body != 0 else 0}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_morning_star(self, open_p: pd.Series, high: pd.Series,
                            low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect morning star pattern."""
        patterns = []
        
        for i in range(2, len(close)):
            # First candle: bearish
            first_body = close.iloc[i-2] - open_p.iloc[i-2]
            # Second candle: small body (star)
            second_body = abs(close.iloc[i-1] - open_p.iloc[i-1])
            # Third candle: bullish
            third_body = close.iloc[i] - open_p.iloc[i]
            
            if (first_body < 0 and 
                second_body < abs(first_body) * 0.3 and
                third_body > 0 and
                close.iloc[i] > (open_p.iloc[i-2] + close.iloc[i-2]) / 2):
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name="morning_star",
                    start_idx=i-2,
                    end_idx=i,
                    start_date=close.index[i-2],
                    end_date=close.index[i],
                    confidence=0.8,
                    direction="bullish",
                    entry_price=close.iloc[i],
                    target_price=close.iloc[i] * 1.04,
                    stop_loss=min(low.iloc[i-2:i+1]) * 0.98,
                    metadata={"star_size": second_body / abs(first_body) if first_body != 0 else 0}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_evening_star(self, open_p: pd.Series, high: pd.Series,
                            low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect evening star pattern."""
        patterns = []
        
        for i in range(2, len(close)):
            # First candle: bullish
            first_body = close.iloc[i-2] - open_p.iloc[i-2]
            # Second candle: small body (star)
            second_body = abs(close.iloc[i-1] - open_p.iloc[i-1])
            # Third candle: bearish
            third_body = close.iloc[i] - open_p.iloc[i]
            
            if (first_body > 0 and 
                second_body < abs(first_body) * 0.3 and
                third_body < 0 and
                close.iloc[i] < (open_p.iloc[i-2] + close.iloc[i-2]) / 2):
                
                pattern = PatternResult(
                    pattern_type="candlestick",
                    pattern_name="evening_star",
                    start_idx=i-2,
                    end_idx=i,
                    start_date=close.index[i-2],
                    end_date=close.index[i],
                    confidence=0.8,
                    direction="bearish",
                    entry_price=close.iloc[i],
                    target_price=close.iloc[i] * 0.96,
                    stop_loss=max(high.iloc[i-2:i+1]) * 1.02,
                    metadata={"star_size": second_body / abs(first_body) if first_body != 0 else 0}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_three_soldiers(self, open_p: pd.Series, high: pd.Series,
                              low: pd.Series, close: pd.Series) -> List[PatternResult]:
        """Detect three white soldiers and three black crows."""
        patterns = []
        
        for i in range(2, len(close)):
            bodies = [
                close.iloc[i-2] - open_p.iloc[i-2],
                close.iloc[i-1] - open_p.iloc[i-1],
                close.iloc[i] - open_p.iloc[i]
            ]
            
            # Three white soldiers (bullish)
            if all(b > 0 for b in bodies):
                if (close.iloc[i-1] > close.iloc[i-2] and
                    close.iloc[i] > close.iloc[i-1] and
                    open_p.iloc[i-1] > open_p.iloc[i-2] and
                    open_p.iloc[i] > open_p.iloc[i-1]):
                    
                    pattern = PatternResult(
                        pattern_type="candlestick",
                        pattern_name="three_white_soldiers",
                        start_idx=i-2,
                        end_idx=i,
                        start_date=close.index[i-2],
                        end_date=close.index[i],
                        confidence=0.85,
                        direction="bullish",
                        entry_price=close.iloc[i],
                        target_price=close.iloc[i] * 1.05,
                        stop_loss=min(low.iloc[i-2:i+1]) * 0.97,
                        metadata={"avg_body_size": np.mean([abs(b) for b in bodies])}
                    )
                    patterns.append(pattern)
            
            # Three black crows (bearish)
            elif all(b < 0 for b in bodies):
                if (close.iloc[i-1] < close.iloc[i-2] and
                    close.iloc[i] < close.iloc[i-1] and
                    open_p.iloc[i-1] < open_p.iloc[i-2] and
                    open_p.iloc[i] < open_p.iloc[i-1]):
                    
                    pattern = PatternResult(
                        pattern_type="candlestick",
                        pattern_name="three_black_crows",
                        start_idx=i-2,
                        end_idx=i,
                        start_date=close.index[i-2],
                        end_date=close.index[i],
                        confidence=0.85,
                        direction="bearish",
                        entry_price=close.iloc[i],
                        target_price=close.iloc[i] * 0.95,
                        stop_loss=max(high.iloc[i-2:i+1]) * 1.03,
                        metadata={"avg_body_size": np.mean([abs(b) for b in bodies])}
                    )
                    patterns.append(pattern)
        
        return patterns


class BreakoutPatterns:
    """
    Detects breakout patterns like triangles, rectangles, and wedges.
    """
    
    def __init__(self, min_touches: int = 2, breakout_threshold: float = 0.02):
        """
        Initialize breakout pattern detector.
        
        Args:
            min_touches: Minimum touches for pattern validation
            breakout_threshold: Percentage move for breakout confirmation
        """
        self.min_touches = min_touches
        self.breakout_threshold = breakout_threshold
    
    def detect_all_patterns(self,
                           high: pd.Series,
                           low: pd.Series,
                           close: pd.Series,
                           volume: Optional[pd.Series] = None) -> List[PatternResult]:
        """
        Detect all breakout patterns.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Optional volume data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Triangle patterns
        patterns.extend(self._detect_ascending_triangle(high, low, close, volume))
        patterns.extend(self._detect_descending_triangle(high, low, close, volume))
        patterns.extend(self._detect_symmetrical_triangle(high, low, close, volume))
        
        # Rectangle patterns
        patterns.extend(self._detect_rectangle(high, low, close, volume))
        
        # Wedge patterns
        patterns.extend(self._detect_wedge(high, low, close, volume))
        
        return patterns
    
    def _detect_ascending_triangle(self, high: pd.Series, low: pd.Series,
                                  close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect ascending triangle pattern."""
        patterns = []
        window = 20  # Look for patterns in 20-bar windows
        
        for i in range(window, len(close)):
            window_high = high.iloc[i-window:i]
            window_low = low.iloc[i-window:i]
            window_close = close.iloc[i-window:i]
            
            # Find peaks and troughs
            peaks_idx = signal.argrelextrema(window_high.values, np.greater, order=2)[0]
            troughs_idx = signal.argrelextrema(window_low.values, np.less, order=2)[0]
            
            if len(peaks_idx) >= self.min_touches and len(troughs_idx) >= self.min_touches:
                # Check for horizontal resistance (peaks at similar levels)
                peak_values = window_high.iloc[peaks_idx].values
                peak_std = np.std(peak_values)
                peak_mean = np.mean(peak_values)
                
                # Check for ascending support (troughs trending up)
                trough_values = window_low.iloc[troughs_idx].values
                trough_slope = np.polyfit(troughs_idx, trough_values, 1)[0]
                
                # Validate ascending triangle
                if peak_std / peak_mean < 0.02 and trough_slope > 0:
                    # Check for breakout
                    if close.iloc[i] > peak_mean * (1 + self.breakout_threshold):
                        pattern = PatternResult(
                            pattern_type="breakout",
                            pattern_name="ascending_triangle",
                            start_idx=i-window,
                            end_idx=i,
                            start_date=close.index[i-window],
                            end_date=close.index[i],
                            confidence=0.75,
                            direction="bullish",
                            entry_price=close.iloc[i],
                            target_price=close.iloc[i] + (peak_mean - min(trough_values)),
                            stop_loss=peak_mean * 0.98,
                            metadata={
                                "resistance_level": peak_mean,
                                "support_slope": trough_slope,
                                "breakout_volume": volume.iloc[i] if volume is not None else None
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_descending_triangle(self, high: pd.Series, low: pd.Series,
                                   close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect descending triangle pattern."""
        patterns = []
        window = 20
        
        for i in range(window, len(close)):
            window_high = high.iloc[i-window:i]
            window_low = low.iloc[i-window:i]
            window_close = close.iloc[i-window:i]
            
            peaks_idx = signal.argrelextrema(window_high.values, np.greater, order=2)[0]
            troughs_idx = signal.argrelextrema(window_low.values, np.less, order=2)[0]
            
            if len(peaks_idx) >= self.min_touches and len(troughs_idx) >= self.min_touches:
                # Check for horizontal support
                trough_values = window_low.iloc[troughs_idx].values
                trough_std = np.std(trough_values)
                trough_mean = np.mean(trough_values)
                
                # Check for descending resistance
                peak_values = window_high.iloc[peaks_idx].values
                peak_slope = np.polyfit(peaks_idx, peak_values, 1)[0]
                
                # Validate descending triangle
                if trough_std / trough_mean < 0.02 and peak_slope < 0:
                    # Check for breakout
                    if close.iloc[i] < trough_mean * (1 - self.breakout_threshold):
                        pattern = PatternResult(
                            pattern_type="breakout",
                            pattern_name="descending_triangle",
                            start_idx=i-window,
                            end_idx=i,
                            start_date=close.index[i-window],
                            end_date=close.index[i],
                            confidence=0.75,
                            direction="bearish",
                            entry_price=close.iloc[i],
                            target_price=close.iloc[i] - (max(peak_values) - trough_mean),
                            stop_loss=trough_mean * 1.02,
                            metadata={
                                "support_level": trough_mean,
                                "resistance_slope": peak_slope,
                                "breakout_volume": volume.iloc[i] if volume is not None else None
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_symmetrical_triangle(self, high: pd.Series, low: pd.Series,
                                    close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect symmetrical triangle pattern."""
        patterns = []
        window = 20
        
        for i in range(window, len(close)):
            window_high = high.iloc[i-window:i]
            window_low = low.iloc[i-window:i]
            window_close = close.iloc[i-window:i]
            
            peaks_idx = signal.argrelextrema(window_high.values, np.greater, order=2)[0]
            troughs_idx = signal.argrelextrema(window_low.values, np.less, order=2)[0]
            
            if len(peaks_idx) >= self.min_touches and len(troughs_idx) >= self.min_touches:
                # Check for converging trendlines
                peak_values = window_high.iloc[peaks_idx].values
                trough_values = window_low.iloc[troughs_idx].values
                
                peak_slope = np.polyfit(peaks_idx, peak_values, 1)[0]
                trough_slope = np.polyfit(troughs_idx, trough_values, 1)[0]
                
                # Validate symmetrical triangle (converging lines)
                if peak_slope < 0 and trough_slope > 0:
                    # Calculate apex
                    current_resistance = peak_values[-1] + peak_slope * (i - peaks_idx[-1])
                    current_support = trough_values[-1] + trough_slope * (i - troughs_idx[-1])
                    
                    # Check for breakout
                    if close.iloc[i] > current_resistance * (1 + self.breakout_threshold):
                        direction = "bullish"
                        target = close.iloc[i] + (peak_values[0] - trough_values[0])
                    elif close.iloc[i] < current_support * (1 - self.breakout_threshold):
                        direction = "bearish"
                        target = close.iloc[i] - (peak_values[0] - trough_values[0])
                    else:
                        continue
                    
                    pattern = PatternResult(
                        pattern_type="breakout",
                        pattern_name="symmetrical_triangle",
                        start_idx=i-window,
                        end_idx=i,
                        start_date=close.index[i-window],
                        end_date=close.index[i],
                        confidence=0.7,
                        direction=direction,
                        entry_price=close.iloc[i],
                        target_price=target,
                        stop_loss=close.iloc[i] * (0.98 if direction == "bullish" else 1.02),
                        metadata={
                            "resistance_slope": peak_slope,
                            "support_slope": trough_slope,
                            "breakout_volume": volume.iloc[i] if volume is not None else None
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_rectangle(self, high: pd.Series, low: pd.Series,
                         close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect rectangle (consolidation) pattern."""
        patterns = []
        window = 20
        
        for i in range(window, len(close)):
            window_high = high.iloc[i-window:i]
            window_low = low.iloc[i-window:i]
            window_close = close.iloc[i-window:i]
            
            # Calculate support and resistance levels
            resistance = window_high.max()
            support = window_low.min()
            range_size = resistance - support
            
            # Count touches
            resistance_touches = sum(abs(window_high - resistance) / resistance < 0.01)
            support_touches = sum(abs(window_low - support) / support < 0.01)
            
            if resistance_touches >= self.min_touches and support_touches >= self.min_touches:
                # Check if price is consolidating
                price_std = window_close.std()
                price_mean = window_close.mean()
                
                if price_std / price_mean < 0.05:  # Low volatility indicates consolidation
                    # Check for breakout
                    if close.iloc[i] > resistance * (1 + self.breakout_threshold):
                        direction = "bullish"
                        target = close.iloc[i] + range_size
                    elif close.iloc[i] < support * (1 - self.breakout_threshold):
                        direction = "bearish"
                        target = close.iloc[i] - range_size
                    else:
                        continue
                    
                    pattern = PatternResult(
                        pattern_type="breakout",
                        pattern_name="rectangle",
                        start_idx=i-window,
                        end_idx=i,
                        start_date=close.index[i-window],
                        end_date=close.index[i],
                        confidence=0.8,
                        direction=direction,
                        entry_price=close.iloc[i],
                        target_price=target,
                        stop_loss=resistance if direction == "bearish" else support,
                        metadata={
                            "resistance": resistance,
                            "support": support,
                            "range_size": range_size,
                            "consolidation_period": window
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_wedge(self, high: pd.Series, low: pd.Series,
                     close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect rising and falling wedge patterns."""
        patterns = []
        window = 20
        
        for i in range(window, len(close)):
            window_high = high.iloc[i-window:i]
            window_low = low.iloc[i-window:i]
            
            # Get trendlines
            high_slope = np.polyfit(range(window), window_high.values, 1)[0]
            low_slope = np.polyfit(range(window), window_low.values, 1)[0]
            
            # Rising wedge (bearish)
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                if close.iloc[i] < window_low.iloc[-1]:
                    pattern = PatternResult(
                        pattern_type="breakout",
                        pattern_name="rising_wedge",
                        start_idx=i-window,
                        end_idx=i,
                        start_date=close.index[i-window],
                        end_date=close.index[i],
                        confidence=0.7,
                        direction="bearish",
                        entry_price=close.iloc[i],
                        target_price=close.iloc[i] * 0.95,
                        stop_loss=window_high.iloc[-1],
                        metadata={
                            "upper_slope": high_slope,
                            "lower_slope": low_slope
                        }
                    )
                    patterns.append(pattern)
            
            # Falling wedge (bullish)
            elif high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope):
                if close.iloc[i] > window_high.iloc[-1]:
                    pattern = PatternResult(
                        pattern_type="breakout",
                        pattern_name="falling_wedge",
                        start_idx=i-window,
                        end_idx=i,
                        start_date=close.index[i-window],
                        end_date=close.index[i],
                        confidence=0.7,
                        direction="bullish",
                        entry_price=close.iloc[i],
                        target_price=close.iloc[i] * 1.05,
                        stop_loss=window_low.iloc[-1],
                        metadata={
                            "upper_slope": high_slope,
                            "lower_slope": low_slope
                        }
                    )
                    patterns.append(pattern)
        
        return patterns


class ClassicPatterns:
    """
    Detects classic chart patterns like head and shoulders, flags, and pennants.
    """
    
    def __init__(self, min_pattern_bars: int = 10):
        """
        Initialize classic pattern detector.
        
        Args:
            min_pattern_bars: Minimum bars for pattern formation
        """
        self.min_pattern_bars = min_pattern_bars
    
    def detect_all_patterns(self,
                           high: pd.Series,
                           low: pd.Series,
                           close: pd.Series,
                           volume: Optional[pd.Series] = None) -> List[PatternResult]:
        """Detect all classic patterns."""
        patterns = []
        
        patterns.extend(self._detect_head_and_shoulders(high, low, close, volume))
        patterns.extend(self._detect_flag(high, low, close, volume))
        patterns.extend(self._detect_pennant(high, low, close, volume))
        
        return patterns
    
    def _detect_head_and_shoulders(self, high: pd.Series, low: pd.Series,
                                  close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect head and shoulders patterns."""
        patterns = []
        window = 30  # Need larger window for H&S
        
        for i in range(window, len(close) - 5):
            window_high = high.iloc[i-window:i]
            window_low = low.iloc[i-window:i]
            
            # Find peaks (potential shoulders and head)
            peaks_idx = signal.argrelextrema(window_high.values, np.greater, order=3)[0]
            
            if len(peaks_idx) >= 3:
                # Look for three-peak pattern
                for j in range(len(peaks_idx) - 2):
                    left_shoulder_idx = peaks_idx[j]
                    head_idx = peaks_idx[j + 1]
                    right_shoulder_idx = peaks_idx[j + 2]
                    
                    left_shoulder = window_high.iloc[left_shoulder_idx]
                    head = window_high.iloc[head_idx]
                    right_shoulder = window_high.iloc[right_shoulder_idx]
                    
                    # Validate H&S pattern
                    if (head > left_shoulder and head > right_shoulder and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                        
                        # Find neckline (troughs between shoulders and head)
                        left_trough = window_low.iloc[left_shoulder_idx:head_idx].min()
                        right_trough = window_low.iloc[head_idx:right_shoulder_idx].min()
                        neckline = (left_trough + right_trough) / 2
                        
                        # Check for neckline break
                        if close.iloc[i] < neckline:
                            pattern = PatternResult(
                                pattern_type="classic",
                                pattern_name="head_and_shoulders",
                                start_idx=i-window+left_shoulder_idx,
                                end_idx=i,
                                start_date=close.index[i-window+left_shoulder_idx],
                                end_date=close.index[i],
                                confidence=0.85,
                                direction="bearish",
                                entry_price=close.iloc[i],
                                target_price=neckline - (head - neckline),
                                stop_loss=right_shoulder * 1.02,
                                metadata={
                                    "left_shoulder": left_shoulder,
                                    "head": head,
                                    "right_shoulder": right_shoulder,
                                    "neckline": neckline
                                }
                            )
                            patterns.append(pattern)
                            break
        
        return patterns
    
    def _detect_flag(self, high: pd.Series, low: pd.Series,
                    close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect flag patterns."""
        patterns = []
        
        for i in range(20, len(close) - 10):
            # Look for strong move (pole)
            pole_start = i - 20
            pole_end = i - 10
            pole_move = close.iloc[pole_end] - close.iloc[pole_start]
            
            if abs(pole_move) / close.iloc[pole_start] > 0.05:  # 5% move
                # Look for consolidation (flag)
                flag_high = high.iloc[pole_end:i]
                flag_low = low.iloc[pole_end:i]
                flag_close = close.iloc[pole_end:i]
                
                # Calculate flag characteristics
                flag_slope = np.polyfit(range(len(flag_close)), flag_close.values, 1)[0]
                flag_volatility = flag_close.std() / flag_close.mean()
                
                # Validate flag pattern
                if flag_volatility < 0.02:  # Low volatility in flag
                    if pole_move > 0 and flag_slope < 0:
                        # Bullish flag
                        if close.iloc[i] > flag_high.max():
                            pattern = PatternResult(
                                pattern_type="classic",
                                pattern_name="bull_flag",
                                start_idx=pole_start,
                                end_idx=i,
                                start_date=close.index[pole_start],
                                end_date=close.index[i],
                                confidence=0.75,
                                direction="bullish",
                                entry_price=close.iloc[i],
                                target_price=close.iloc[i] + abs(pole_move),
                                stop_loss=flag_low.min(),
                                metadata={
                                    "pole_height": abs(pole_move),
                                    "flag_slope": flag_slope
                                }
                            )
                            patterns.append(pattern)
                    
                    elif pole_move < 0 and flag_slope > 0:
                        # Bearish flag
                        if close.iloc[i] < flag_low.min():
                            pattern = PatternResult(
                                pattern_type="classic",
                                pattern_name="bear_flag",
                                start_idx=pole_start,
                                end_idx=i,
                                start_date=close.index[pole_start],
                                end_date=close.index[i],
                                confidence=0.75,
                                direction="bearish",
                                entry_price=close.iloc[i],
                                target_price=close.iloc[i] - abs(pole_move),
                                stop_loss=flag_high.max(),
                                metadata={
                                    "pole_height": abs(pole_move),
                                    "flag_slope": flag_slope
                                }
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_pennant(self, high: pd.Series, low: pd.Series,
                       close: pd.Series, volume: Optional[pd.Series]) -> List[PatternResult]:
        """Detect pennant patterns."""
        patterns = []
        
        for i in range(20, len(close) - 10):
            # Look for strong move (pole)
            pole_start = i - 20
            pole_end = i - 10
            pole_move = close.iloc[pole_end] - close.iloc[pole_start]
            
            if abs(pole_move) / close.iloc[pole_start] > 0.05:
                # Look for converging consolidation (pennant)
                pennant_high = high.iloc[pole_end:i]
                pennant_low = low.iloc[pole_end:i]
                
                # Calculate converging trendlines
                high_slope = np.polyfit(range(len(pennant_high)), pennant_high.values, 1)[0]
                low_slope = np.polyfit(range(len(pennant_low)), pennant_low.values, 1)[0]
                
                # Validate pennant (converging lines)
                if high_slope < 0 and low_slope > 0:
                    if pole_move > 0 and close.iloc[i] > pennant_high.max():
                        # Bullish pennant
                        pattern = PatternResult(
                            pattern_type="classic",
                            pattern_name="bull_pennant",
                            start_idx=pole_start,
                            end_idx=i,
                            start_date=close.index[pole_start],
                            end_date=close.index[i],
                            confidence=0.7,
                            direction="bullish",
                            entry_price=close.iloc[i],
                            target_price=close.iloc[i] + abs(pole_move),
                            stop_loss=pennant_low.min(),
                            metadata={
                                "pole_height": abs(pole_move),
                                "upper_slope": high_slope,
                                "lower_slope": low_slope
                            }
                        )
                        patterns.append(pattern)
                    
                    elif pole_move < 0 and close.iloc[i] < pennant_low.min():
                        # Bearish pennant
                        pattern = PatternResult(
                            pattern_type="classic",
                            pattern_name="bear_pennant",
                            start_idx=pole_start,
                            end_idx=i,
                            start_date=close.index[pole_start],
                            end_date=close.index[i],
                            confidence=0.7,
                            direction="bearish",
                            entry_price=close.iloc[i],
                            target_price=close.iloc[i] - abs(pole_move),
                            stop_loss=pennant_high.max(),
                            metadata={
                                "pole_height": abs(pole_move),
                                "upper_slope": high_slope,
                                "lower_slope": low_slope
                            }
                        )
                        patterns.append(pattern)
        
        return patterns


class GapAnalysis:
    """
    Analyzes price gaps in market data.
    """
    
    def __init__(self, min_gap_size: float = 0.002):
        """
        Initialize gap analyzer.
        
        Args:
            min_gap_size: Minimum gap size as percentage of price
        """
        self.min_gap_size = min_gap_size
    
    def detect_gaps(self,
                   open_p: pd.Series,
                   high: pd.Series,
                   low: pd.Series,
                   close: pd.Series) -> List[PatternResult]:
        """Detect and classify gaps."""
        gaps = []
        
        for i in range(1, len(close)):
            gap_size = open_p.iloc[i] - close.iloc[i-1]
            gap_pct = abs(gap_size) / close.iloc[i-1]
            
            if gap_pct > self.min_gap_size:
                # Classify gap type
                gap_type, direction = self._classify_gap(
                    i, gap_size, open_p, high, low, close
                )
                
                # Determine if gap was filled
                filled = self._check_gap_filled(i, gap_size, high, low, close)
                
                pattern = PatternResult(
                    pattern_type="gap",
                    pattern_name=gap_type,
                    start_idx=i-1,
                    end_idx=i,
                    start_date=close.index[i-1],
                    end_date=close.index[i],
                    confidence=0.9,  # Gaps are objective
                    direction=direction,
                    entry_price=open_p.iloc[i],
                    target_price=self._calculate_gap_target(gap_type, gap_size, close.iloc[i]),
                    stop_loss=close.iloc[i-1] if gap_size > 0 else open_p.iloc[i] * 1.02,
                    metadata={
                        "gap_size": gap_size,
                        "gap_percentage": gap_pct * 100,
                        "filled": filled,
                        "gap_direction": "up" if gap_size > 0 else "down"
                    }
                )
                gaps.append(pattern)
        
        return gaps
    
    def _classify_gap(self, idx: int, gap_size: float,
                     open_p: pd.Series, high: pd.Series,
                     low: pd.Series, close: pd.Series) -> Tuple[str, str]:
        """Classify gap type."""
        if idx < 5 or idx >= len(close) - 5:
            return "common_gap", "neutral"
        
        # Look at context
        prev_trend = close.iloc[idx-5:idx].mean()
        curr_price = close.iloc[idx]
        
        if gap_size > 0:  # Gap up
            if curr_price > prev_trend * 1.02:
                # Check if it's a breakaway gap (at the start of a move)
                if idx < 10 or close.iloc[idx-10:idx].std() / close.iloc[idx-10:idx].mean() < 0.02:
                    return "breakaway_gap", "bullish"
                # Check if it's a runaway gap (in the middle of a move)
                elif close.iloc[idx-5:idx].mean() > close.iloc[idx-10:idx-5].mean():
                    return "runaway_gap", "bullish"
                # Could be exhaustion gap (at the end of a move)
                elif idx < len(close) - 5 and close.iloc[idx:idx+5].mean() < curr_price:
                    return "exhaustion_gap", "bearish"
            return "common_gap", "neutral"
        
        else:  # Gap down
            if curr_price < prev_trend * 0.98:
                if idx < 10 or close.iloc[idx-10:idx].std() / close.iloc[idx-10:idx].mean() < 0.02:
                    return "breakaway_gap", "bearish"
                elif close.iloc[idx-5:idx].mean() < close.iloc[idx-10:idx-5].mean():
                    return "runaway_gap", "bearish"
                elif idx < len(close) - 5 and close.iloc[idx:idx+5].mean() > curr_price:
                    return "exhaustion_gap", "bullish"
            return "common_gap", "neutral"
    
    def _check_gap_filled(self, idx: int, gap_size: float,
                         high: pd.Series, low: pd.Series, close: pd.Series) -> bool:
        """Check if gap was filled in subsequent bars."""
        if idx >= len(close) - 10:
            return False
        
        gap_level = close.iloc[idx-1]
        
        for i in range(idx, min(idx + 10, len(close))):
            if gap_size > 0:  # Gap up
                if low.iloc[i] <= gap_level:
                    return True
            else:  # Gap down
                if high.iloc[i] >= gap_level:
                    return True
        
        return False
    
    def _calculate_gap_target(self, gap_type: str, gap_size: float, current_price: float) -> float:
        """Calculate price target based on gap type."""
        if gap_type == "breakaway_gap":
            return current_price + gap_size * 2  # Expect continuation
        elif gap_type == "runaway_gap":
            return current_price + gap_size  # Expect further movement
        elif gap_type == "exhaustion_gap":
            return current_price - gap_size * 0.5  # Expect reversal
        else:  # Common gap
            return current_price  # Expect gap fill


class PatternSuccessTracker:
    """
    Tracks pattern success rates for validation and improvement.
    """
    
    def __init__(self):
        self.pattern_history: List[Dict[str, Any]] = []
        self.performance_stats: Dict[str, PatternPerformance] = {}
    
    def track_pattern(self, pattern: PatternResult, 
                     actual_outcome: Dict[str, Any]) -> None:
        """
        Track a pattern's actual outcome.
        
        Args:
            pattern: The detected pattern
            actual_outcome: Dictionary with actual price movement and success
        """
        record = {
            'pattern': pattern,
            'outcome': actual_outcome,
            'timestamp': datetime.now()
        }
        self.pattern_history.append(record)
        
        # Update performance statistics
        self._update_performance_stats(pattern.pattern_name, actual_outcome)
    
    def _update_performance_stats(self, pattern_name: str, outcome: Dict[str, Any]) -> None:
        """Update performance statistics for a pattern."""
        if pattern_name not in self.performance_stats:
            self.performance_stats[pattern_name] = PatternPerformance(
                pattern_name=pattern_name,
                total_occurrences=0,
                successful_predictions=0,
                failed_predictions=0,
                success_rate=0,
                avg_return=0,
                max_return=float('-inf'),
                min_return=float('inf'),
                avg_duration_bars=0
            )
        
        stats = self.performance_stats[pattern_name]
        stats.total_occurrences += 1
        
        if outcome.get('success', False):
            stats.successful_predictions += 1
        else:
            stats.failed_predictions += 1
        
        # Update success rate
        stats.success_rate = stats.successful_predictions / stats.total_occurrences
        
        # Update return statistics
        return_pct = outcome.get('return_percentage', 0)
        stats.avg_return = ((stats.avg_return * (stats.total_occurrences - 1) + return_pct) / 
                           stats.total_occurrences)
        stats.max_return = max(stats.max_return, return_pct)
        stats.min_return = min(stats.min_return, return_pct)
        
        # Update duration
        duration = outcome.get('duration_bars', 0)
        stats.avg_duration_bars = ((stats.avg_duration_bars * (stats.total_occurrences - 1) + duration) / 
                                  stats.total_occurrences)
    
    def get_pattern_performance(self, pattern_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for patterns."""
        if pattern_name:
            return self.performance_stats.get(pattern_name, None)
        
        # Return all statistics
        return {
            'patterns': self.performance_stats,
            'summary': {
                'total_patterns_tracked': len(self.pattern_history),
                'unique_patterns': len(self.performance_stats),
                'overall_success_rate': self._calculate_overall_success_rate(),
                'best_performing': self._get_best_pattern(),
                'worst_performing': self._get_worst_pattern()
            }
        }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all patterns."""
        if not self.performance_stats:
            return 0
        
        total_success = sum(s.successful_predictions for s in self.performance_stats.values())
        total_patterns = sum(s.total_occurrences for s in self.performance_stats.values())
        
        return total_success / total_patterns if total_patterns > 0 else 0
    
    def _get_best_pattern(self) -> Optional[str]:
        """Get the best performing pattern."""
        if not self.performance_stats:
            return None
        
        return max(self.performance_stats.keys(), 
                  key=lambda k: self.performance_stats[k].success_rate)
    
    def _get_worst_pattern(self) -> Optional[str]:
        """Get the worst performing pattern."""
        if not self.performance_stats:
            return None
        
        return min(self.performance_stats.keys(), 
                  key=lambda k: self.performance_stats[k].success_rate)