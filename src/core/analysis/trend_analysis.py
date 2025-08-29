"""
Trend analysis algorithms for market data.
Implements HP filter, Mann-Kendall test, trend reversal detection, and support/resistance identification.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats, signal
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import warnings

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrendResult:
    """Result from trend analysis."""
    trend_type: str  # 'uptrend', 'downtrend', 'sideways'
    strength: float  # 0-1 scale
    confidence: float  # Statistical confidence
    start_date: datetime
    end_date: datetime
    slope: float
    intercept: float
    r_squared: float
    metadata: Dict[str, Any]


@dataclass
class SupportResistanceLevel:
    """Support or resistance level."""
    level: float
    level_type: str  # 'support' or 'resistance'
    strength: int  # Number of touches
    first_touch: datetime
    last_touch: datetime
    breaks: int  # Number of times broken
    confidence: float


class HodrickPrescottFilter:
    """
    Hodrick-Prescott filter for trend-cycle decomposition.
    Separates a time series into trend and cyclical components.
    """
    
    def __init__(self, lambda_param: float = 1600):
        """
        Initialize HP filter.
        
        Args:
            lambda_param: Smoothing parameter (1600 for quarterly, 6.25 for yearly, 129600 for monthly)
        """
        self.lambda_param = lambda_param
    
    def decompose(self, series: pd.Series) -> Dict[str, pd.Series]:
        """
        Decompose series into trend and cycle components.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with 'trend' and 'cycle' components
        """
        if len(series) < 4:
            raise ValueError("Need at least 4 data points for HP filter")
        
        # Remove NaN values
        clean_series = series.dropna()
        n = len(clean_series)
        
        # Create second difference matrix
        D = self._create_second_diff_matrix(n)
        
        # Solve (I + λD'D)τ = y for trend τ
        I = np.eye(n)
        H = I + self.lambda_param * (D.T @ D)
        
        # Use sparse matrix for efficiency
        H_sparse = csr_matrix(H)
        trend_values = spsolve(H_sparse, clean_series.values)
        
        # Create trend series
        trend = pd.Series(trend_values, index=clean_series.index)
        
        # Calculate cycle component
        cycle = clean_series - trend
        
        # Calculate noise component (high-frequency variations)
        noise = self._estimate_noise(cycle)
        
        return {
            'trend': trend,
            'cycle': cycle,
            'noise': noise,
            'original': clean_series,
            'trend_strength': self._calculate_trend_strength(trend, clean_series)
        }
    
    def _create_second_diff_matrix(self, n: int) -> np.ndarray:
        """Create second difference matrix for HP filter."""
        D = np.zeros((n-2, n))
        for i in range(n-2):
            D[i, i:i+3] = [1, -2, 1]
        return D
    
    def _estimate_noise(self, cycle: pd.Series) -> pd.Series:
        """Estimate noise component from cycle."""
        # Use rolling standard deviation as noise estimate
        window = min(10, len(cycle) // 4)
        noise = cycle.rolling(window=window, center=True).std()
        noise = noise.fillna(method='bfill').fillna(method='ffill')
        return noise
    
    def _calculate_trend_strength(self, trend: pd.Series, original: pd.Series) -> float:
        """Calculate how much of the variation is explained by the trend."""
        # R-squared of trend vs original
        ss_res = np.sum((original - trend) ** 2)
        ss_tot = np.sum((original - original.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return max(0, min(1, r_squared))


class MannKendallTest:
    """
    Mann-Kendall trend test for statistical validation.
    Non-parametric test for monotonic trends.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize Mann-Kendall test.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def test(self, series: pd.Series) -> Dict[str, Any]:
        """
        Perform Mann-Kendall trend test.
        
        Args:
            series: Time series data
            
        Returns:
            Test results including trend direction and significance
        """
        if len(series) < 4:
            raise ValueError("Need at least 4 data points for Mann-Kendall test")
        
        # Remove NaN values
        clean_series = series.dropna().values
        n = len(clean_series)
        
        # Calculate S statistic
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(clean_series[j] - clean_series[i])
        
        # Calculate variance
        var_s = self._calculate_variance(clean_series, n)
        
        # Calculate z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend
        if p_value < self.alpha:
            if z > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no_trend'
        
        # Calculate Sen's slope (Theil-Sen estimator)
        sen_slope = self._calculate_sen_slope(clean_series)
        
        return {
            'trend': trend,
            'p_value': p_value,
            'z_score': z,
            's_statistic': s,
            'sen_slope': sen_slope,
            'significant': p_value < self.alpha,
            'confidence': 1 - p_value if p_value < self.alpha else 0,
            'tau': s / (n * (n - 1) / 2)  # Kendall's tau
        }
    
    def _calculate_variance(self, data: np.ndarray, n: int) -> float:
        """Calculate variance for Mann-Kendall test."""
        # Check for ties
        unique, counts = np.unique(data, return_counts=True)
        ties = counts[counts > 1]
        
        # Base variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Adjust for ties
        if len(ties) > 0:
            tie_adjustment = np.sum(ties * (ties - 1) * (2 * ties + 5)) / 18
            var_s -= tie_adjustment
        
        return var_s
    
    def _calculate_sen_slope(self, data: np.ndarray) -> float:
        """Calculate Sen's slope estimator (median of all slopes)."""
        n = len(data)
        slopes = []
        
        for i in range(n-1):
            for j in range(i+1, n):
                if j - i > 0:
                    slope = (data[j] - data[i]) / (j - i)
                    slopes.append(slope)
        
        return np.median(slopes) if slopes else 0


class TrendReversalDetector:
    """
    Detects trend reversals using multiple confirmation methods.
    """
    
    def __init__(self, lookback: int = 20, confirmation_bars: int = 3):
        """
        Initialize trend reversal detector.
        
        Args:
            lookback: Period for trend calculation
            confirmation_bars: Bars needed to confirm reversal
        """
        self.lookback = lookback
        self.confirmation_bars = confirmation_bars
    
    def detect_reversals(self, prices: pd.Series) -> List[Dict[str, Any]]:
        """
        Detect trend reversals in price series.
        
        Args:
            prices: Price series
            
        Returns:
            List of reversal points with details
        """
        if len(prices) < self.lookback * 2:
            return []
        
        reversals = []
        
        # Calculate indicators for reversal detection
        sma_short = prices.rolling(window=self.lookback).mean()
        sma_long = prices.rolling(window=self.lookback * 2).mean()
        
        # Find local extrema
        peaks = signal.argrelextrema(prices.values, np.greater, order=self.lookback//2)[0]
        troughs = signal.argrelextrema(prices.values, np.less, order=self.lookback//2)[0]
        
        # Detect reversals at peaks (potential top)
        for peak_idx in peaks:
            if peak_idx < len(prices) - self.confirmation_bars:
                reversal = self._check_top_reversal(
                    prices, peak_idx, sma_short, sma_long
                )
                if reversal:
                    reversals.append(reversal)
        
        # Detect reversals at troughs (potential bottom)
        for trough_idx in troughs:
            if trough_idx < len(prices) - self.confirmation_bars:
                reversal = self._check_bottom_reversal(
                    prices, trough_idx, sma_short, sma_long
                )
                if reversal:
                    reversals.append(reversal)
        
        # Add pattern-based reversals
        pattern_reversals = self._detect_pattern_reversals(prices)
        reversals.extend(pattern_reversals)
        
        # Sort by date and remove duplicates
        reversals = self._remove_duplicate_reversals(reversals)
        
        return reversals
    
    def _check_top_reversal(self, prices: pd.Series, idx: int, 
                           sma_short: pd.Series, sma_long: pd.Series) -> Optional[Dict]:
        """Check for top reversal pattern."""
        # Confirm reversal conditions
        conditions = []
        
        # Price breaks below short SMA
        if idx + self.confirmation_bars < len(prices):
            price_below_sma = all(
                prices.iloc[idx + i] < sma_short.iloc[idx + i]
                for i in range(1, self.confirmation_bars + 1)
                if not pd.isna(sma_short.iloc[idx + i])
            )
            conditions.append(price_below_sma)
        
        # Momentum shift (using rate of change)
        if idx >= 10:
            momentum_before = (prices.iloc[idx] - prices.iloc[idx-10]) / prices.iloc[idx-10]
            momentum_after = (prices.iloc[min(idx+10, len(prices)-1)] - prices.iloc[idx]) / prices.iloc[idx]
            momentum_shift = momentum_before > 0 and momentum_after < 0
            conditions.append(momentum_shift)
        
        # Volume confirmation would go here if we had volume data
        
        if sum(conditions) >= 1:  # At least one condition met
            return {
                'type': 'top_reversal',
                'index': idx,
                'date': prices.index[idx],
                'price': prices.iloc[idx],
                'confidence': sum(conditions) / 2,  # Normalize to 0-1
                'confirmed': True,
                'pattern': 'peak_reversal'
            }
        
        return None
    
    def _check_bottom_reversal(self, prices: pd.Series, idx: int,
                              sma_short: pd.Series, sma_long: pd.Series) -> Optional[Dict]:
        """Check for bottom reversal pattern."""
        conditions = []
        
        # Price breaks above short SMA
        if idx + self.confirmation_bars < len(prices):
            price_above_sma = all(
                prices.iloc[idx + i] > sma_short.iloc[idx + i]
                for i in range(1, self.confirmation_bars + 1)
                if not pd.isna(sma_short.iloc[idx + i])
            )
            conditions.append(price_above_sma)
        
        # Momentum shift
        if idx >= 10 and idx + 10 < len(prices):
            momentum_before = (prices.iloc[idx] - prices.iloc[idx-10]) / prices.iloc[idx-10]
            momentum_after = (prices.iloc[idx+10] - prices.iloc[idx]) / prices.iloc[idx]
            momentum_shift = momentum_before < 0 and momentum_after > 0
            conditions.append(momentum_shift)
        
        if sum(conditions) >= 1:
            return {
                'type': 'bottom_reversal',
                'index': idx,
                'date': prices.index[idx],
                'price': prices.iloc[idx],
                'confidence': sum(conditions) / 2,
                'confirmed': True,
                'pattern': 'trough_reversal'
            }
        
        return None
    
    def _detect_pattern_reversals(self, prices: pd.Series) -> List[Dict]:
        """Detect reversal patterns like double top/bottom, head and shoulders."""
        reversals = []
        
        # Double top detection
        for i in range(self.lookback, len(prices) - self.lookback):
            pattern = self._check_double_top(prices, i)
            if pattern:
                reversals.append(pattern)
        
        # Double bottom detection
        for i in range(self.lookback, len(prices) - self.lookback):
            pattern = self._check_double_bottom(prices, i)
            if pattern:
                reversals.append(pattern)
        
        return reversals
    
    def _check_double_top(self, prices: pd.Series, idx: int) -> Optional[Dict]:
        """Check for double top pattern."""
        window = self.lookback
        
        # Find two peaks of similar height
        left_window = prices.iloc[max(0, idx-window):idx]
        right_window = prices.iloc[idx:min(len(prices), idx+window)]
        
        if len(left_window) < window//2 or len(right_window) < window//2:
            return None
        
        left_peak_idx = left_window.idxmax()
        right_peak_idx = right_window.idxmax()
        
        if pd.isna(left_peak_idx) or pd.isna(right_peak_idx):
            return None
        
        left_peak = prices.loc[left_peak_idx]
        right_peak = prices.loc[right_peak_idx]
        
        # Check if peaks are similar (within 2%)
        if abs(left_peak - right_peak) / left_peak < 0.02:
            # Find the valley between peaks
            valley_window = prices.loc[left_peak_idx:right_peak_idx]
            if len(valley_window) > 2:
                valley = valley_window.min()
                
                # Confirm neckline break
                if idx < len(prices) - 3:
                    if prices.iloc[idx:idx+3].min() < valley:
                        return {
                            'type': 'top_reversal',
                            'index': idx,
                            'date': prices.index[idx],
                            'price': right_peak,
                            'confidence': 0.7,
                            'confirmed': True,
                            'pattern': 'double_top',
                            'neckline': valley
                        }
        
        return None
    
    def _check_double_bottom(self, prices: pd.Series, idx: int) -> Optional[Dict]:
        """Check for double bottom pattern."""
        window = self.lookback
        
        left_window = prices.iloc[max(0, idx-window):idx]
        right_window = prices.iloc[idx:min(len(prices), idx+window)]
        
        if len(left_window) < window//2 or len(right_window) < window//2:
            return None
        
        left_trough_idx = left_window.idxmin()
        right_trough_idx = right_window.idxmin()
        
        if pd.isna(left_trough_idx) or pd.isna(right_trough_idx):
            return None
        
        left_trough = prices.loc[left_trough_idx]
        right_trough = prices.loc[right_trough_idx]
        
        # Check if troughs are similar
        if abs(left_trough - right_trough) / left_trough < 0.02:
            # Find the peak between troughs
            peak_window = prices.loc[left_trough_idx:right_trough_idx]
            if len(peak_window) > 2:
                peak = peak_window.max()
                
                # Confirm neckline break
                if idx < len(prices) - 3:
                    if prices.iloc[idx:idx+3].max() > peak:
                        return {
                            'type': 'bottom_reversal',
                            'index': idx,
                            'date': prices.index[idx],
                            'price': right_trough,
                            'confidence': 0.7,
                            'confirmed': True,
                            'pattern': 'double_bottom',
                            'neckline': peak
                        }
        
        return None
    
    def _remove_duplicate_reversals(self, reversals: List[Dict]) -> List[Dict]:
        """Remove duplicate reversal signals close in time."""
        if not reversals:
            return []
        
        # Sort by date
        sorted_reversals = sorted(reversals, key=lambda x: x['index'])
        
        # Remove duplicates within confirmation_bars distance
        filtered = [sorted_reversals[0]]
        for reversal in sorted_reversals[1:]:
            if reversal['index'] - filtered[-1]['index'] > self.confirmation_bars:
                filtered.append(reversal)
            elif reversal['confidence'] > filtered[-1]['confidence']:
                # Replace with higher confidence signal
                filtered[-1] = reversal
        
        return filtered


class SupportResistanceIdentifier:
    """
    Identifies support and resistance levels using multiple methods.
    """
    
    def __init__(self, lookback: int = 100, min_touches: int = 2, tolerance: float = 0.02):
        """
        Initialize support/resistance identifier.
        
        Args:
            lookback: Period to analyze
            min_touches: Minimum touches to confirm level
            tolerance: Price tolerance for level (2% default)
        """
        self.lookback = lookback
        self.min_touches = min_touches
        self.tolerance = tolerance
    
    def identify_levels(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series) -> List[SupportResistanceLevel]:
        """
        Identify support and resistance levels.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            List of support/resistance levels
        """
        if len(close) < self.lookback:
            return []
        
        levels = []
        
        # Method 1: Local extrema
        extrema_levels = self._find_extrema_levels(high, low, close)
        levels.extend(extrema_levels)
        
        # Method 2: High-volume price levels (if we had volume)
        # volume_levels = self._find_volume_levels(close, volume)
        
        # Method 3: Fibonacci retracements
        fib_levels = self._find_fibonacci_levels(high, low)
        levels.extend(fib_levels)
        
        # Method 4: Psychological levels (round numbers)
        psych_levels = self._find_psychological_levels(close)
        levels.extend(psych_levels)
        
        # Consolidate nearby levels
        consolidated = self._consolidate_levels(levels, close)
        
        # Classify as support or resistance
        classified = self._classify_levels(consolidated, close)
        
        return classified
    
    def _find_extrema_levels(self, high: pd.Series, low: pd.Series, 
                            close: pd.Series) -> List[SupportResistanceLevel]:
        """Find levels based on local extrema."""
        levels = []
        
        # Find peaks and troughs
        order = max(3, self.lookback // 20)
        peaks = signal.argrelextrema(high.values, np.greater, order=order)[0]
        troughs = signal.argrelextrema(low.values, np.less, order=order)[0]
        
        # Count touches for each level
        price_levels = {}
        
        for peak_idx in peaks:
            price = high.iloc[peak_idx]
            rounded_price = round(price / (price * self.tolerance)) * (price * self.tolerance)
            
            if rounded_price not in price_levels:
                price_levels[rounded_price] = {
                    'touches': [],
                    'breaks': 0,
                    'type': 'resistance'
                }
            price_levels[rounded_price]['touches'].append(high.index[peak_idx])
        
        for trough_idx in troughs:
            price = low.iloc[trough_idx]
            rounded_price = round(price / (price * self.tolerance)) * (price * self.tolerance)
            
            if rounded_price not in price_levels:
                price_levels[rounded_price] = {
                    'touches': [],
                    'breaks': 0,
                    'type': 'support'
                }
            price_levels[rounded_price]['touches'].append(low.index[trough_idx])
        
        # Create level objects
        for price, info in price_levels.items():
            if len(info['touches']) >= self.min_touches:
                level = SupportResistanceLevel(
                    level=price,
                    level_type=info['type'],
                    strength=len(info['touches']),
                    first_touch=min(info['touches']),
                    last_touch=max(info['touches']),
                    breaks=info['breaks'],
                    confidence=min(1.0, len(info['touches']) / 10)
                )
                levels.append(level)
        
        return levels
    
    def _find_fibonacci_levels(self, high: pd.Series, low: pd.Series) -> List[SupportResistanceLevel]:
        """Find Fibonacci retracement levels."""
        levels = []
        
        # Get recent high and low
        recent_high = high.iloc[-self.lookback:].max()
        recent_low = low.iloc[-self.lookback:].min()
        price_range = recent_high - recent_low
        
        # Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            # Retracement from high
            level_price = recent_high - (price_range * ratio)
            
            level = SupportResistanceLevel(
                level=level_price,
                level_type='support' if ratio > 0.5 else 'resistance',
                strength=1,
                first_touch=high.index[-self.lookback],
                last_touch=high.index[-1],
                breaks=0,
                confidence=0.5  # Fibonacci levels have moderate confidence
            )
            levels.append(level)
        
        return levels
    
    def _find_psychological_levels(self, close: pd.Series) -> List[SupportResistanceLevel]:
        """Find psychological (round number) levels."""
        levels = []
        
        current_price = close.iloc[-1]
        price_magnitude = 10 ** (len(str(int(current_price))) - 2)
        
        # Find round numbers near current price
        for i in range(-5, 6):
            round_level = round(current_price / price_magnitude) * price_magnitude + i * price_magnitude
            
            if round_level > 0:
                # Check how many times price touched this level
                touches = 0
                touch_dates = []
                
                for idx in range(len(close)):
                    if abs(close.iloc[idx] - round_level) / round_level < self.tolerance:
                        touches += 1
                        touch_dates.append(close.index[idx])
                
                if touches >= self.min_touches:
                    level = SupportResistanceLevel(
                        level=round_level,
                        level_type='support' if round_level < current_price else 'resistance',
                        strength=touches,
                        first_touch=min(touch_dates) if touch_dates else close.index[0],
                        last_touch=max(touch_dates) if touch_dates else close.index[-1],
                        breaks=0,
                        confidence=min(1.0, touches / 20)
                    )
                    levels.append(level)
        
        return levels
    
    def _consolidate_levels(self, levels: List[SupportResistanceLevel], 
                          close: pd.Series) -> List[SupportResistanceLevel]:
        """Consolidate nearby levels into single stronger levels."""
        if not levels:
            return []
        
        # Sort by price level
        sorted_levels = sorted(levels, key=lambda x: x.level)
        
        consolidated = []
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if close to previous level
            if abs(level.level - current_group[-1].level) / current_group[-1].level < self.tolerance:
                current_group.append(level)
            else:
                # Consolidate current group
                if current_group:
                    consolidated_level = self._merge_levels(current_group)
                    consolidated.append(consolidated_level)
                current_group = [level]
        
        # Don't forget the last group
        if current_group:
            consolidated_level = self._merge_levels(current_group)
            consolidated.append(consolidated_level)
        
        return consolidated
    
    def _merge_levels(self, levels: List[SupportResistanceLevel]) -> SupportResistanceLevel:
        """Merge multiple levels into one."""
        avg_level = np.mean([l.level for l in levels])
        total_strength = sum(l.strength for l in levels)
        
        # Determine type by majority
        support_count = sum(1 for l in levels if l.level_type == 'support')
        level_type = 'support' if support_count > len(levels) / 2 else 'resistance'
        
        return SupportResistanceLevel(
            level=avg_level,
            level_type=level_type,
            strength=total_strength,
            first_touch=min(l.first_touch for l in levels),
            last_touch=max(l.last_touch for l in levels),
            breaks=sum(l.breaks for l in levels),
            confidence=min(1.0, np.mean([l.confidence for l in levels]) * 1.2)
        )
    
    def _classify_levels(self, levels: List[SupportResistanceLevel], 
                        close: pd.Series) -> List[SupportResistanceLevel]:
        """Classify levels as support or resistance based on current price."""
        current_price = close.iloc[-1]
        
        for level in levels:
            if level.level < current_price:
                level.level_type = 'support'
            else:
                level.level_type = 'resistance'
        
        return levels


class TrendlineDetector:
    """
    Detects trendlines using linear regression and validates their significance.
    """
    
    def __init__(self, min_points: int = 3, max_deviation: float = 0.05):
        """
        Initialize trendline detector.
        
        Args:
            min_points: Minimum points to form a trendline
            max_deviation: Maximum deviation from line (5% default)
        """
        self.min_points = min_points
        self.max_deviation = max_deviation
    
    def detect_trendlines(self, prices: pd.Series) -> List[Dict[str, Any]]:
        """
        Detect significant trendlines.
        
        Args:
            prices: Price series
            
        Returns:
            List of trendlines with parameters
        """
        if len(prices) < self.min_points * 2:
            return []
        
        trendlines = []
        
        # Find peaks and troughs
        peaks = signal.argrelextrema(prices.values, np.greater, order=5)[0]
        troughs = signal.argrelextrema(prices.values, np.less, order=5)[0]
        
        # Find upper trendlines (connecting peaks)
        upper_lines = self._find_trendlines_from_points(prices, peaks, 'resistance')
        trendlines.extend(upper_lines)
        
        # Find lower trendlines (connecting troughs)
        lower_lines = self._find_trendlines_from_points(prices, troughs, 'support')
        trendlines.extend(lower_lines)
        
        # Find channel lines (parallel trendlines)
        channels = self._find_channels(upper_lines, lower_lines, prices)
        trendlines.extend(channels)
        
        return trendlines
    
    def _find_trendlines_from_points(self, prices: pd.Series, 
                                    points: np.ndarray, line_type: str) -> List[Dict]:
        """Find trendlines connecting given points."""
        trendlines = []
        
        if len(points) < self.min_points:
            return trendlines
        
        # Try different combinations of points
        for i in range(len(points) - self.min_points + 1):
            for j in range(i + self.min_points - 1, len(points)):
                # Select points for regression
                selected_points = points[i:j+1]
                
                if len(selected_points) >= self.min_points:
                    # Perform linear regression
                    x = selected_points
                    y = prices.iloc[selected_points].values
                    
                    # Calculate line parameters
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Check if line is significant
                    if abs(r_value) > 0.8 and p_value < 0.05:
                        # Validate line touches
                        touches = self._count_line_touches(prices, slope, intercept, selected_points)
                        
                        if touches >= self.min_points:
                            trendline = {
                                'type': line_type,
                                'slope': slope,
                                'intercept': intercept,
                                'r_squared': r_value ** 2,
                                'p_value': p_value,
                                'touches': touches,
                                'start_idx': selected_points[0],
                                'end_idx': selected_points[-1],
                                'start_date': prices.index[selected_points[0]],
                                'end_date': prices.index[selected_points[-1]],
                                'angle_degrees': np.degrees(np.arctan(slope)),
                                'strength': abs(r_value)
                            }
                            trendlines.append(trendline)
        
        # Remove duplicate/similar lines
        return self._remove_duplicate_lines(trendlines)
    
    def _count_line_touches(self, prices: pd.Series, slope: float, 
                           intercept: float, points: np.ndarray) -> int:
        """Count how many times price touches the trendline."""
        touches = 0
        
        for idx in range(len(prices)):
            expected_price = slope * idx + intercept
            actual_price = prices.iloc[idx]
            
            # Check if price is close to line
            if abs(actual_price - expected_price) / expected_price < self.max_deviation:
                touches += 1
        
        return touches
    
    def _find_channels(self, upper_lines: List[Dict], lower_lines: List[Dict], 
                      prices: pd.Series) -> List[Dict]:
        """Find parallel channel lines."""
        channels = []
        
        for upper in upper_lines:
            for lower in lower_lines:
                # Check if lines are roughly parallel
                slope_diff = abs(upper['slope'] - lower['slope'])
                avg_slope = (abs(upper['slope']) + abs(lower['slope'])) / 2
                
                if avg_slope > 0 and slope_diff / avg_slope < 0.2:  # Within 20% slope difference
                    # Check if they form a valid channel
                    if upper['intercept'] > lower['intercept']:
                        channel = {
                            'type': 'channel',
                            'upper_slope': upper['slope'],
                            'upper_intercept': upper['intercept'],
                            'lower_slope': lower['slope'],
                            'lower_intercept': lower['intercept'],
                            'width': upper['intercept'] - lower['intercept'],
                            'start_date': max(upper['start_date'], lower['start_date']),
                            'end_date': min(upper['end_date'], lower['end_date']),
                            'strength': (upper['strength'] + lower['strength']) / 2
                        }
                        channels.append(channel)
        
        return channels
    
    def _remove_duplicate_lines(self, trendlines: List[Dict]) -> List[Dict]:
        """Remove duplicate or very similar trendlines."""
        if len(trendlines) <= 1:
            return trendlines
        
        unique_lines = [trendlines[0]]
        
        for line in trendlines[1:]:
            is_duplicate = False
            
            for unique in unique_lines:
                # Check if slopes and intercepts are similar
                slope_similar = abs(line['slope'] - unique['slope']) < abs(unique['slope']) * 0.1
                intercept_similar = abs(line['intercept'] - unique['intercept']) < abs(unique['intercept']) * 0.1
                
                if slope_similar and intercept_similar:
                    is_duplicate = True
                    # Keep the stronger line
                    if line['r_squared'] > unique['r_squared']:
                        unique_lines.remove(unique)
                        unique_lines.append(line)
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        return unique_lines


class TrendStrengthAnalyzer:
    """
    Measures trend strength using multiple indicators.
    """
    
    def __init__(self):
        self.hp_filter = HodrickPrescottFilter()
        self.mk_test = MannKendallTest()
    
    def measure_strength(self, prices: pd.Series, period: int = 20) -> Dict[str, Any]:
        """
        Measure trend strength using multiple methods.
        
        Args:
            prices: Price series
            period: Period for calculations
            
        Returns:
            Dictionary with strength measurements
        """
        if len(prices) < period * 2:
            return {'error': 'Insufficient data'}
        
        strength_metrics = {}
        
        # 1. ADX (Average Directional Index) - simplified version
        adx = self._calculate_adx(prices, period)
        strength_metrics['adx'] = adx
        strength_metrics['adx_interpretation'] = self._interpret_adx(adx)
        
        # 2. Linear regression strength
        lr_strength = self._calculate_linear_regression_strength(prices)
        strength_metrics['linear_regression'] = lr_strength
        
        # 3. Mann-Kendall test
        mk_result = self.mk_test.test(prices)
        strength_metrics['mann_kendall'] = {
            'trend': mk_result['trend'],
            'confidence': mk_result['confidence'],
            'tau': mk_result['tau']
        }
        
        # 4. HP filter trend strength
        hp_result = self.hp_filter.decompose(prices)
        strength_metrics['hp_filter_strength'] = hp_result['trend_strength']
        
        # 5. Price position relative to moving averages
        ma_strength = self._calculate_ma_alignment(prices)
        strength_metrics['ma_alignment'] = ma_strength
        
        # 6. Momentum strength
        momentum = self._calculate_momentum_strength(prices, period)
        strength_metrics['momentum'] = momentum
        
        # Overall trend strength (0-100 scale)
        overall = self._calculate_overall_strength(strength_metrics)
        strength_metrics['overall_strength'] = overall
        strength_metrics['trend_quality'] = self._assess_trend_quality(overall)
        
        return strength_metrics
    
    def _calculate_adx(self, prices: pd.Series, period: int) -> float:
        """Simplified ADX calculation."""
        # Calculate directional movement
        price_changes = prices.diff()
        
        # Positive and negative directional movement
        plus_dm = price_changes.where(price_changes > 0, 0)
        minus_dm = -price_changes.where(price_changes < 0, 0)
        
        # Smooth the directional movements
        plus_di = plus_dm.rolling(window=period).mean()
        minus_di = minus_dm.rolling(window=period).mean()
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean().iloc[-1]
        
        return adx if not pd.isna(adx) else 0
    
    def _interpret_adx(self, adx: float) -> str:
        """Interpret ADX value."""
        if adx < 25:
            return 'weak_trend'
        elif adx < 50:
            return 'strong_trend'
        elif adx < 75:
            return 'very_strong_trend'
        else:
            return 'extremely_strong_trend'
    
    def _calculate_linear_regression_strength(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate linear regression strength."""
        x = np.arange(len(prices))
        y = prices.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'r_squared': r_value ** 2,
            'slope': slope,
            'p_value': p_value,
            'trend_direction': 'up' if slope > 0 else 'down'
        }
    
    def _calculate_ma_alignment(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate moving average alignment strength."""
        ma_periods = [10, 20, 50]
        mas = {}
        
        for period in ma_periods:
            if len(prices) >= period:
                mas[f'ma{period}'] = prices.rolling(window=period).mean().iloc[-1]
        
        if len(mas) < 2:
            return {'aligned': False, 'strength': 0}
        
        # Check if MAs are properly aligned
        ma_values = list(mas.values())
        current_price = prices.iloc[-1]
        
        # Uptrend: price > MA10 > MA20 > MA50
        # Downtrend: price < MA10 < MA20 < MA50
        if len(ma_values) == 3:
            if current_price > ma_values[0] > ma_values[1] > ma_values[2]:
                alignment = 'perfect_uptrend'
                strength = 1.0
            elif current_price < ma_values[0] < ma_values[1] < ma_values[2]:
                alignment = 'perfect_downtrend'
                strength = 1.0
            else:
                alignment = 'mixed'
                # Calculate partial alignment strength
                uptrend_score = sum([
                    current_price > ma_values[0],
                    ma_values[0] > ma_values[1] if len(ma_values) > 1 else False,
                    ma_values[1] > ma_values[2] if len(ma_values) > 2 else False
                ])
                strength = uptrend_score / 3
        else:
            alignment = 'insufficient_data'
            strength = 0
        
        return {
            'aligned': strength > 0.66,
            'alignment_type': alignment,
            'strength': strength,
            'mas': mas
        }
    
    def _calculate_momentum_strength(self, prices: pd.Series, period: int) -> Dict[str, float]:
        """Calculate momentum strength."""
        # Rate of change
        roc = ((prices.iloc[-1] - prices.iloc[-period]) / prices.iloc[-period]) * 100
        
        # Momentum oscillator
        momentum = prices.iloc[-1] - prices.iloc[-period]
        
        # Momentum acceleration
        if len(prices) >= period * 2:
            prev_momentum = prices.iloc[-period] - prices.iloc[-period*2]
            acceleration = momentum - prev_momentum
        else:
            acceleration = 0
        
        return {
            'rate_of_change': roc,
            'momentum': momentum,
            'acceleration': acceleration,
            'strength': min(abs(roc) / 10, 1.0)  # Normalize to 0-1
        }
    
    def _calculate_overall_strength(self, metrics: Dict) -> float:
        """Calculate overall trend strength score (0-100)."""
        scores = []
        
        # ADX score (0-100 scale)
        if 'adx' in metrics:
            scores.append(min(metrics['adx'], 100))
        
        # R-squared score (0-100 scale)
        if 'linear_regression' in metrics:
            scores.append(metrics['linear_regression']['r_squared'] * 100)
        
        # Mann-Kendall confidence (0-100 scale)
        if 'mann_kendall' in metrics:
            scores.append(metrics['mann_kendall']['confidence'] * 100)
        
        # HP filter strength (0-100 scale)
        if 'hp_filter_strength' in metrics:
            scores.append(metrics['hp_filter_strength'] * 100)
        
        # MA alignment (0-100 scale)
        if 'ma_alignment' in metrics:
            scores.append(metrics['ma_alignment']['strength'] * 100)
        
        # Momentum strength (0-100 scale)
        if 'momentum' in metrics:
            scores.append(metrics['momentum']['strength'] * 100)
        
        return np.mean(scores) if scores else 0
    
    def _assess_trend_quality(self, overall_strength: float) -> str:
        """Assess trend quality based on overall strength."""
        if overall_strength < 20:
            return 'no_trend'
        elif overall_strength < 40:
            return 'weak_trend'
        elif overall_strength < 60:
            return 'moderate_trend'
        elif overall_strength < 80:
            return 'strong_trend'
        else:
            return 'very_strong_trend'