"""
Volume-based technical indicators for market analysis.
Implements OBV, VROC, VWAP, A/D Line, and volume spike detection.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import Indicator, IndicatorResult, IndicatorConfig


@dataclass
class VolumeIndicatorConfig(IndicatorConfig):
    """Configuration for volume indicators."""
    spike_threshold: float = 2.0  # Multiple of average volume for spike detection
    spike_lookback: int = 20  # Days to calculate average volume
    vroc_period: int = 14  # Period for Volume Rate of Change
    vwap_anchor: str = "session"  # 'session', 'week', 'month'
    ad_use_close: bool = True  # Use close price for A/D calculation


class OnBalanceVolume(Indicator):
    """
    On-Balance Volume (OBV) indicator.
    Measures buying and selling pressure by adding volume on up days
    and subtracting volume on down days.
    """
    
    def __init__(self, config: Optional[VolumeIndicatorConfig] = None):
        super().__init__(config or VolumeIndicatorConfig())
        self.name = "OBV"
        self.description = "On-Balance Volume - Cumulative volume flow indicator"
    
    def calculate(self, prices: pd.Series, volumes: pd.Series) -> IndicatorResult:
        """
        Calculate On-Balance Volume.
        
        Args:
            prices: Series of closing prices
            volumes: Series of volume data
            
        Returns:
            IndicatorResult with OBV values
        """
        if len(prices) != len(volumes):
            raise ValueError("Price and volume series must have same length")
        
        if len(prices) < 2:
            raise ValueError("Need at least 2 data points for OBV calculation")
        
        # Calculate price changes
        price_changes = prices.diff()
        
        # Initialize OBV
        obv = np.zeros(len(prices))
        obv[0] = volumes.iloc[0]
        
        # Calculate OBV
        for i in range(1, len(prices)):
            if price_changes.iloc[i] > 0:
                # Price went up, add volume
                obv[i] = obv[i-1] + volumes.iloc[i]
            elif price_changes.iloc[i] < 0:
                # Price went down, subtract volume
                obv[i] = obv[i-1] - volumes.iloc[i]
            else:
                # Price unchanged, OBV stays same
                obv[i] = obv[i-1]
        
        # Create result series
        obv_series = pd.Series(obv, index=prices.index)
        
        # Calculate signal line (20-day EMA of OBV)
        signal_line = obv_series.ewm(span=20, adjust=False).mean()
        
        # Generate signals
        signals = self._generate_obv_signals(obv_series, signal_line, prices)
        
        return IndicatorResult(
            indicator_name=self.name,
            values=obv_series,
            signal=signals['signal'],
            metadata={
                'signal_line': signal_line,
                'divergence': signals['divergence'],
                'trend': signals['trend'],
                'last_obv': obv_series.iloc[-1],
                'obv_change_pct': ((obv_series.iloc[-1] - obv_series.iloc[-20]) / abs(obv_series.iloc[-20]) * 100) if len(obv_series) > 20 else 0
            }
        )
    
    def _generate_obv_signals(self, obv: pd.Series, signal_line: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """Generate trading signals from OBV."""
        signals = np.zeros(len(obv))
        divergence = []
        
        # Detect OBV/Price divergence
        if len(obv) >= 20:
            for i in range(20, len(obv)):
                # Bullish divergence: price making lower lows, OBV making higher lows
                price_ll = prices.iloc[i] < prices.iloc[i-10:i].min()
                obv_hl = obv.iloc[i] > obv.iloc[i-10:i].min()
                
                if price_ll and obv_hl:
                    signals[i] = 1  # Buy signal
                    divergence.append(('bullish', i))
                
                # Bearish divergence: price making higher highs, OBV making lower highs
                price_hh = prices.iloc[i] > prices.iloc[i-10:i].max()
                obv_lh = obv.iloc[i] < obv.iloc[i-10:i].max()
                
                if price_hh and obv_lh:
                    signals[i] = -1  # Sell signal
                    divergence.append(('bearish', i))
        
        # Determine trend
        trend = 'neutral'
        if len(obv) >= 50:
            recent_obv = obv.iloc[-20:].mean()
            older_obv = obv.iloc[-50:-20].mean()
            if recent_obv > older_obv * 1.05:
                trend = 'bullish'
            elif recent_obv < older_obv * 0.95:
                trend = 'bearish'
        
        return {
            'signal': pd.Series(signals, index=obv.index),
            'divergence': divergence,
            'trend': trend
        }


class VolumeRateOfChange(Indicator):
    """
    Volume Rate of Change (VROC) indicator.
    Measures the rate of change in volume over a specified period.
    """
    
    def __init__(self, config: Optional[VolumeIndicatorConfig] = None):
        super().__init__(config or VolumeIndicatorConfig())
        self.name = "VROC"
        self.description = "Volume Rate of Change - Momentum indicator for volume"
    
    def calculate(self, volumes: pd.Series, period: Optional[int] = None) -> IndicatorResult:
        """
        Calculate Volume Rate of Change.
        
        Args:
            volumes: Series of volume data
            period: Lookback period (default from config)
            
        Returns:
            IndicatorResult with VROC values
        """
        period = period or self.config.vroc_period
        
        if len(volumes) < period + 1:
            raise ValueError(f"Need at least {period + 1} data points for VROC calculation")
        
        # Calculate VROC: ((Volume - Volume[n periods ago]) / Volume[n periods ago]) * 100
        vroc = ((volumes - volumes.shift(period)) / volumes.shift(period)) * 100
        
        # Smooth with 3-period MA
        vroc_smooth = vroc.rolling(window=3).mean()
        
        # Generate signals based on VROC levels and crossovers
        signals = self._generate_vroc_signals(vroc, vroc_smooth)
        
        return IndicatorResult(
            indicator_name=self.name,
            values=vroc,
            signal=signals['signal'],
            metadata={
                'vroc_smooth': vroc_smooth,
                'current_vroc': vroc.iloc[-1] if not vroc.empty else None,
                'vroc_trend': signals['trend'],
                'extreme_levels': signals['extremes'],
                'period': period
            }
        )
    
    def _generate_vroc_signals(self, vroc: pd.Series, vroc_smooth: pd.Series) -> Dict[str, Any]:
        """Generate trading signals from VROC."""
        signals = np.zeros(len(vroc))
        extremes = []
        
        # Signal when VROC crosses zero line
        for i in range(1, len(vroc)):
            if not pd.isna(vroc.iloc[i]) and not pd.isna(vroc.iloc[i-1]):
                # Bullish: VROC crosses above zero
                if vroc.iloc[i-1] <= 0 and vroc.iloc[i] > 0:
                    signals[i] = 1
                
                # Bearish: VROC crosses below zero
                elif vroc.iloc[i-1] >= 0 and vroc.iloc[i] < 0:
                    signals[i] = -1
                
                # Mark extreme levels
                if vroc.iloc[i] > 50:
                    extremes.append(('overbought', i))
                elif vroc.iloc[i] < -50:
                    extremes.append(('oversold', i))
        
        # Determine trend
        trend = 'neutral'
        if len(vroc) >= 10 and not pd.isna(vroc.iloc[-5:]).any():
            recent_vroc = vroc.iloc[-5:].mean()
            if recent_vroc > 10:
                trend = 'increasing_volume'
            elif recent_vroc < -10:
                trend = 'decreasing_volume'
        
        return {
            'signal': pd.Series(signals, index=vroc.index),
            'trend': trend,
            'extremes': extremes
        }


class VolumeWeightedAveragePrice(Indicator):
    """
    Volume-Weighted Average Price (VWAP) indicator.
    Calculates the average price weighted by volume.
    """
    
    def __init__(self, config: Optional[VolumeIndicatorConfig] = None):
        super().__init__(config or VolumeIndicatorConfig())
        self.name = "VWAP"
        self.description = "Volume-Weighted Average Price"
    
    def calculate(self, 
                 high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 volume: pd.Series,
                 anchor: Optional[str] = None) -> IndicatorResult:
        """
        Calculate VWAP.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            anchor: Anchor point for VWAP ('session', 'week', 'month')
            
        Returns:
            IndicatorResult with VWAP values
        """
        anchor = anchor or self.config.vwap_anchor
        
        if not all(len(s) == len(high) for s in [low, close, volume]):
            raise ValueError("All price and volume series must have same length")
        
        # Calculate typical price: (High + Low + Close) / 3
        typical_price = (high + low + close) / 3
        
        # Calculate cumulative values based on anchor
        if anchor == 'session':
            # Reset daily (assuming datetime index)
            cumulative_volume = volume.groupby(volume.index.date).cumsum()
            cumulative_pv = (typical_price * volume).groupby(volume.index.date).cumsum()
        elif anchor == 'week':
            # Reset weekly
            cumulative_volume = volume.groupby(pd.Grouper(freq='W')).cumsum()
            cumulative_pv = (typical_price * volume).groupby(pd.Grouper(freq='W')).cumsum()
        elif anchor == 'month':
            # Reset monthly
            cumulative_volume = volume.groupby(pd.Grouper(freq='M')).cumsum()
            cumulative_pv = (typical_price * volume).groupby(pd.Grouper(freq='M')).cumsum()
        else:
            # No reset (cumulative from start)
            cumulative_volume = volume.cumsum()
            cumulative_pv = (typical_price * volume).cumsum()
        
        # Calculate VWAP
        vwap = cumulative_pv / cumulative_volume
        
        # Calculate VWAP bands (1 and 2 standard deviations)
        vwap_std = self._calculate_vwap_bands(typical_price, vwap, volume)
        
        # Generate signals
        signals = self._generate_vwap_signals(close, vwap, vwap_std)
        
        return IndicatorResult(
            indicator_name=self.name,
            values=vwap,
            signal=signals['signal'],
            metadata={
                'upper_band_1': vwap + vwap_std,
                'lower_band_1': vwap - vwap_std,
                'upper_band_2': vwap + 2 * vwap_std,
                'lower_band_2': vwap - 2 * vwap_std,
                'current_vwap': vwap.iloc[-1] if not vwap.empty else None,
                'price_position': signals['price_position'],
                'anchor': anchor
            }
        )
    
    def _calculate_vwap_bands(self, typical_price: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate VWAP standard deviation bands."""
        # Calculate squared deviations weighted by volume
        squared_deviations = ((typical_price - vwap) ** 2) * volume
        
        # Calculate cumulative weighted variance
        cumulative_variance = squared_deviations.cumsum() / volume.cumsum()
        
        # Return standard deviation
        return np.sqrt(cumulative_variance)
    
    def _generate_vwap_signals(self, close: pd.Series, vwap: pd.Series, vwap_std: pd.Series) -> Dict[str, Any]:
        """Generate trading signals from VWAP."""
        signals = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if pd.isna(vwap.iloc[i]) or pd.isna(vwap_std.iloc[i]):
                continue
            
            # Buy signal: price crosses above VWAP from below
            if close.iloc[i-1] < vwap.iloc[i-1] and close.iloc[i] > vwap.iloc[i]:
                signals[i] = 1
            
            # Sell signal: price crosses below VWAP from above
            elif close.iloc[i-1] > vwap.iloc[i-1] and close.iloc[i] < vwap.iloc[i]:
                signals[i] = -1
            
            # Strong buy: price touches lower band 2
            elif close.iloc[i] <= vwap.iloc[i] - 2 * vwap_std.iloc[i]:
                signals[i] = 2
            
            # Strong sell: price touches upper band 2
            elif close.iloc[i] >= vwap.iloc[i] + 2 * vwap_std.iloc[i]:
                signals[i] = -2
        
        # Determine price position relative to VWAP
        price_position = 'at_vwap'
        if len(close) > 0 and not pd.isna(vwap.iloc[-1]):
            if close.iloc[-1] > vwap.iloc[-1] + vwap_std.iloc[-1]:
                price_position = 'above_upper_band'
            elif close.iloc[-1] > vwap.iloc[-1]:
                price_position = 'above_vwap'
            elif close.iloc[-1] < vwap.iloc[-1] - vwap_std.iloc[-1]:
                price_position = 'below_lower_band'
            elif close.iloc[-1] < vwap.iloc[-1]:
                price_position = 'below_vwap'
        
        return {
            'signal': pd.Series(signals, index=close.index),
            'price_position': price_position
        }


class AccumulationDistributionLine(Indicator):
    """
    Accumulation/Distribution Line indicator.
    Measures the cumulative flow of money into and out of a security.
    """
    
    def __init__(self, config: Optional[VolumeIndicatorConfig] = None):
        super().__init__(config or VolumeIndicatorConfig())
        self.name = "A/D Line"
        self.description = "Accumulation/Distribution Line"
    
    def calculate(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 volume: pd.Series) -> IndicatorResult:
        """
        Calculate Accumulation/Distribution Line.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            
        Returns:
            IndicatorResult with A/D Line values
        """
        if not all(len(s) == len(high) for s in [low, close, volume]):
            raise ValueError("All price and volume series must have same length")
        
        # Calculate Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
        mf_multiplier = np.where(
            high != low,
            ((close - low) - (high - close)) / (high - low),
            0  # If high equals low, multiplier is 0
        )
        
        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Calculate A/D Line (cumulative sum)
        ad_line = pd.Series(mf_volume, index=close.index).cumsum()
        
        # Calculate signal line (3-day EMA)
        signal_line = ad_line.ewm(span=3, adjust=False).mean()
        
        # Generate signals
        signals = self._generate_ad_signals(ad_line, signal_line, close)
        
        return IndicatorResult(
            indicator_name=self.name,
            values=ad_line,
            signal=signals['signal'],
            metadata={
                'signal_line': signal_line,
                'money_flow_multiplier': pd.Series(mf_multiplier, index=close.index),
                'money_flow_volume': pd.Series(mf_volume, index=close.index),
                'current_ad': ad_line.iloc[-1] if not ad_line.empty else None,
                'divergence': signals['divergence'],
                'trend': signals['trend']
            }
        )
    
    def _generate_ad_signals(self, ad_line: pd.Series, signal_line: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """Generate trading signals from A/D Line."""
        signals = np.zeros(len(ad_line))
        divergence = []
        
        # Detect divergences
        if len(ad_line) >= 20:
            for i in range(20, len(ad_line)):
                # Bullish divergence
                price_ll = prices.iloc[i] < prices.iloc[i-10:i].min()
                ad_hl = ad_line.iloc[i] > ad_line.iloc[i-10:i].min()
                
                if price_ll and ad_hl:
                    signals[i] = 1
                    divergence.append(('bullish', i))
                
                # Bearish divergence
                price_hh = prices.iloc[i] > prices.iloc[i-10:i].max()
                ad_lh = ad_line.iloc[i] < ad_line.iloc[i-10:i].max()
                
                if price_hh and ad_lh:
                    signals[i] = -1
                    divergence.append(('bearish', i))
        
        # Crossover signals
        for i in range(1, len(ad_line)):
            if ad_line.iloc[i-1] < signal_line.iloc[i-1] and ad_line.iloc[i] > signal_line.iloc[i]:
                if signals[i] == 0:  # Don't override divergence signals
                    signals[i] = 0.5  # Weak buy
            elif ad_line.iloc[i-1] > signal_line.iloc[i-1] and ad_line.iloc[i] < signal_line.iloc[i]:
                if signals[i] == 0:
                    signals[i] = -0.5  # Weak sell
        
        # Determine trend
        trend = 'neutral'
        if len(ad_line) >= 50:
            recent_ad = ad_line.iloc[-20:].mean()
            older_ad = ad_line.iloc[-50:-20].mean()
            if recent_ad > older_ad * 1.05:
                trend = 'accumulation'
            elif recent_ad < older_ad * 0.95:
                trend = 'distribution'
        
        return {
            'signal': pd.Series(signals, index=ad_line.index),
            'divergence': divergence,
            'trend': trend
        }


class VolumeSpikeDetector(Indicator):
    """
    Volume Spike Detection algorithm.
    Identifies unusual volume activity that may signal important market events.
    """
    
    def __init__(self, config: Optional[VolumeIndicatorConfig] = None):
        super().__init__(config or VolumeIndicatorConfig())
        self.name = "Volume Spike Detector"
        self.description = "Detects unusual volume spikes"
    
    def calculate(self, 
                 volume: pd.Series,
                 close: Optional[pd.Series] = None) -> IndicatorResult:
        """
        Detect volume spikes.
        
        Args:
            volume: Series of volume data
            close: Optional series of closing prices for context
            
        Returns:
            IndicatorResult with spike detection
        """
        lookback = self.config.spike_lookback
        threshold = self.config.spike_threshold
        
        if len(volume) < lookback:
            raise ValueError(f"Need at least {lookback} data points for spike detection")
        
        # Calculate rolling average and standard deviation
        volume_ma = volume.rolling(window=lookback).mean()
        volume_std = volume.rolling(window=lookback).std()
        
        # Calculate z-score
        z_score = (volume - volume_ma) / volume_std
        
        # Detect spikes
        spikes = np.zeros(len(volume))
        spike_events = []
        
        for i in range(lookback, len(volume)):
            # Check for volume spike (using both threshold and z-score)
            if volume.iloc[i] > threshold * volume_ma.iloc[i] or abs(z_score.iloc[i]) > 2:
                spikes[i] = 1
                
                # Classify spike type
                spike_type = self._classify_spike(
                    volume.iloc[i],
                    volume_ma.iloc[i],
                    close.iloc[i] if close is not None else None,
                    close.iloc[i-1] if close is not None and i > 0 else None
                )
                
                spike_events.append({
                    'index': i,
                    'timestamp': volume.index[i],
                    'volume': volume.iloc[i],
                    'average_volume': volume_ma.iloc[i],
                    'spike_ratio': volume.iloc[i] / volume_ma.iloc[i],
                    'z_score': z_score.iloc[i],
                    'type': spike_type
                })
        
        # Analyze spike patterns
        spike_analysis = self._analyze_spike_patterns(spike_events, volume)
        
        return IndicatorResult(
            indicator_name=self.name,
            values=pd.Series(spikes, index=volume.index),
            signal=pd.Series(spikes, index=volume.index),
            metadata={
                'volume_ma': volume_ma,
                'volume_std': volume_std,
                'z_score': z_score,
                'spike_events': spike_events,
                'spike_count': len(spike_events),
                'spike_analysis': spike_analysis,
                'threshold': threshold,
                'lookback': lookback
            }
        )
    
    def _classify_spike(self, 
                       current_volume: float,
                       avg_volume: float,
                       current_price: Optional[float],
                       prev_price: Optional[float]) -> str:
        """Classify the type of volume spike."""
        spike_ratio = current_volume / avg_volume
        
        # Determine spike magnitude
        if spike_ratio > 5:
            magnitude = 'extreme'
        elif spike_ratio > 3:
            magnitude = 'high'
        elif spike_ratio > 2:
            magnitude = 'moderate'
        else:
            magnitude = 'mild'
        
        # Determine price action if available
        price_action = 'unknown'
        if current_price is not None and prev_price is not None:
            price_change = (current_price - prev_price) / prev_price * 100
            if price_change > 1:
                price_action = 'bullish'
            elif price_change < -1:
                price_action = 'bearish'
            else:
                price_action = 'neutral'
        
        return f"{magnitude}_{price_action}"
    
    def _analyze_spike_patterns(self, spike_events: List[Dict], volume: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in volume spikes."""
        if not spike_events:
            return {'pattern': 'no_spikes', 'frequency': 0}
        
        # Calculate spike frequency
        total_days = len(volume)
        spike_frequency = len(spike_events) / total_days * 100
        
        # Analyze clustering
        clustering = 'none'
        if len(spike_events) >= 3:
            # Check for clustered spikes (multiple spikes within 5 days)
            for i in range(len(spike_events) - 2):
                if spike_events[i+2]['index'] - spike_events[i]['index'] <= 5:
                    clustering = 'clustered'
                    break
        
        # Analyze trend
        if len(spike_events) >= 2:
            recent_spikes = [s for s in spike_events if s['index'] >= len(volume) - 20]
            older_spikes = [s for s in spike_events if s['index'] < len(volume) - 20]
            
            if len(recent_spikes) > len(older_spikes):
                trend = 'increasing'
            elif len(recent_spikes) < len(older_spikes):
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'pattern': clustering,
            'frequency': spike_frequency,
            'trend': trend,
            'avg_spike_ratio': np.mean([s['spike_ratio'] for s in spike_events]),
            'max_spike_ratio': max([s['spike_ratio'] for s in spike_events])
        }


class VolumeAnalyzer:
    """
    Comprehensive volume analysis combining multiple indicators.
    """
    
    def __init__(self, config: Optional[VolumeIndicatorConfig] = None):
        self.config = config or VolumeIndicatorConfig()
        self.obv = OnBalanceVolume(self.config)
        self.vroc = VolumeRateOfChange(self.config)
        self.vwap = VolumeWeightedAveragePrice(self.config)
        self.ad_line = AccumulationDistributionLine(self.config)
        self.spike_detector = VolumeSpikeDetector(self.config)
    
    def analyze(self,
               high: pd.Series,
               low: pd.Series,
               close: pd.Series,
               volume: pd.Series) -> Dict[str, IndicatorResult]:
        """
        Perform comprehensive volume analysis.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            
        Returns:
            Dictionary of indicator results
        """
        results = {}
        
        # Calculate all volume indicators
        try:
            results['obv'] = self.obv.calculate(close, volume)
        except Exception as e:
            print(f"OBV calculation failed: {e}")
        
        try:
            results['vroc'] = self.vroc.calculate(volume)
        except Exception as e:
            print(f"VROC calculation failed: {e}")
        
        try:
            results['vwap'] = self.vwap.calculate(high, low, close, volume)
        except Exception as e:
            print(f"VWAP calculation failed: {e}")
        
        try:
            results['ad_line'] = self.ad_line.calculate(high, low, close, volume)
        except Exception as e:
            print(f"A/D Line calculation failed: {e}")
        
        try:
            results['volume_spikes'] = self.spike_detector.calculate(volume, close)
        except Exception as e:
            print(f"Volume spike detection failed: {e}")
        
        # Generate composite signal
        composite_signal = self._generate_composite_signal(results)
        
        # Add summary
        results['summary'] = {
            'composite_signal': composite_signal,
            'indicators_calculated': list(results.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _generate_composite_signal(self, results: Dict[str, IndicatorResult]) -> str:
        """Generate composite signal from all volume indicators."""
        signals = []
        
        # Collect latest signals from each indicator
        for name, result in results.items():
            if hasattr(result, 'signal') and len(result.signal) > 0:
                latest_signal = result.signal.iloc[-1]
                if latest_signal != 0:
                    signals.append(latest_signal)
        
        if not signals:
            return 'neutral'
        
        # Calculate average signal
        avg_signal = np.mean(signals)
        
        if avg_signal > 0.5:
            return 'strong_buy'
        elif avg_signal > 0:
            return 'buy'
        elif avg_signal < -0.5:
            return 'strong_sell'
        elif avg_signal < 0:
            return 'sell'
        else:
            return 'neutral'