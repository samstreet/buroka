import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.core.analysis.volume_price_analysis import (
    PriceVolumeDivergence,
    UnusualVolumePatterns,
    AccumulationDistributionPatterns,
    VolumeConfirmation,
    VolumeProfileAnalysis,
    VolumeSignalStrength,
    DivergenceType,
    VolumePatternType,
    WyckoffPhase,
    SignalCategory
)


class TestPriceVolumeDivergence:
    def setup_method(self):
        self.divergence = PriceVolumeDivergence()
        self.dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
    def test_bullish_divergence_detection(self):
        prices = np.array([100 - i*0.5 for i in range(50)])
        volumes = np.array([1000000 + i*10000 for i in range(50)])
        
        divergences = self.divergence.detect_divergence(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates[:50]
        )
        
        assert len(divergences) > 0
        bullish = [d for d in divergences if d['type'] == DivergenceType.BULLISH]
        assert len(bullish) > 0
        assert bullish[0]['confidence'] > 0.5
        
    def test_bearish_divergence_detection(self):
        prices = np.array([100 + i*0.5 for i in range(50)])
        volumes = np.array([1000000 - i*10000 for i in range(50)])
        
        divergences = self.divergence.detect_divergence(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates[:50]
        )
        
        bearish = [d for d in divergences if d['type'] == DivergenceType.BEARISH]
        assert len(bearish) > 0
        assert bearish[0]['confidence'] > 0.5
        
    def test_hidden_divergence(self):
        prices = np.concatenate([
            [100 + i for i in range(20)],
            [120 - i*0.5 for i in range(10)],
            [115 + i for i in range(20)]
        ])
        volumes = np.concatenate([
            [1000000 + i*5000 for i in range(20)],
            [1100000 - i*10000 for i in range(10)],
            [1000000 - i*5000 for i in range(20)]
        ])
        
        divergences = self.divergence.detect_divergence(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates[:50]
        )
        
        hidden = [d for d in divergences if d['type'] in [DivergenceType.HIDDEN_BULLISH, DivergenceType.HIDDEN_BEARISH]]
        assert len(hidden) > 0
        
    def test_divergence_strength_calculation(self):
        prices = np.array([100 - i for i in range(30)])
        volumes = np.array([1000000 + i*20000 for i in range(30)])
        
        strength = self.divergence.calculate_divergence_strength(
            price_trend=-1.0,
            volume_trend=1.0,
            correlation=-0.8
        )
        
        assert 0 <= strength <= 1
        assert strength > 0.7
        
    def test_volume_confirmation_requirement(self):
        prices = np.array([100 + np.sin(i/5) * 10 for i in range(50)])
        volumes = np.array([1000000 + np.random.randn() * 100000 for _ in range(50)])
        
        divergences = self.divergence.detect_divergence(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates[:50],
            min_confidence=0.8
        )
        
        for divergence in divergences:
            assert divergence['confidence'] >= 0.8
            assert divergence['volume_confirmation'] is not None


class TestUnusualVolumePatterns:
    def setup_method(self):
        self.patterns = UnusualVolumePatterns()
        self.dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
    def test_volume_climax_detection(self):
        volumes = np.array([1000000] * 50 + [5000000] * 2 + [1000000] * 48)
        prices = np.array([100] * 50 + [110, 108] + [100] * 48)
        
        patterns = self.patterns.detect_unusual_patterns(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        climax = [p for p in patterns if p['type'] == VolumePatternType.CLIMAX]
        assert len(climax) > 0
        assert climax[0]['confidence'] > 0.7
        
    def test_volume_dry_up_detection(self):
        volumes = np.array([1000000] * 50 + [100000] * 10 + [1000000] * 40)
        prices = np.array([100 + i*0.1 for i in range(100)])
        
        patterns = self.patterns.detect_unusual_patterns(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        dry_up = [p for p in patterns if p['type'] == VolumePatternType.DRY_UP]
        assert len(dry_up) > 0
        
    def test_breakout_volume_detection(self):
        volumes = np.array([1000000] * 70 + [3000000] * 5 + [2000000] * 25)
        prices = np.array([100] * 70 + [105, 107, 109, 111, 113] + [112] * 25)
        
        patterns = self.patterns.detect_unusual_patterns(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        breakout = [p for p in patterns if p['type'] == VolumePatternType.BREAKOUT]
        assert len(breakout) > 0
        assert breakout[0]['price_change'] > 0
        
    def test_churning_detection(self):
        volumes = np.array([3000000] * 100)
        prices = np.array([100 + np.sin(i/5) * 2 for i in range(100)])
        
        patterns = self.patterns.detect_unusual_patterns(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        churning = [p for p in patterns if p['type'] == VolumePatternType.CHURNING]
        assert len(churning) > 0
        
    def test_pocket_pivot_detection(self):
        volumes = np.array([1000000] * 60 + [2500000] + [1500000] * 39)
        prices = np.array([100] * 60 + [105] + [105] * 39)
        
        patterns = self.patterns.detect_unusual_patterns(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        pivot = [p for p in patterns if p['type'] == VolumePatternType.POCKET_PIVOT]
        assert len(pivot) > 0


class TestAccumulationDistributionPatterns:
    def setup_method(self):
        self.ad_patterns = AccumulationDistributionPatterns()
        self.dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
    def test_accumulation_phase_detection(self):
        prices = np.array([100 - i*0.1 for i in range(50)] + 
                         [95 + np.sin(i/5) * 2 for i in range(100)] +
                         [95 + i*0.2 for i in range(50)])
        volumes = np.array([1000000 + i*5000 for i in range(200)])
        
        patterns = self.ad_patterns.detect_accumulation_distribution(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        accumulation = [p for p in patterns if 'accumulation' in p['pattern'].lower()]
        assert len(accumulation) > 0
        
    def test_distribution_phase_detection(self):
        prices = np.array([100 + i*0.2 for i in range(50)] + 
                         [110 + np.sin(i/5) * 2 for i in range(100)] +
                         [110 - i*0.1 for i in range(50)])
        volumes = np.array([1000000 + i*5000 for i in range(50)] +
                          [1500000] * 100 +
                          [1500000 - i*10000 for i in range(50)])
        
        patterns = self.ad_patterns.detect_accumulation_distribution(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        distribution = [p for p in patterns if 'distribution' in p['pattern'].lower()]
        assert len(distribution) > 0
        
    def test_wyckoff_phase_identification(self):
        prices = np.array([100] * 30 + [95] * 30 + [97] * 30 + [102] * 30)
        volumes = np.array([1000000] * 30 + [1500000] * 30 + [1200000] * 30 + [2000000] * 30)
        
        phase = self.ad_patterns.identify_wyckoff_phase(
            prices=prices,
            volumes=volumes
        )
        
        assert phase in WyckoffPhase
        assert phase != WyckoffPhase.NONE
        
    def test_smart_money_flow(self):
        prices = np.array([100 + i*0.1 for i in range(100)])
        volumes = np.array([1000000] * 50 + [2000000] * 50)
        highs = prices + 1
        lows = prices - 1
        
        flow = self.ad_patterns.calculate_smart_money_flow(
            prices=prices,
            volumes=volumes,
            highs=highs,
            lows=lows
        )
        
        assert len(flow) == len(prices)
        assert np.mean(flow[50:]) > np.mean(flow[:50])


class TestVolumeConfirmation:
    def setup_method(self):
        self.confirmation = VolumeConfirmation()
        self.dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
    def test_breakout_confirmation(self):
        prices = np.array([100] * 70 + [105, 107, 109] + [108] * 27)
        volumes = np.array([1000000] * 70 + [3000000, 2500000, 2000000] + [1500000] * 27)
        
        confirmed = self.confirmation.confirm_price_pattern(
            pattern_type='breakout',
            pattern_data={
                'breakout_index': 70,
                'resistance_level': 100,
                'breakout_price': 105
            },
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        assert confirmed['is_confirmed'] == True
        assert confirmed['confidence'] > 0.7
        assert confirmed['volume_ratio'] > 2.0
        
    def test_reversal_confirmation(self):
        prices = np.array([100 - i*0.5 for i in range(50)] + 
                         [75 + i*0.5 for i in range(50)])
        volumes = np.array([1000000] * 45 + [2500000] * 10 + [1500000] * 45)
        
        confirmed = self.confirmation.confirm_price_pattern(
            pattern_type='reversal',
            pattern_data={
                'reversal_index': 50,
                'reversal_price': 75
            },
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        assert confirmed['is_confirmed'] == True
        assert confirmed['volume_surge'] == True
        
    def test_continuation_confirmation(self):
        prices = np.array([100 + i*0.3 for i in range(100)])
        volumes = np.array([1500000 + i*5000 for i in range(100)])
        
        confirmed = self.confirmation.confirm_price_pattern(
            pattern_type='continuation',
            pattern_data={
                'trend_direction': 1,
                'pattern_start': 20,
                'pattern_end': 80
            },
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        assert confirmed['is_confirmed'] == True
        assert confirmed['volume_trend'] == 'increasing'
        
    def test_failed_confirmation(self):
        prices = np.array([100] * 70 + [105] * 30)
        volumes = np.array([1000000] * 70 + [800000] * 30)
        
        confirmed = self.confirmation.confirm_price_pattern(
            pattern_type='breakout',
            pattern_data={
                'breakout_index': 70,
                'resistance_level': 100
            },
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        assert confirmed['is_confirmed'] == False
        assert confirmed['failure_reason'] == 'insufficient_volume'


class TestVolumeProfileAnalysis:
    def setup_method(self):
        self.profile = VolumeProfileAnalysis()
        self.dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
    def test_volume_profile_creation(self):
        prices = np.array([100 + np.sin(i/10) * 10 for i in range(200)])
        volumes = np.array([1000000 + np.random.randn() * 100000 for _ in range(200)])
        
        profile = self.profile.create_volume_profile(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates,
            bins=20
        )
        
        assert 'price_levels' in profile
        assert 'volume_distribution' in profile
        assert 'poc' in profile
        assert 'value_area_high' in profile
        assert 'value_area_low' in profile
        assert len(profile['price_levels']) == 20
        
    def test_point_of_control_identification(self):
        prices = np.concatenate([
            np.random.normal(100, 2, 50),
            np.random.normal(105, 1, 100),
            np.random.normal(110, 2, 50)
        ])
        volumes = np.concatenate([
            np.ones(50) * 1000000,
            np.ones(100) * 2000000,
            np.ones(50) * 1000000
        ])
        
        profile = self.profile.create_volume_profile(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates,
            bins=20
        )
        
        assert 103 < profile['poc'] < 107
        
    def test_value_area_calculation(self):
        prices = np.random.normal(100, 5, 200)
        volumes = np.random.uniform(900000, 1100000, 200)
        
        profile = self.profile.create_volume_profile(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates,
            bins=30
        )
        
        assert profile['value_area_low'] < profile['poc']
        assert profile['value_area_high'] > profile['poc']
        assert profile['value_area_volume_pct'] >= 0.68
        
    def test_market_structure_identification(self):
        prices = np.concatenate([
            np.linspace(100, 105, 50),
            np.linspace(105, 103, 30),
            np.linspace(103, 108, 50),
            np.linspace(108, 106, 30),
            np.linspace(106, 110, 40)
        ])
        volumes = np.random.uniform(900000, 1100000, 200)
        
        structure = self.profile.identify_market_structure(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        assert 'structure_type' in structure
        assert 'confidence' in structure
        assert 'key_levels' in structure
        assert len(structure['key_levels']) > 0
        
    def test_volume_node_detection(self):
        prices = np.concatenate([
            np.random.normal(100, 1, 50),
            np.random.normal(105, 1, 30),
            np.random.normal(110, 1, 50),
            np.random.normal(115, 1, 30),
            np.random.normal(120, 1, 40)
        ])
        volumes = np.concatenate([
            np.ones(50) * 2000000,
            np.ones(30) * 500000,
            np.ones(50) * 2500000,
            np.ones(30) * 600000,
            np.ones(40) * 1800000
        ])
        
        nodes = self.profile.detect_volume_nodes(
            prices=prices,
            volumes=volumes,
            timestamps=self.dates
        )
        
        assert 'high_volume_nodes' in nodes
        assert 'low_volume_nodes' in nodes
        assert len(nodes['high_volume_nodes']) >= 2
        assert len(nodes['low_volume_nodes']) >= 1


class TestVolumeSignalStrength:
    def setup_method(self):
        self.signal_strength = VolumeSignalStrength()
        
    def test_signal_strength_calculation(self):
        signal_data = {
            'volume_ratio': 2.5,
            'price_change': 0.05,
            'pattern_confidence': 0.8,
            'market_context': {
                'trend': 'bullish',
                'volatility': 'normal'
            }
        }
        
        strength = self.signal_strength.calculate_signal_strength(
            signal_type='breakout',
            signal_data=signal_data
        )
        
        assert 'overall_strength' in strength
        assert 'components' in strength
        assert 'category' in strength
        assert 0 <= strength['overall_strength'] <= 1
        assert strength['category'] in SignalCategory
        
    def test_multi_signal_aggregation(self):
        signals = [
            {
                'type': 'divergence',
                'strength': 0.8,
                'confidence': 0.9,
                'timestamp': datetime.now()
            },
            {
                'type': 'volume_spike',
                'strength': 0.7,
                'confidence': 0.8,
                'timestamp': datetime.now()
            },
            {
                'type': 'accumulation',
                'strength': 0.6,
                'confidence': 0.7,
                'timestamp': datetime.now()
            }
        ]
        
        aggregated = self.signal_strength.aggregate_signals(signals)
        
        assert 'combined_strength' in aggregated
        assert 'signal_count' in aggregated
        assert 'dominant_signal' in aggregated
        assert 'recommendation' in aggregated
        assert aggregated['signal_count'] == 3
        
    def test_signal_decay_over_time(self):
        old_signal = {
            'type': 'breakout',
            'strength': 0.9,
            'timestamp': datetime.now() - timedelta(hours=48)
        }
        
        recent_signal = {
            'type': 'breakout',
            'strength': 0.9,
            'timestamp': datetime.now() - timedelta(minutes=30)
        }
        
        old_strength = self.signal_strength.apply_time_decay(
            old_signal['strength'],
            old_signal['timestamp']
        )
        
        recent_strength = self.signal_strength.apply_time_decay(
            recent_signal['strength'],
            recent_signal['timestamp']
        )
        
        assert old_strength < recent_strength
        assert old_strength < old_signal['strength']
        
    def test_signal_category_classification(self):
        weak_signal = {'overall_strength': 0.3}
        moderate_signal = {'overall_strength': 0.6}
        strong_signal = {'overall_strength': 0.85}
        very_strong_signal = {'overall_strength': 0.95}
        
        assert self.signal_strength.classify_signal(weak_signal) == SignalCategory.WEAK
        assert self.signal_strength.classify_signal(moderate_signal) == SignalCategory.MODERATE
        assert self.signal_strength.classify_signal(strong_signal) == SignalCategory.STRONG
        assert self.signal_strength.classify_signal(very_strong_signal) == SignalCategory.VERY_STRONG
        
    def test_contextual_adjustment(self):
        base_signal = {
            'type': 'breakout',
            'strength': 0.7,
            'volume_ratio': 3.0
        }
        
        bull_market_strength = self.signal_strength.calculate_signal_strength(
            signal_type='breakout',
            signal_data=base_signal,
            market_context={'trend': 'strong_uptrend', 'volatility': 'low'}
        )
        
        bear_market_strength = self.signal_strength.calculate_signal_strength(
            signal_type='breakout',
            signal_data=base_signal,
            market_context={'trend': 'strong_downtrend', 'volatility': 'high'}
        )
        
        assert bull_market_strength['overall_strength'] > bear_market_strength['overall_strength']


class TestIntegration:
    def setup_method(self):
        self.divergence = PriceVolumeDivergence()
        self.patterns = UnusualVolumePatterns()
        self.ad_patterns = AccumulationDistributionPatterns()
        self.confirmation = VolumeConfirmation()
        self.profile = VolumeProfileAnalysis()
        self.signal_strength = VolumeSignalStrength()
        
        self.dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
    def test_complete_volume_analysis_workflow(self):
        np.random.seed(42)
        prices = np.array([100 + i*0.1 + np.sin(i/10)*5 for i in range(200)])
        volumes = np.array([1000000 + np.random.randn()*100000 + i*1000 for i in range(200)])
        
        divergences = self.divergence.detect_divergence(prices, volumes, self.dates)
        unusual = self.patterns.detect_unusual_patterns(prices, volumes, self.dates)
        ad_patterns = self.ad_patterns.detect_accumulation_distribution(prices, volumes, self.dates)
        
        all_signals = []
        
        for div in divergences:
            signal = {
                'type': 'divergence',
                'subtype': div['type'].value,
                'strength': div['confidence'],
                'timestamp': div['start_date'],
                'data': div
            }
            all_signals.append(signal)
            
        for pattern in unusual:
            signal = {
                'type': 'volume_pattern',
                'subtype': pattern['type'].value,
                'strength': pattern['confidence'],
                'timestamp': pattern['timestamp'],
                'data': pattern
            }
            all_signals.append(signal)
            
        if all_signals:
            aggregated = self.signal_strength.aggregate_signals(all_signals)
            assert aggregated['combined_strength'] > 0
            assert aggregated['signal_count'] == len(all_signals)
            
        profile = self.profile.create_volume_profile(prices, volumes, self.dates)
        assert profile['poc'] > 0
        assert profile['value_area_high'] > profile['value_area_low']
        
    def test_pattern_confirmation_workflow(self):
        prices = np.array([100]*50 + [105, 107, 109, 111] + [110]*46)
        volumes = np.array([1000000]*50 + [3000000, 2800000, 2600000, 2400000] + [1500000]*46)
        
        patterns = self.patterns.detect_unusual_patterns(prices, volumes, self.dates[:100])
        
        breakouts = [p for p in patterns if p['type'] == VolumePatternType.BREAKOUT]
        
        if breakouts:
            breakout = breakouts[0]
            confirmed = self.confirmation.confirm_price_pattern(
                pattern_type='breakout',
                pattern_data={
                    'breakout_index': 50,
                    'resistance_level': 100,
                    'breakout_price': 105
                },
                prices=prices,
                volumes=volumes,
                timestamps=self.dates[:100]
            )
            
            signal_data = {
                'volume_ratio': confirmed['volume_ratio'],
                'price_change': 0.11,
                'pattern_confidence': breakout['confidence']
            }
            
            strength = self.signal_strength.calculate_signal_strength(
                signal_type='breakout',
                signal_data=signal_data
            )
            
            assert strength['overall_strength'] > 0.6
            assert confirmed['is_confirmed'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])