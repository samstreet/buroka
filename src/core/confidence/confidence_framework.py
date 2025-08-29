from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ConfidenceComponent(Enum):
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    HISTORICAL_SUCCESS = "historical_success"
    VOLUME_CONFIRMATION = "volume_confirmation"
    TIMEFRAME_ALIGNMENT = "timeframe_alignment"
    MARKET_CONDITION = "market_condition"
    PATTERN_QUALITY = "pattern_quality"
    CORRELATION_STRENGTH = "correlation_strength"
    MOMENTUM_ALIGNMENT = "momentum_alignment"


class MarketCondition(Enum):
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class ConfidenceScore:
    overall_score: float
    components: Dict[ConfidenceComponent, float]
    weights: Dict[ConfidenceComponent, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def confidence_level(self) -> str:
        if self.overall_score >= 0.9:
            return "VERY_HIGH"
        elif self.overall_score >= 0.75:
            return "HIGH"
        elif self.overall_score >= 0.6:
            return "MEDIUM"
        elif self.overall_score >= 0.4:
            return "LOW"
        else:
            return "VERY_LOW"
    
    @property
    def recommendation(self) -> str:
        if self.overall_score >= 0.8:
            return "STRONG_SIGNAL"
        elif self.overall_score >= 0.65:
            return "MODERATE_SIGNAL"
        elif self.overall_score >= 0.5:
            return "WEAK_SIGNAL"
        else:
            return "NO_ACTION"
    
    def get_component_contribution(self, component: ConfidenceComponent) -> float:
        if component in self.components and component in self.weights:
            return self.components[component] * self.weights[component]
        return 0.0


class ConfidenceFramework:
    
    DEFAULT_WEIGHTS = {
        ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.30,
        ConfidenceComponent.HISTORICAL_SUCCESS: 0.25,
        ConfidenceComponent.VOLUME_CONFIRMATION: 0.20,
        ConfidenceComponent.TIMEFRAME_ALIGNMENT: 0.15,
        ConfidenceComponent.MARKET_CONDITION: 0.10
    }
    
    def __init__(self, custom_weights: Optional[Dict[ConfidenceComponent, float]] = None):
        self.weights = custom_weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
        
    def _validate_weights(self):
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            for key in self.weights:
                self.weights[key] /= total_weight
                
    def calculate_confidence(self, 
                            pattern_data: Dict[str, Any],
                            market_data: Optional[Dict[str, Any]] = None,
                            historical_data: Optional[pd.DataFrame] = None) -> ConfidenceScore:
        
        components = {}
        
        components[ConfidenceComponent.STATISTICAL_SIGNIFICANCE] = self._calculate_statistical_significance(
            pattern_data, historical_data
        )
        
        components[ConfidenceComponent.HISTORICAL_SUCCESS] = self._calculate_historical_success(
            pattern_data
        )
        
        components[ConfidenceComponent.VOLUME_CONFIRMATION] = self._calculate_volume_confirmation(
            pattern_data, market_data
        )
        
        components[ConfidenceComponent.TIMEFRAME_ALIGNMENT] = self._calculate_timeframe_alignment(
            pattern_data, market_data
        )
        
        components[ConfidenceComponent.MARKET_CONDITION] = self._calculate_market_condition_adjustment(
            pattern_data, market_data
        )
        
        overall_score = self._calculate_overall_score(components)
        
        return ConfidenceScore(
            overall_score=overall_score,
            components=components,
            weights=self.weights.copy(),
            metadata={
                'pattern_type': pattern_data.get('pattern_type'),
                'pattern_name': pattern_data.get('pattern_name'),
                'market_condition': self._identify_market_condition(market_data)
            }
        )
    
    def _calculate_statistical_significance(self, pattern_data: Dict[str, Any], 
                                          historical_data: Optional[pd.DataFrame]) -> float:
        score = 0.5
        
        if 'price_points' in pattern_data and historical_data is not None:
            prices = pattern_data['price_points']
            if len(prices) >= 3:
                returns = np.diff(prices) / prices[:-1]
                
                if len(historical_data) > 0:
                    historical_returns = historical_data['close'].pct_change().dropna()
                    
                    if len(historical_returns) > 30:
                        t_stat, p_value = stats.ttest_ind(returns, historical_returns)
                        
                        if p_value < 0.01:
                            score = 0.95
                        elif p_value < 0.05:
                            score = 0.85
                        elif p_value < 0.10:
                            score = 0.75
                        else:
                            score = 0.5 + (0.1 - p_value) * 2
                
                pattern_std = np.std(returns)
                if pattern_std > 0:
                    sharpe_ratio = np.mean(returns) / pattern_std
                    score *= min(1.0, 0.7 + sharpe_ratio * 0.15)
        
        if 'pattern_points' in pattern_data:
            points = pattern_data['pattern_points']
            if len(points) >= 5:
                score *= 1.1
            elif len(points) >= 3:
                score *= 1.05
        
        if 'r_squared' in pattern_data:
            r_squared = pattern_data['r_squared']
            if r_squared > 0.9:
                score *= 1.15
            elif r_squared > 0.7:
                score *= 1.05
        
        return min(1.0, score)
    
    def _calculate_historical_success(self, pattern_data: Dict[str, Any]) -> float:
        score = 0.5
        
        if 'historical_success_rate' in pattern_data:
            success_rate = pattern_data['historical_success_rate']
            score = success_rate
            
            if 'sample_size' in pattern_data:
                sample_size = pattern_data['sample_size']
                if sample_size < 10:
                    score *= 0.7
                elif sample_size < 30:
                    score *= 0.85
                elif sample_size > 100:
                    score *= 1.1
        
        if 'recent_performance' in pattern_data:
            recent = pattern_data['recent_performance']
            if recent > pattern_data.get('historical_success_rate', 0.5):
                score = score * 0.7 + recent * 0.3
        
        if 'avg_return' in pattern_data:
            avg_return = pattern_data['avg_return']
            if avg_return > 0.05:
                score *= 1.2
            elif avg_return > 0.02:
                score *= 1.1
            elif avg_return < -0.02:
                score *= 0.8
        
        if 'win_loss_ratio' in pattern_data:
            ratio = pattern_data['win_loss_ratio']
            if ratio > 2.0:
                score *= 1.15
            elif ratio > 1.5:
                score *= 1.05
            elif ratio < 1.0:
                score *= 0.9
        
        return min(1.0, score)
    
    def _calculate_volume_confirmation(self, pattern_data: Dict[str, Any],
                                      market_data: Optional[Dict[str, Any]]) -> float:
        score = 0.5
        
        if market_data and 'volume' in market_data:
            current_volume = market_data['volume']
            avg_volume = market_data.get('avg_volume', current_volume)
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            pattern_type = pattern_data.get('pattern_type', '').lower()
            
            if 'breakout' in pattern_type or 'reversal' in pattern_type:
                if volume_ratio > 2.0:
                    score = 0.95
                elif volume_ratio > 1.5:
                    score = 0.85
                elif volume_ratio > 1.2:
                    score = 0.75
                else:
                    score = 0.4 + volume_ratio * 0.2
            else:
                if 0.8 <= volume_ratio <= 1.5:
                    score = 0.8
                elif volume_ratio > 1.5:
                    score = 0.7
                else:
                    score = 0.5
        
        if 'volume_trend' in pattern_data:
            trend = pattern_data['volume_trend']
            if trend == 'increasing' and pattern_data.get('direction') == 'bullish':
                score *= 1.15
            elif trend == 'decreasing' and pattern_data.get('direction') == 'bearish':
                score *= 1.1
            elif trend == 'diverging':
                score *= 0.85
        
        if 'volume_profile' in market_data:
            profile = market_data['volume_profile']
            if 'at_poc' in profile and profile['at_poc']:
                score *= 1.1
            if 'above_vwap' in profile and profile['above_vwap']:
                score *= 1.05
        
        return min(1.0, score)
    
    def _calculate_timeframe_alignment(self, pattern_data: Dict[str, Any],
                                      market_data: Optional[Dict[str, Any]]) -> float:
        score = 0.5
        
        if market_data and 'timeframe_signals' in market_data:
            signals = market_data['timeframe_signals']
            aligned_count = 0
            total_count = len(signals)
            
            pattern_direction = pattern_data.get('direction', 'neutral')
            
            for tf, signal in signals.items():
                if signal.get('direction') == pattern_direction:
                    aligned_count += 1
                    if tf in ['1d', '4h']:
                        aligned_count += 0.5
            
            if total_count > 0:
                alignment_ratio = aligned_count / total_count
                score = 0.3 + alignment_ratio * 0.7
        
        if 'higher_timeframe_trend' in pattern_data:
            ht_trend = pattern_data['higher_timeframe_trend']
            pattern_direction = pattern_data.get('direction', 'neutral')
            
            if ht_trend == pattern_direction:
                score *= 1.2
            elif ht_trend == 'neutral':
                score *= 1.0
            else:
                score *= 0.7
        
        if 'confluence_zones' in pattern_data:
            zones = pattern_data['confluence_zones']
            if len(zones) >= 3:
                score *= 1.15
            elif len(zones) >= 2:
                score *= 1.08
        
        return min(1.0, score)
    
    def _calculate_market_condition_adjustment(self, pattern_data: Dict[str, Any],
                                              market_data: Optional[Dict[str, Any]]) -> float:
        score = 0.5
        
        if market_data:
            condition = self._identify_market_condition(market_data)
            pattern_type = pattern_data.get('pattern_type', '').lower()
            
            adjustments = self._get_condition_adjustments(condition, pattern_type)
            score = adjustments.get('base_score', 0.5)
            
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                if volatility > 0.3:
                    if 'breakout' in pattern_type:
                        score *= 0.85
                    else:
                        score *= 0.9
                elif volatility < 0.1:
                    if 'breakout' in pattern_type:
                        score *= 1.1
                    else:
                        score *= 1.05
            
            if 'trend_strength' in market_data:
                strength = market_data['trend_strength']
                if strength > 0.7:
                    if pattern_data.get('direction') == market_data.get('trend_direction'):
                        score *= 1.15
                    else:
                        score *= 0.85
        
        if 'market_phase' in pattern_data:
            phase = pattern_data['market_phase']
            if phase in ['accumulation', 'markup']:
                if pattern_data.get('direction') == 'bullish':
                    score *= 1.1
            elif phase in ['distribution', 'markdown']:
                if pattern_data.get('direction') == 'bearish':
                    score *= 1.1
        
        return min(1.0, score)
    
    def _identify_market_condition(self, market_data: Optional[Dict[str, Any]]) -> MarketCondition:
        if not market_data:
            return MarketCondition.RANGING
        
        trend_strength = market_data.get('trend_strength', 0)
        volatility = market_data.get('volatility', 0.15)
        
        if trend_strength > 0.7:
            return MarketCondition.STRONG_TREND
        elif trend_strength > 0.4:
            return MarketCondition.WEAK_TREND
        elif volatility > 0.25:
            return MarketCondition.VOLATILE
        elif volatility < 0.1:
            return MarketCondition.CALM
        elif market_data.get('is_breakout', False):
            return MarketCondition.BREAKOUT
        elif market_data.get('is_reversal', False):
            return MarketCondition.REVERSAL
        else:
            return MarketCondition.RANGING
    
    def _get_condition_adjustments(self, condition: MarketCondition, 
                                  pattern_type: str) -> Dict[str, float]:
        adjustments = {
            MarketCondition.STRONG_TREND: {
                'base_score': 0.7,
                'continuation': 1.2,
                'reversal': 0.7
            },
            MarketCondition.WEAK_TREND: {
                'base_score': 0.6,
                'continuation': 1.0,
                'reversal': 0.9
            },
            MarketCondition.RANGING: {
                'base_score': 0.5,
                'breakout': 1.1,
                'reversal': 1.05
            },
            MarketCondition.VOLATILE: {
                'base_score': 0.4,
                'all': 0.85
            },
            MarketCondition.CALM: {
                'base_score': 0.6,
                'breakout': 1.15
            },
            MarketCondition.BREAKOUT: {
                'base_score': 0.75,
                'continuation': 1.1
            },
            MarketCondition.REVERSAL: {
                'base_score': 0.65,
                'reversal': 1.2
            }
        }
        
        return adjustments.get(condition, {'base_score': 0.5})
    
    def _calculate_overall_score(self, components: Dict[ConfidenceComponent, float]) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, score in components.items():
            if component in self.weights:
                weight = self.weights[component]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight > 0:
            base_score = weighted_sum / total_weight
        else:
            base_score = 0.5
        
        consistency_bonus = self._calculate_consistency_bonus(components)
        base_score *= (1 + consistency_bonus)
        
        return min(1.0, base_score)
    
    def _calculate_consistency_bonus(self, components: Dict[ConfidenceComponent, float]) -> float:
        scores = list(components.values())
        if len(scores) < 2:
            return 0.0
        
        std_dev = np.std(scores)
        mean_score = np.mean(scores)
        
        if mean_score > 0.7 and std_dev < 0.15:
            return 0.1
        elif mean_score > 0.6 and std_dev < 0.2:
            return 0.05
        elif std_dev > 0.3:
            return -0.05
        
        return 0.0
    
    def update_weights(self, performance_data: Dict[str, Any]):
        if 'component_performance' not in performance_data:
            return
        
        component_perf = performance_data['component_performance']
        
        total_success = sum(component_perf.values())
        if total_success > 0:
            for component in ConfidenceComponent:
                if component in component_perf:
                    success_rate = component_perf[component] / total_success
                    current_weight = self.weights.get(component, 0)
                    
                    adjustment = (success_rate - current_weight) * 0.1
                    new_weight = current_weight + adjustment
                    
                    self.weights[component] = max(0.05, min(0.5, new_weight))
        
        self._validate_weights()
    
    def get_confidence_explanation(self, confidence_score: ConfidenceScore) -> Dict[str, Any]:
        explanation = {
            'overall_confidence': confidence_score.overall_score,
            'confidence_level': confidence_score.confidence_level,
            'recommendation': confidence_score.recommendation,
            'component_breakdown': [],
            'key_factors': [],
            'warnings': []
        }
        
        for component, score in confidence_score.components.items():
            weight = confidence_score.weights.get(component, 0)
            contribution = score * weight
            
            explanation['component_breakdown'].append({
                'component': component.value,
                'score': score,
                'weight': weight,
                'contribution': contribution,
                'percentage': contribution / confidence_score.overall_score * 100 if confidence_score.overall_score > 0 else 0
            })
        
        sorted_components = sorted(
            explanation['component_breakdown'],
            key=lambda x: x['contribution'],
            reverse=True
        )
        
        for comp in sorted_components[:3]:
            if comp['score'] > 0.7:
                explanation['key_factors'].append(f"Strong {comp['component']}: {comp['score']:.2f}")
        
        for comp in sorted_components[-2:]:
            if comp['score'] < 0.4:
                explanation['warnings'].append(f"Weak {comp['component']}: {comp['score']:.2f}")
        
        return explanation


class AdaptiveConfidenceFramework(ConfidenceFramework):
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.performance_history: List[Dict[str, Any]] = []
        self.weight_history: List[Dict[ConfidenceComponent, float]] = []
        
    def learn_from_outcome(self, confidence_score: ConfidenceScore, 
                          actual_outcome: bool, actual_return: float):
        
        performance_record = {
            'timestamp': datetime.utcnow(),
            'confidence_score': confidence_score.overall_score,
            'predicted_success': confidence_score.overall_score > 0.6,
            'actual_success': actual_outcome,
            'actual_return': actual_return,
            'components': confidence_score.components.copy()
        }
        
        self.performance_history.append(performance_record)
        
        error = float(actual_outcome) - confidence_score.overall_score
        
        for component, score in confidence_score.components.items():
            if component in self.weights:
                gradient = error * score * self.learning_rate
                self.weights[component] += gradient
        
        self._validate_weights()
        self.weight_history.append(self.weights.copy())
        
        if len(self.performance_history) % 100 == 0:
            self._evaluate_performance()
    
    def _evaluate_performance(self):
        if len(self.performance_history) < 50:
            return
        
        recent = self.performance_history[-50:]
        
        component_success = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for record in recent:
            for component, score in record['components'].items():
                component_success[component]['total'] += 1
                if (score > 0.6) == record['actual_success']:
                    component_success[component]['correct'] += 1
        
        for component, stats in component_success.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                
                if accuracy > 0.7:
                    self.weights[component] *= 1.05
                elif accuracy < 0.5:
                    self.weights[component] *= 0.95
        
        self._validate_weights()
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        
        metrics = {
            'total_predictions': len(self.performance_history),
            'accuracy': (df['predicted_success'] == df['actual_success']).mean(),
            'avg_confidence': df['confidence_score'].mean(),
            'avg_return': df['actual_return'].mean(),
            'confidence_correlation': df['confidence_score'].corr(df['actual_return']),
            'weight_evolution': self.weight_history[-1] if self.weight_history else self.weights
        }
        
        if len(self.performance_history) > 100:
            early = df.iloc[:50]
            recent = df.iloc[-50:]
            
            metrics['improvement'] = {
                'accuracy': recent['predicted_success'].eq(recent['actual_success']).mean() - 
                           early['predicted_success'].eq(early['actual_success']).mean(),
                'correlation': recent['confidence_score'].corr(recent['actual_return']) -
                              early['confidence_score'].corr(early['actual_return'])
            }
        
        return metrics