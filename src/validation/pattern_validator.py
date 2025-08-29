from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import logging
from collections import defaultdict
import json

from src.data.models.pattern_models import (
    PatternType, PatternDirection, PatternStatus,
    PatternDetectionResult, Pattern
)
from src.core.confidence.confidence_framework import ConfidenceFramework

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    OUT_OF_SAMPLE = "out_of_sample"
    ROLLING_WINDOW = "rolling_window"
    WALK_FORWARD = "walk_forward"
    CROSS_VALIDATION = "cross_validation"
    MONTE_CARLO = "monte_carlo"


class DegradationStatus(Enum):
    STABLE = "stable"
    DEGRADING = "degrading"
    IMPROVING = "improving"
    UNSTABLE = "unstable"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ValidationResult:
    method: ValidationMethod
    total_patterns: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    avg_return: float
    validation_period: Tuple[datetime, datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confusion_matrix(self) -> np.ndarray:
        return np.array([
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives]
        ])
    
    @property
    def matthews_correlation(self) -> float:
        tp, fp, tn, fn = self.true_positives, self.false_positives, self.true_negatives, self.false_negatives
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0


@dataclass
class DegradationAnalysis:
    pattern_type: str
    pattern_name: str
    status: DegradationStatus
    current_performance: float
    historical_avg: float
    degradation_rate: float
    confidence_interval: Tuple[float, float]
    recent_failures: int
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    variant_a: str
    variant_b: str
    metric: str
    a_performance: float
    b_performance: float
    improvement: float
    p_value: float
    is_significant: bool
    sample_size_a: int
    sample_size_b: int
    confidence_level: float
    recommendation: str


class PatternValidator:
    
    def __init__(self, confidence_framework: Optional[ConfidenceFramework] = None):
        self.confidence_framework = confidence_framework or ConfidenceFramework()
        self.validation_cache: Dict[str, ValidationResult] = {}
        
    def validate_out_of_sample(self, 
                               train_data: pd.DataFrame,
                               test_data: pd.DataFrame,
                               pattern_detector: Callable,
                               threshold: float = 0.6) -> ValidationResult:
        
        train_patterns = pattern_detector(train_data)
        
        test_patterns = pattern_detector(test_data)
        test_outcomes = self._get_pattern_outcomes(test_data, test_patterns)
        
        tp = fp = tn = fn = 0
        returns = []
        
        for pattern in test_patterns:
            confidence = self._calculate_pattern_confidence(pattern, test_data)
            predicted = confidence > threshold
            actual = test_outcomes.get(pattern, {}).get('success', False)
            pattern_return = test_outcomes.get(pattern, {}).get('return', 0)
            
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn += 1
            
            if predicted:
                returns.append(pattern_return)
        
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        returns_array = np.array(returns) if returns else np.array([0])
        sharpe = self._calculate_sharpe_ratio(returns_array)
        max_dd = self._calculate_max_drawdown(returns_array)
        profit_factor = self._calculate_profit_factor(returns_array)
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        
        return ValidationResult(
            method=ValidationMethod.OUT_OF_SAMPLE,
            total_patterns=len(test_patterns),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_return=np.mean(returns_array),
            validation_period=(test_data.index[0], test_data.index[-1]),
            metadata={
                'train_size': len(train_data),
                'test_size': len(test_data),
                'threshold': threshold
            }
        )
    
    def rolling_window_backtest(self,
                               data: pd.DataFrame,
                               pattern_detector: Callable,
                               window_size: int = 100,
                               step_size: int = 20,
                               validation_size: int = 20) -> List[ValidationResult]:
        
        results = []
        
        for start_idx in range(0, len(data) - window_size - validation_size, step_size):
            train_end = start_idx + window_size
            test_end = train_end + validation_size
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            result = self.validate_out_of_sample(train_data, test_data, pattern_detector)
            results.append(result)
        
        return results
    
    def walk_forward_analysis(self,
                             data: pd.DataFrame,
                             pattern_detector: Callable,
                             initial_train_size: int = 500,
                             test_size: int = 100,
                             retrain_frequency: int = 50) -> List[ValidationResult]:
        
        results = []
        train_start = 0
        train_end = initial_train_size
        
        while train_end + test_size <= len(data):
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[train_end:train_end + test_size]
            
            result = self.validate_out_of_sample(train_data, test_data, pattern_detector)
            results.append(result)
            
            train_end += retrain_frequency
            if train_end - train_start > initial_train_size * 2:
                train_start += retrain_frequency
        
        return results
    
    def cross_validate(self,
                      data: pd.DataFrame,
                      pattern_detector: Callable,
                      n_splits: int = 5) -> List[ValidationResult]:
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            result = self.validate_out_of_sample(train_data, test_data, pattern_detector)
            results.append(result)
        
        return results
    
    def detect_pattern_degradation(self,
                                  historical_performance: List[Dict[str, Any]],
                                  lookback_periods: int = 10,
                                  significance_level: float = 0.05) -> DegradationAnalysis:
        
        if len(historical_performance) < lookback_periods:
            return DegradationAnalysis(
                pattern_type="unknown",
                pattern_name="unknown",
                status=DegradationStatus.INSUFFICIENT_DATA,
                current_performance=0,
                historical_avg=0,
                degradation_rate=0,
                confidence_interval=(0, 0),
                recent_failures=0,
                recommendation="Insufficient data for analysis"
            )
        
        recent = historical_performance[-lookback_periods:]
        older = historical_performance[:-lookback_periods]
        
        recent_success_rate = sum(p['success'] for p in recent) / len(recent)
        older_success_rate = sum(p['success'] for p in older) / len(older) if older else recent_success_rate
        
        recent_returns = [p['return'] for p in recent]
        older_returns = [p['return'] for p in older] if older else recent_returns
        
        t_stat, p_value = stats.ttest_ind(recent_returns, older_returns)
        
        degradation_rate = (recent_success_rate - older_success_rate) / older_success_rate if older_success_rate > 0 else 0
        
        ci_lower, ci_upper = stats.t.interval(
            1 - significance_level,
            len(recent_returns) - 1,
            loc=np.mean(recent_returns),
            scale=stats.sem(recent_returns)
        )
        
        recent_failures = sum(1 for p in recent if not p['success'])
        
        if degradation_rate < -0.2 and p_value < significance_level:
            status = DegradationStatus.DEGRADING
            recommendation = "Pattern showing significant degradation. Consider reducing position size or avoiding."
        elif degradation_rate > 0.2 and p_value < significance_level:
            status = DegradationStatus.IMPROVING
            recommendation = "Pattern performance improving. Consider increasing confidence weight."
        elif abs(degradation_rate) > 0.3:
            status = DegradationStatus.UNSTABLE
            recommendation = "Pattern showing high variability. Use with caution."
        else:
            status = DegradationStatus.STABLE
            recommendation = "Pattern performance stable. Continue normal usage."
        
        return DegradationAnalysis(
            pattern_type=historical_performance[0].get('pattern_type', 'unknown'),
            pattern_name=historical_performance[0].get('pattern_name', 'unknown'),
            status=status,
            current_performance=recent_success_rate,
            historical_avg=older_success_rate,
            degradation_rate=degradation_rate,
            confidence_interval=(ci_lower, ci_upper),
            recent_failures=recent_failures,
            recommendation=recommendation,
            metadata={
                'p_value': p_value,
                't_statistic': t_stat,
                'sample_size': len(recent)
            }
        )
    
    def ab_test_patterns(self,
                        variant_a_results: List[Dict[str, Any]],
                        variant_b_results: List[Dict[str, Any]],
                        metric: str = 'return',
                        confidence_level: float = 0.95) -> ABTestResult:
        
        a_values = [r[metric] for r in variant_a_results if metric in r]
        b_values = [r[metric] for r in variant_b_results if metric in r]
        
        if not a_values or not b_values:
            return ABTestResult(
                variant_a="A",
                variant_b="B",
                metric=metric,
                a_performance=0,
                b_performance=0,
                improvement=0,
                p_value=1.0,
                is_significant=False,
                sample_size_a=len(a_values),
                sample_size_b=len(b_values),
                confidence_level=confidence_level,
                recommendation="Insufficient data for comparison"
            )
        
        a_mean = np.mean(a_values)
        b_mean = np.mean(b_values)
        
        t_stat, p_value = stats.ttest_ind(a_values, b_values)
        
        improvement = (b_mean - a_mean) / abs(a_mean) if a_mean != 0 else 0
        is_significant = p_value < (1 - confidence_level)
        
        if is_significant and improvement > 0.1:
            recommendation = f"Variant B shows significant improvement ({improvement:.1%}). Consider adopting."
        elif is_significant and improvement < -0.1:
            recommendation = f"Variant A performs better ({-improvement:.1%}). Keep current approach."
        else:
            recommendation = "No significant difference detected. Continue testing."
        
        return ABTestResult(
            variant_a="A",
            variant_b="B",
            metric=metric,
            a_performance=a_mean,
            b_performance=b_mean,
            improvement=improvement,
            p_value=p_value,
            is_significant=is_significant,
            sample_size_a=len(a_values),
            sample_size_b=len(b_values),
            confidence_level=confidence_level,
            recommendation=recommendation
        )
    
    def analyze_false_positives_negatives(self,
                                         predictions: List[Dict[str, Any]],
                                         actuals: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        false_positives = []
        false_negatives = []
        
        for pred, actual in zip(predictions, actuals):
            pred_signal = pred.get('signal', False)
            actual_success = actual.get('success', False)
            
            if pred_signal and not actual_success:
                false_positives.append({
                    'pattern': pred.get('pattern_name'),
                    'confidence': pred.get('confidence'),
                    'timestamp': pred.get('timestamp'),
                    'metadata': pred.get('metadata', {})
                })
            elif not pred_signal and actual_success:
                false_negatives.append({
                    'pattern': pred.get('pattern_name'),
                    'confidence': pred.get('confidence'),
                    'timestamp': pred.get('timestamp'),
                    'metadata': pred.get('metadata', {})
                })
        
        fp_patterns = defaultdict(int)
        fn_patterns = defaultdict(int)
        
        for fp in false_positives:
            fp_patterns[fp['pattern']] += 1
        
        for fn in false_negatives:
            fn_patterns[fn['pattern']] += 1
        
        fp_confidence_dist = [fp['confidence'] for fp in false_positives]
        fn_confidence_dist = [fn['confidence'] for fn in false_negatives]
        
        analysis = {
            'total_false_positives': len(false_positives),
            'total_false_negatives': len(false_negatives),
            'false_positive_rate': len(false_positives) / len(predictions) if predictions else 0,
            'false_negative_rate': len(false_negatives) / len(predictions) if predictions else 0,
            'most_common_fp_patterns': dict(sorted(fp_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            'most_common_fn_patterns': dict(sorted(fn_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            'avg_fp_confidence': np.mean(fp_confidence_dist) if fp_confidence_dist else 0,
            'avg_fn_confidence': np.mean(fn_confidence_dist) if fn_confidence_dist else 0,
            'recommendations': self._generate_fp_fn_recommendations(false_positives, false_negatives)
        }
        
        return analysis
    
    def generate_performance_report(self,
                                   validation_results: List[ValidationResult],
                                   pattern_type: Optional[str] = None) -> Dict[str, Any]:
        
        if not validation_results:
            return {'error': 'No validation results provided'}
        
        aggregated = {
            'total_validations': len(validation_results),
            'pattern_type': pattern_type,
            'avg_accuracy': np.mean([r.accuracy for r in validation_results]),
            'avg_precision': np.mean([r.precision for r in validation_results]),
            'avg_recall': np.mean([r.recall for r in validation_results]),
            'avg_f1_score': np.mean([r.f1_score for r in validation_results]),
            'avg_sharpe_ratio': np.mean([r.sharpe_ratio for r in validation_results]),
            'avg_max_drawdown': np.mean([r.max_drawdown for r in validation_results]),
            'avg_profit_factor': np.mean([r.profit_factor for r in validation_results]),
            'avg_win_rate': np.mean([r.win_rate for r in validation_results]),
            'avg_return': np.mean([r.avg_return for r in validation_results]),
            'best_validation': self._get_best_validation(validation_results),
            'worst_validation': self._get_worst_validation(validation_results),
            'stability_score': self._calculate_stability_score(validation_results),
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        return aggregated
    
    def _calculate_pattern_confidence(self, pattern: Dict[str, Any], market_data: pd.DataFrame) -> float:
        pattern_data = {
            'pattern_type': pattern.get('type'),
            'pattern_name': pattern.get('name'),
            'direction': pattern.get('direction'),
            'historical_success_rate': pattern.get('success_rate', 0.5)
        }
        
        market_dict = {
            'volume': market_data['volume'].iloc[-1] if 'volume' in market_data else None,
            'avg_volume': market_data['volume'].mean() if 'volume' in market_data else None
        }
        
        confidence_score = self.confidence_framework.calculate_confidence(
            pattern_data, market_dict, market_data
        )
        
        return confidence_score.overall_score
    
    def _get_pattern_outcomes(self, data: pd.DataFrame, patterns: List[Dict[str, Any]]) -> Dict:
        outcomes = {}
        
        for pattern in patterns:
            entry_idx = pattern.get('entry_index')
            exit_idx = pattern.get('exit_index', min(entry_idx + 20, len(data) - 1))
            
            if entry_idx and exit_idx and entry_idx < len(data) and exit_idx < len(data):
                entry_price = data.iloc[entry_idx]['close']
                exit_price = data.iloc[exit_idx]['close']
                
                returns = (exit_price - entry_price) / entry_price
                success = returns > 0 if pattern.get('direction') == 'bullish' else returns < 0
                
                outcomes[pattern] = {
                    'success': success,
                    'return': returns
                }
        
        return outcomes
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        if np.std(excess_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses > 0 else float('inf') if gains > 0 else 0
    
    def _get_best_validation(self, results: List[ValidationResult]) -> Dict[str, Any]:
        if not results:
            return {}
        best = max(results, key=lambda r: r.f1_score)
        return {
            'f1_score': best.f1_score,
            'accuracy': best.accuracy,
            'sharpe_ratio': best.sharpe_ratio,
            'period': best.validation_period
        }
    
    def _get_worst_validation(self, results: List[ValidationResult]) -> Dict[str, Any]:
        if not results:
            return {}
        worst = min(results, key=lambda r: r.f1_score)
        return {
            'f1_score': worst.f1_score,
            'accuracy': worst.accuracy,
            'sharpe_ratio': worst.sharpe_ratio,
            'period': worst.validation_period
        }
    
    def _calculate_stability_score(self, results: List[ValidationResult]) -> float:
        if len(results) < 2:
            return 0
        
        f1_scores = [r.f1_score for r in results]
        std_dev = np.std(f1_scores)
        mean_score = np.mean(f1_scores)
        
        cv = std_dev / mean_score if mean_score > 0 else float('inf')
        
        if cv < 0.1:
            return 0.9
        elif cv < 0.2:
            return 0.7
        elif cv < 0.3:
            return 0.5
        else:
            return 0.3
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        recommendations = []
        
        avg_f1 = np.mean([r.f1_score for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_drawdown = np.mean([r.max_drawdown for r in results])
        
        if avg_f1 < 0.5:
            recommendations.append("Pattern shows poor predictive power. Consider refinement or removal.")
        elif avg_f1 > 0.7:
            recommendations.append("Pattern shows strong predictive power. Consider increasing position sizing.")
        
        if avg_sharpe < 0.5:
            recommendations.append("Risk-adjusted returns are poor. Review entry/exit criteria.")
        elif avg_sharpe > 1.5:
            recommendations.append("Excellent risk-adjusted returns. Pattern is performing well.")
        
        if avg_drawdown < -0.2:
            recommendations.append("High drawdown risk. Implement stricter risk management.")
        
        return recommendations
    
    def _generate_fp_fn_recommendations(self, false_positives: List[Dict], 
                                       false_negatives: List[Dict]) -> List[str]:
        recommendations = []
        
        if len(false_positives) > len(false_negatives) * 2:
            recommendations.append("High false positive rate. Consider raising confidence threshold.")
        
        if len(false_negatives) > len(false_positives) * 2:
            recommendations.append("High false negative rate. Consider lowering confidence threshold.")
        
        if false_positives:
            avg_fp_conf = np.mean([fp['confidence'] for fp in false_positives])
            if avg_fp_conf > 0.7:
                recommendations.append("False positives occurring at high confidence. Review pattern criteria.")
        
        return recommendations