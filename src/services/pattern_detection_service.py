import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import redis
from enum import Enum

from src.data.models.pattern_models import (
    PatternType, PatternDirection, PatternStatus, Timeframe,
    PatternDetectionResult, PatternFilter, Pattern
)
from src.data.storage.pattern_repository import PatternRepository
from src.core.patterns.chart_patterns import CandlestickPatterns, BreakoutPatterns, ClassicPatterns
from src.core.analysis.trend_analysis import TrendlineDetector, SupportResistanceIdentifier
from src.core.analysis.volume_price_analysis import PriceVolumeDivergence, UnusualVolumePatterns

logger = logging.getLogger(__name__)


class DetectorPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class NotificationChannel(Enum):
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DATABASE = "database"


@dataclass
class PatternNotification:
    pattern: PatternDetectionResult
    channels: List[NotificationChannel]
    priority: DetectorPriority
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PatternDetector(ABC):
    
    def __init__(self, name: str, pattern_type: PatternType, priority: DetectorPriority = DetectorPriority.MEDIUM):
        self.name = name
        self.pattern_type = pattern_type
        self.priority = priority
        
    @abstractmethod
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        return all(col in data.columns for col in required_columns)


class CandlestickDetector(PatternDetector):
    
    def __init__(self):
        super().__init__("CandlestickDetector", PatternType.CANDLESTICK, DetectorPriority.HIGH)
        self.candlestick_patterns = CandlestickPatterns()
        
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        if not self.validate_data(data):
            return []
            
        results = []
        
        patterns = await asyncio.to_thread(
            self.candlestick_patterns.detect_all_patterns,
            data['open'].values,
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['timestamp'].values
        )
        
        for pattern in patterns:
            direction = self._determine_direction(pattern['pattern'])
            
            result = PatternDetectionResult(
                pattern_type=self.pattern_type,
                pattern_name=pattern['pattern'],
                direction=direction,
                confidence=pattern['confidence'],
                timeframe=timeframe,
                start_time=pattern['timestamp'],
                end_time=pattern.get('end_time', pattern['timestamp']),
                symbol=symbol,
                entry_price=pattern.get('entry_price'),
                target_price=pattern.get('target_price'),
                stop_loss=pattern.get('stop_loss'),
                strength=pattern.get('strength'),
                metadata=pattern
            )
            results.append(result)
            
        return results
    
    def _determine_direction(self, pattern_name: str) -> PatternDirection:
        bullish_patterns = ['hammer', 'inverted_hammer', 'bullish_engulfing', 'morning_star', 
                          'three_white_soldiers', 'bullish_harami', 'tweezer_bottom']
        bearish_patterns = ['shooting_star', 'hanging_man', 'bearish_engulfing', 'evening_star',
                          'three_black_crows', 'bearish_harami', 'tweezer_top']
        
        if pattern_name.lower() in bullish_patterns:
            return PatternDirection.BULLISH
        elif pattern_name.lower() in bearish_patterns:
            return PatternDirection.BEARISH
        else:
            return PatternDirection.NEUTRAL


class ChartPatternDetector(PatternDetector):
    
    def __init__(self):
        super().__init__("ChartPatternDetector", PatternType.CHART, DetectorPriority.HIGH)
        self.classic_patterns = ClassicPatterns()
        self.breakout_patterns = BreakoutPatterns()
        
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        if not self.validate_data(data):
            return []
            
        results = []
        
        classic = await asyncio.to_thread(
            self.classic_patterns.detect_all_patterns,
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['timestamp'].values
        )
        
        breakouts = await asyncio.to_thread(
            self.breakout_patterns.detect_all_patterns,
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values,
            data['timestamp'].values
        )
        
        all_patterns = classic + breakouts
        
        for pattern in all_patterns:
            result = PatternDetectionResult(
                pattern_type=self.pattern_type,
                pattern_name=pattern['pattern'],
                direction=PatternDirection(pattern.get('direction', 'neutral')),
                confidence=pattern['confidence'],
                timeframe=timeframe,
                start_time=pattern['start_time'],
                end_time=pattern.get('end_time'),
                symbol=symbol,
                entry_price=pattern.get('entry_price'),
                target_price=pattern.get('target_price'),
                stop_loss=pattern.get('stop_loss'),
                strength=pattern.get('strength'),
                metadata=pattern
            )
            results.append(result)
            
        return results


class VolumePatternDetector(PatternDetector):
    
    def __init__(self):
        super().__init__("VolumePatternDetector", PatternType.VOLUME, DetectorPriority.MEDIUM)
        self.divergence_detector = PriceVolumeDivergence()
        self.unusual_patterns = UnusualVolumePatterns()
        
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        if not self.validate_data(data):
            return []
            
        results = []
        
        divergences = await asyncio.to_thread(
            self.divergence_detector.detect_divergence,
            data['close'].values,
            data['volume'].values,
            data['timestamp'].values
        )
        
        unusual = await asyncio.to_thread(
            self.unusual_patterns.detect_unusual_patterns,
            data['close'].values,
            data['volume'].values,
            data['timestamp'].values
        )
        
        for div in divergences:
            result = PatternDetectionResult(
                pattern_type=self.pattern_type,
                pattern_name=f"Volume {div['type'].value} Divergence",
                direction=self._divergence_to_direction(div['type'].value),
                confidence=div['confidence'],
                timeframe=timeframe,
                start_time=div['start_date'],
                end_time=div['end_date'],
                symbol=symbol,
                strength=div.get('strength'),
                metadata=div
            )
            results.append(result)
            
        for pattern in unusual:
            result = PatternDetectionResult(
                pattern_type=self.pattern_type,
                pattern_name=f"Volume {pattern['type'].value}",
                direction=self._volume_pattern_to_direction(pattern['type'].value),
                confidence=pattern['confidence'],
                timeframe=timeframe,
                start_time=pattern['timestamp'],
                symbol=symbol,
                metadata=pattern
            )
            results.append(result)
            
        return results
    
    def _divergence_to_direction(self, divergence_type: str) -> PatternDirection:
        if 'bullish' in divergence_type.lower():
            return PatternDirection.BULLISH
        elif 'bearish' in divergence_type.lower():
            return PatternDirection.BEARISH
        return PatternDirection.NEUTRAL
    
    def _volume_pattern_to_direction(self, pattern_type: str) -> PatternDirection:
        bullish_patterns = ['breakout', 'accumulation', 'pocket_pivot']
        bearish_patterns = ['distribution', 'climax']
        
        if any(p in pattern_type.lower() for p in bullish_patterns):
            return PatternDirection.BULLISH
        elif any(p in pattern_type.lower() for p in bearish_patterns):
            return PatternDirection.BEARISH
        return PatternDirection.NEUTRAL


class PatternDetectionService:
    
    def __init__(self, repository: PatternRepository, redis_client: Optional[redis.Redis] = None):
        self.repository = repository
        self.redis_client = redis_client
        self.detectors: Dict[str, PatternDetector] = {}
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        self.scanning_tasks: Dict[str, asyncio.Task] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.cache_ttl = 300
        
        self._initialize_default_detectors()
        
    def _initialize_default_detectors(self):
        self.register_detector(CandlestickDetector())
        self.register_detector(ChartPatternDetector())
        self.register_detector(VolumePatternDetector())
        
    def register_detector(self, detector: PatternDetector):
        self.detectors[detector.name] = detector
        logger.info(f"Registered detector: {detector.name}")
        
    def unregister_detector(self, detector_name: str):
        if detector_name in self.detectors:
            del self.detectors[detector_name]
            logger.info(f"Unregistered detector: {detector_name}")
            
    def register_notification_handler(self, channel: NotificationChannel, handler: Callable):
        self.notification_handlers[channel] = handler
        logger.info(f"Registered notification handler for channel: {channel.value}")
        
    async def scan_for_patterns(self, data: pd.DataFrame, symbol: str, 
                               timeframe: Timeframe, detectors: Optional[List[str]] = None) -> List[PatternDetectionResult]:
        cache_key = self._get_cache_key(symbol, timeframe, data.index[-1] if len(data) > 0 else None)
        
        if self.redis_client and cache_key:
            cached = await self._get_cached_patterns(cache_key)
            if cached:
                return cached
                
        active_detectors = detectors if detectors else list(self.detectors.keys())
        all_results = []
        
        detection_tasks = []
        for detector_name in active_detectors:
            if detector_name in self.detectors:
                detector = self.detectors[detector_name]
                task = detector.detect(data, symbol, timeframe)
                detection_tasks.append((detector.priority, task))
                
        detection_tasks.sort(key=lambda x: x[0].value)
        
        for priority, task in detection_tasks:
            try:
                results = await task
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
                
        filtered_results = await self.filter_and_rank_patterns(all_results)
        
        for result in filtered_results:
            await self.save_pattern(result)
            
        if self.redis_client and cache_key:
            await self._cache_patterns(cache_key, filtered_results)
            
        await self.analyze_correlations(filtered_results)
        
        await self.send_notifications(filtered_results)
        
        return filtered_results
    
    async def start_real_time_scanning(self, symbol: str, timeframe: Timeframe, 
                                      data_source: Callable, interval_seconds: int = 60):
        task_key = f"{symbol}_{timeframe.value}"
        
        if task_key in self.scanning_tasks:
            logger.warning(f"Scanning already active for {task_key}")
            return
            
        async def scanning_loop():
            while True:
                try:
                    data = await data_source(symbol, timeframe)
                    if data is not None and not data.empty:
                        patterns = await self.scan_for_patterns(data, symbol, timeframe)
                        logger.info(f"Detected {len(patterns)} patterns for {symbol} on {timeframe.value}")
                except Exception as e:
                    logger.error(f"Error in real-time scanning: {e}")
                    
                await asyncio.sleep(interval_seconds)
                
        task = asyncio.create_task(scanning_loop())
        self.scanning_tasks[task_key] = task
        logger.info(f"Started real-time scanning for {task_key}")
        
    async def stop_real_time_scanning(self, symbol: str, timeframe: Timeframe):
        task_key = f"{symbol}_{timeframe.value}"
        
        if task_key in self.scanning_tasks:
            task = self.scanning_tasks[task_key]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
                
            del self.scanning_tasks[task_key]
            logger.info(f"Stopped real-time scanning for {task_key}")
            
    async def filter_and_rank_patterns(self, patterns: List[PatternDetectionResult]) -> List[PatternDetectionResult]:
        filtered = []
        
        for pattern in patterns:
            if pattern.confidence < 0.6:
                continue
                
            if pattern.risk_reward_ratio and pattern.risk_reward_ratio < 1.5:
                continue
                
            filtered.append(pattern)
            
        filtered.sort(key=lambda p: (
            p.confidence,
            p.strength or 0,
            p.quality_score or 0
        ), reverse=True)
        
        seen_patterns = set()
        unique_patterns = []
        
        for pattern in filtered:
            pattern_key = (
                pattern.pattern_type,
                pattern.pattern_name,
                pattern.symbol,
                pattern.timeframe,
                pattern.start_time.date() if pattern.start_time else None
            )
            
            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                unique_patterns.append(pattern)
                
        return unique_patterns[:50]
    
    async def analyze_correlations(self, patterns: List[PatternDetectionResult]):
        if len(patterns) < 2:
            return
            
        correlation_pairs = []
        
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i+1:]:
                if self._are_patterns_correlated(pattern1, pattern2):
                    correlation_pairs.append((pattern1, pattern2))
                    
        for p1, p2 in correlation_pairs:
            try:
                saved_p1 = await self._get_saved_pattern(p1)
                saved_p2 = await self._get_saved_pattern(p2)
                
                if saved_p1 and saved_p2:
                    correlation_strength = self._calculate_correlation_strength(p1, p2)
                    temporal_relationship = self._determine_temporal_relationship(p1, p2)
                    
                    self.repository.save_correlation(
                        saved_p1.id,
                        saved_p2.id,
                        correlation_type="co_occurrence",
                        correlation_strength=correlation_strength,
                        temporal_relationship=temporal_relationship
                    )
            except Exception as e:
                logger.error(f"Error saving correlation: {e}")
                
    def _are_patterns_correlated(self, p1: PatternDetectionResult, p2: PatternDetectionResult) -> bool:
        if p1.symbol != p2.symbol:
            return False
            
        if p1.timeframe != p2.timeframe:
            return False
            
        time_diff = abs((p1.start_time - p2.start_time).total_seconds())
        max_time_diff = p1.timeframe.seconds * 10
        
        if time_diff > max_time_diff:
            return False
            
        if p1.direction == p2.direction:
            return True
            
        return False
    
    def _calculate_correlation_strength(self, p1: PatternDetectionResult, p2: PatternDetectionResult) -> float:
        strength = 0.5
        
        if p1.direction == p2.direction:
            strength += 0.2
            
        confidence_similarity = 1 - abs(p1.confidence - p2.confidence)
        strength += confidence_similarity * 0.2
        
        time_diff = abs((p1.start_time - p2.start_time).total_seconds())
        time_factor = max(0, 1 - (time_diff / (p1.timeframe.seconds * 10)))
        strength += time_factor * 0.1
        
        return min(1.0, strength)
    
    def _determine_temporal_relationship(self, p1: PatternDetectionResult, p2: PatternDetectionResult) -> str:
        if p1.start_time < p2.start_time:
            return "p1_leads"
        elif p2.start_time < p1.start_time:
            return "p2_leads"
        else:
            return "concurrent"
            
    async def send_notifications(self, patterns: List[PatternDetectionResult]):
        for pattern in patterns:
            priority = self._determine_notification_priority(pattern)
            channels = self._determine_notification_channels(pattern, priority)
            
            notification = PatternNotification(
                pattern=pattern,
                channels=channels,
                priority=priority,
                metadata={
                    'symbol': pattern.symbol,
                    'timeframe': pattern.timeframe.value,
                    'confidence': pattern.confidence
                }
            )
            
            for channel in channels:
                if channel in self.notification_handlers:
                    try:
                        handler = self.notification_handlers[channel]
                        await handler(notification)
                    except Exception as e:
                        logger.error(f"Error sending notification to {channel.value}: {e}")
                        
    def _determine_notification_priority(self, pattern: PatternDetectionResult) -> DetectorPriority:
        if pattern.confidence >= 0.9:
            return DetectorPriority.CRITICAL
        elif pattern.confidence >= 0.8:
            return DetectorPriority.HIGH
        elif pattern.confidence >= 0.7:
            return DetectorPriority.MEDIUM
        else:
            return DetectorPriority.LOW
            
    def _determine_notification_channels(self, pattern: PatternDetectionResult, 
                                        priority: DetectorPriority) -> List[NotificationChannel]:
        channels = [NotificationChannel.DATABASE]
        
        if priority == DetectorPriority.CRITICAL:
            channels.extend([NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL])
        elif priority == DetectorPriority.HIGH:
            channels.append(NotificationChannel.WEBSOCKET)
            
        return channels
    
    async def save_pattern(self, pattern: PatternDetectionResult) -> Optional[Pattern]:
        try:
            return self.repository.save_pattern(pattern)
        except Exception as e:
            logger.error(f"Error saving pattern: {e}")
            return None
            
    async def _get_saved_pattern(self, detection_result: PatternDetectionResult) -> Optional[Pattern]:
        filter_params = PatternFilter(
            pattern_types=[detection_result.pattern_type],
            pattern_names=[detection_result.pattern_name],
            symbols=[detection_result.symbol],
            timeframes=[detection_result.timeframe],
            start_date=detection_result.start_time - timedelta(seconds=60),
            end_date=detection_result.start_time + timedelta(seconds=60),
            limit=1
        )
        
        patterns, _ = self.repository.find_patterns(filter_params)
        return patterns[0] if patterns else None
    
    def _get_cache_key(self, symbol: str, timeframe: Timeframe, timestamp: Optional[Any]) -> Optional[str]:
        if not timestamp:
            return None
        return f"patterns:{symbol}:{timeframe.value}:{timestamp}"
        
    async def _get_cached_patterns(self, cache_key: str) -> Optional[List[PatternDetectionResult]]:
        if not self.redis_client:
            return None
            
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                patterns_data = json.loads(cached_data)
                patterns = []
                for data in patterns_data:
                    pattern = PatternDetectionResult(
                        pattern_type=PatternType(data['pattern_type']),
                        pattern_name=data['pattern_name'],
                        direction=PatternDirection(data['direction']),
                        confidence=data['confidence'],
                        timeframe=Timeframe(data['timeframe']),
                        start_time=datetime.fromisoformat(data['start_time']),
                        end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
                        symbol=data.get('symbol'),
                        entry_price=data.get('entry_price'),
                        target_price=data.get('target_price'),
                        stop_loss=data.get('stop_loss'),
                        strength=data.get('strength'),
                        quality_score=data.get('quality_score'),
                        metadata=data.get('metadata', {})
                    )
                    patterns.append(pattern)
                return patterns
        except Exception as e:
            logger.error(f"Error getting cached patterns: {e}")
            return None
            
    async def _cache_patterns(self, cache_key: str, patterns: List[PatternDetectionResult]):
        if not self.redis_client:
            return
            
        try:
            patterns_data = [p.to_dict() for p in patterns]
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(patterns_data, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching patterns: {e}")
            
    async def get_active_patterns(self, symbol: Optional[str] = None, 
                                 timeframe: Optional[Timeframe] = None) -> List[Pattern]:
        filter_params = PatternFilter(
            symbols=[symbol] if symbol else None,
            timeframes=[timeframe] if timeframe else None,
            statuses=[PatternStatus.ACTIVE, PatternStatus.DETECTED, PatternStatus.CONFIRMED],
            order_by='detected_at',
            order_desc=True
        )
        
        patterns, _ = self.repository.find_patterns(filter_params)
        return patterns
    
    async def get_pattern_performance(self, symbol: str, 
                                     pattern_type: Optional[PatternType] = None,
                                     days_back: int = 30) -> Dict[str, Any]:
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        filter_params = PatternFilter(
            symbols=[symbol],
            pattern_types=[pattern_type] if pattern_type else None,
            start_date=start_date,
            statuses=[PatternStatus.COMPLETED, PatternStatus.FAILED]
        )
        
        patterns, total = self.repository.find_patterns(filter_params)
        
        successful = [p for p in patterns if p.actual_outcome == 'success']
        failed = [p for p in patterns if p.actual_outcome == 'failure']
        
        returns = [p.actual_return for p in patterns if p.actual_return is not None]
        
        performance = {
            'total_patterns': total,
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / total if total > 0 else 0,
            'avg_return': sum(returns) / len(returns) if returns else 0,
            'best_return': max(returns) if returns else 0,
            'worst_return': min(returns) if returns else 0,
            'patterns_by_type': defaultdict(int),
            'patterns_by_direction': defaultdict(int)
        }
        
        for pattern in patterns:
            performance['patterns_by_type'][pattern.pattern_type] += 1
            performance['patterns_by_direction'][pattern.direction] += 1
            
        return performance