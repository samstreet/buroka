from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import create_engine, func, and_, or_, desc, asc
from sqlalchemy.orm import Session, sessionmaker, joinedload
from sqlalchemy.exc import IntegrityError
import logging
from contextlib import contextmanager

from src.data.models.pattern_models import (
    Pattern, PatternPerformance, PatternCorrelation, PatternStatistics, Tag,
    PatternType, PatternDirection, PatternStatus, Timeframe,
    PatternDetectionResult, PatternFilter, PatternRetentionPolicy,
    PatternClassification, Base
)

logger = logging.getLogger(__name__)


class PatternRepository:
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_size=20, max_overflow=40)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        
    @contextmanager
    def get_session(self) -> Session:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
            
    def save_pattern(self, detection_result: PatternDetectionResult) -> Pattern:
        with self.get_session() as session:
            pattern = Pattern(
                symbol=detection_result.symbol,
                pattern_type=detection_result.pattern_type.value,
                pattern_name=detection_result.pattern_name,
                direction=detection_result.direction.value,
                timeframe=detection_result.timeframe.value,
                start_time=detection_result.start_time,
                end_time=detection_result.end_time,
                confidence=detection_result.confidence,
                strength=detection_result.strength,
                quality_score=detection_result.quality_score,
                entry_price=detection_result.entry_price,
                target_price=detection_result.target_price,
                stop_loss=detection_result.stop_loss,
                risk_reward_ratio=detection_result.risk_reward_ratio,
                pattern_data=detection_result.metadata
            )
            
            if detection_result.duration:
                pattern.duration_seconds = int(detection_result.duration.total_seconds())
            
            PatternRetentionPolicy.set_expiration(pattern)
            
            for tag_name in detection_result.tags:
                tag = session.query(Tag).filter_by(name=tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                pattern.tags.append(tag)
            
            session.add(pattern)
            session.flush()
            
            self._update_statistics(session, pattern)
            
            return pattern
    
    def update_pattern_status(self, pattern_id: str, status: PatternStatus,
                            outcome: Optional[str] = None,
                            actual_return: Optional[float] = None) -> bool:
        with self.get_session() as session:
            pattern = session.query(Pattern).filter_by(id=pattern_id).first()
            if not pattern:
                return False
            
            pattern.status = status.value
            pattern.updated_at = datetime.utcnow()
            
            if outcome:
                pattern.actual_outcome = outcome
            if actual_return is not None:
                pattern.actual_return = actual_return
            
            if status in [PatternStatus.COMPLETED, PatternStatus.FAILED]:
                pattern.end_time = datetime.utcnow()
                pattern.duration_seconds = int((pattern.end_time - pattern.start_time).total_seconds())
                
                self._update_performance(session, pattern)
                self._update_statistics(session, pattern)
            
            PatternRetentionPolicy.set_expiration(pattern)
            
            return True
    
    def save_performance(self, pattern_id: str, performance_data: Dict[str, Any]) -> PatternPerformance:
        with self.get_session() as session:
            performance = session.query(PatternPerformance).filter_by(pattern_id=pattern_id).first()
            
            if not performance:
                performance = PatternPerformance(pattern_id=pattern_id)
                session.add(performance)
            
            for key, value in performance_data.items():
                if hasattr(performance, key):
                    setattr(performance, key, value)
            
            performance.updated_at = datetime.utcnow()
            
            self._calculate_performance_score(performance)
            
            return performance
    
    def find_patterns(self, filter_params: PatternFilter) -> Tuple[List[Pattern], int]:
        with self.get_session() as session:
            query = session.query(Pattern).options(
                joinedload(Pattern.performance),
                joinedload(Pattern.tags)
            )
            
            if filter_params.pattern_types:
                query = query.filter(Pattern.pattern_type.in_(
                    [pt.value for pt in filter_params.pattern_types]
                ))
            
            if filter_params.pattern_names:
                query = query.filter(Pattern.pattern_name.in_(filter_params.pattern_names))
            
            if filter_params.directions:
                query = query.filter(Pattern.direction.in_(
                    [d.value for d in filter_params.directions]
                ))
            
            if filter_params.symbols:
                query = query.filter(Pattern.symbol.in_(filter_params.symbols))
            
            if filter_params.timeframes:
                query = query.filter(Pattern.timeframe.in_(
                    [tf.value for tf in filter_params.timeframes]
                ))
            
            if filter_params.min_confidence is not None:
                query = query.filter(Pattern.confidence >= filter_params.min_confidence)
            
            if filter_params.max_confidence is not None:
                query = query.filter(Pattern.confidence <= filter_params.max_confidence)
            
            if filter_params.min_strength is not None:
                query = query.filter(Pattern.strength >= filter_params.min_strength)
            
            if filter_params.statuses:
                query = query.filter(Pattern.status.in_(
                    [s.value for s in filter_params.statuses]
                ))
            
            if filter_params.start_date:
                query = query.filter(Pattern.detected_at >= filter_params.start_date)
            
            if filter_params.end_date:
                query = query.filter(Pattern.detected_at <= filter_params.end_date)
            
            if filter_params.tags:
                query = query.join(Pattern.tags).filter(Tag.name.in_(filter_params.tags))
            
            total_count = query.count()
            
            order_column = getattr(Pattern, filter_params.order_by, Pattern.detected_at)
            if filter_params.order_desc:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(asc(order_column))
            
            patterns = query.offset(filter_params.offset).limit(filter_params.limit).all()
            
            return patterns, total_count
    
    def find_correlated_patterns(self, pattern_id: str, min_correlation: float = 0.5) -> List[PatternCorrelation]:
        with self.get_session() as session:
            correlations = session.query(PatternCorrelation).filter(
                and_(
                    PatternCorrelation.pattern_id == pattern_id,
                    PatternCorrelation.correlation_strength >= min_correlation
                )
            ).order_by(desc(PatternCorrelation.correlation_strength)).all()
            
            return correlations
    
    def save_correlation(self, pattern_id: str, correlated_pattern_id: str,
                        correlation_type: str, correlation_strength: float,
                        temporal_relationship: Optional[str] = None,
                        lag_seconds: Optional[int] = None) -> PatternCorrelation:
        with self.get_session() as session:
            correlation = session.query(PatternCorrelation).filter(
                and_(
                    PatternCorrelation.pattern_id == pattern_id,
                    PatternCorrelation.correlated_pattern_id == correlated_pattern_id
                )
            ).first()
            
            if not correlation:
                correlation = PatternCorrelation(
                    pattern_id=pattern_id,
                    correlated_pattern_id=correlated_pattern_id,
                    correlation_type=correlation_type,
                    correlation_strength=correlation_strength,
                    temporal_relationship=temporal_relationship,
                    lag_seconds=lag_seconds
                )
                session.add(correlation)
            else:
                correlation.correlation_strength = correlation_strength
                correlation.co_occurrence_count += 1
            
            return correlation
    
    def get_pattern_statistics(self, pattern_type: Optional[PatternType] = None,
                              pattern_name: Optional[str] = None,
                              timeframe: Optional[Timeframe] = None,
                              symbol: Optional[str] = None) -> List[PatternStatistics]:
        with self.get_session() as session:
            query = session.query(PatternStatistics)
            
            if pattern_type:
                query = query.filter(PatternStatistics.pattern_type == pattern_type.value)
            if pattern_name:
                query = query.filter(PatternStatistics.pattern_name == pattern_name)
            if timeframe:
                query = query.filter(PatternStatistics.timeframe == timeframe.value)
            if symbol:
                query = query.filter(PatternStatistics.symbol == symbol)
            
            return query.order_by(desc(PatternStatistics.success_rate)).all()
    
    def classify_pattern(self, pattern: Pattern, market_data: Dict[str, Any]) -> PatternClassification:
        classification = PatternClassification(
            primary_direction=PatternDirection(pattern.direction)
        )
        
        if pattern.confidence >= 0.8:
            classification.confidence_factors['pattern_confidence'] = pattern.confidence
        
        if pattern.strength and pattern.strength >= 0.7:
            classification.confidence_factors['pattern_strength'] = pattern.strength
        
        if 'trend_direction' in market_data:
            trend = market_data['trend_direction']
            if trend == pattern.direction:
                classification.trend_alignment = True
        
        if 'volume_confirmation' in market_data:
            classification.volume_confirmation = market_data['volume_confirmation']
        
        if 'momentum_indicators' in market_data:
            momentum = market_data['momentum_indicators']
            if momentum.get('rsi') and 30 < momentum['rsi'] < 70:
                classification.momentum_confirmation = True
        
        if pattern.risk_reward_ratio:
            if pattern.risk_reward_ratio >= 3:
                classification.risk_level = 'low'
            elif pattern.risk_reward_ratio >= 2:
                classification.risk_level = 'medium'
            else:
                classification.risk_level = 'high'
        
        if 'market_phase' in market_data:
            classification.market_phase = market_data['market_phase']
        
        trading_bias = classification.get_trading_bias()
        if trading_bias in ['strong_buy', 'buy']:
            classification.suggested_action = 'enter_long'
        elif trading_bias in ['strong_sell', 'sell']:
            classification.suggested_action = 'enter_short'
        else:
            classification.suggested_action = 'wait'
        
        return classification
    
    def cleanup_expired_patterns(self, batch_size: int = 1000) -> int:
        with self.get_session() as session:
            expired_patterns = session.query(Pattern).filter(
                and_(
                    Pattern.expires_at is not None,
                    Pattern.expires_at < datetime.utcnow()
                )
            ).limit(batch_size).all()
            
            deleted_count = 0
            for pattern in expired_patterns:
                session.delete(pattern)
                deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} expired patterns")
            return deleted_count
    
    def archive_old_patterns(self, days_old: int = 365, archive_table: str = 'patterns_archive') -> int:
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            old_patterns = session.query(Pattern).filter(
                Pattern.created_at < cutoff_date
            ).all()
            
            archived_count = 0
            for pattern in old_patterns:
                if PatternRetentionPolicy.should_archive(pattern):
                    session.delete(pattern)
                    archived_count += 1
            
            logger.info(f"Archived {archived_count} old patterns")
            return archived_count
    
    def _update_statistics(self, session: Session, pattern: Pattern) -> None:
        stats = session.query(PatternStatistics).filter(
            and_(
                PatternStatistics.pattern_type == pattern.pattern_type,
                PatternStatistics.pattern_name == pattern.pattern_name,
                PatternStatistics.timeframe == pattern.timeframe,
                PatternStatistics.symbol == pattern.symbol
            )
        ).first()
        
        if not stats:
            stats = PatternStatistics(
                pattern_type=pattern.pattern_type,
                pattern_name=pattern.pattern_name,
                timeframe=pattern.timeframe,
                symbol=pattern.symbol
            )
            session.add(stats)
        
        stats.total_occurrences += 1
        stats.last_occurrence = datetime.utcnow()
        
        if pattern.actual_outcome == 'success':
            stats.successful_occurrences += 1
            stats.last_success = datetime.utcnow()
        elif pattern.actual_outcome == 'failure':
            stats.failed_occurrences += 1
            stats.last_failure = datetime.utcnow()
        
        if stats.total_occurrences > 0:
            stats.success_rate = stats.successful_occurrences / stats.total_occurrences
        
        all_patterns = session.query(Pattern).filter(
            and_(
                Pattern.pattern_type == pattern.pattern_type,
                Pattern.pattern_name == pattern.pattern_name,
                Pattern.timeframe == pattern.timeframe,
                Pattern.symbol == pattern.symbol
            )
        ).all()
        
        if all_patterns:
            confidences = [p.confidence for p in all_patterns if p.confidence]
            returns = [p.actual_return for p in all_patterns if p.actual_return is not None]
            durations = [p.duration_seconds for p in all_patterns if p.duration_seconds]
            risk_rewards = [p.risk_reward_ratio for p in all_patterns if p.risk_reward_ratio]
            
            if confidences:
                stats.avg_confidence = sum(confidences) / len(confidences)
            if returns:
                stats.avg_return = sum(returns) / len(returns)
                stats.best_return = max(returns)
                stats.worst_return = min(returns)
                if len(returns) > 1:
                    mean_return = stats.avg_return
                    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                    stats.std_dev_return = variance ** 0.5
            if durations:
                stats.avg_duration = sum(durations) / len(durations)
            if risk_rewards:
                stats.avg_risk_reward = sum(risk_rewards) / len(risk_rewards)
        
        stats.updated_at = datetime.utcnow()
    
    def _update_performance(self, session: Session, pattern: Pattern) -> None:
        performance = session.query(PatternPerformance).filter_by(pattern_id=pattern.id).first()
        
        if not performance:
            performance = PatternPerformance(pattern_id=pattern.id)
            session.add(performance)
        
        if pattern.actual_return is not None:
            if pattern.actual_return > 0:
                performance.hit_target = True
                performance.max_favorable_excursion = pattern.actual_return
            else:
                performance.hit_stop_loss = True
                performance.max_adverse_excursion = abs(pattern.actual_return)
        
        if pattern.end_time and pattern.start_time:
            duration = (pattern.end_time - pattern.start_time).total_seconds()
            if performance.hit_target:
                performance.time_to_target = int(duration)
            elif performance.hit_stop_loss:
                performance.time_to_stop = int(duration)
        
        self._calculate_performance_score(performance)
        
        performance.updated_at = datetime.utcnow()
    
    def _calculate_performance_score(self, performance: PatternPerformance) -> None:
        score = 0.5
        
        if performance.hit_target:
            score += 0.3
        elif performance.hit_stop_loss:
            score -= 0.2
        
        if performance.max_favorable_excursion:
            if performance.max_favorable_excursion > 0.1:
                score += 0.2
            elif performance.max_favorable_excursion > 0.05:
                score += 0.1
        
        if performance.max_adverse_excursion:
            if performance.max_adverse_excursion > 0.1:
                score -= 0.2
            elif performance.max_adverse_excursion > 0.05:
                score -= 0.1
        
        if performance.follow_through_rate:
            score += performance.follow_through_rate * 0.2
        
        if performance.false_breakout:
            score -= 0.15
        
        performance.performance_score = max(0.0, min(1.0, score))