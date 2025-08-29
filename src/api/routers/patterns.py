from fastapi import APIRouter, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import asyncio
import json
import logging

from src.data.models.pattern_models import (
    PatternType, PatternDirection, PatternStatus, Timeframe,
    PatternDetectionResult, PatternFilter, Pattern, PatternStatistics
)
from src.data.storage.pattern_repository import PatternRepository
from src.services.pattern_detection_service import (
    PatternDetectionService, NotificationChannel, DetectorPriority
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/patterns", tags=["patterns"])


class PatternResponse(BaseModel):
    id: str
    symbol: str
    pattern_type: str
    pattern_name: str
    direction: str
    status: str
    timeframe: str
    detected_at: datetime
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[int]
    confidence: float
    strength: Optional[float]
    quality_score: Optional[float]
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    actual_outcome: Optional[str]
    actual_return: Optional[float]
    metadata: Optional[Dict[str, Any]]
    
    class Config:
        orm_mode = True


class PatternSearchRequest(BaseModel):
    symbols: Optional[List[str]] = Field(None, description="List of symbols to filter")
    pattern_types: Optional[List[str]] = Field(None, description="Pattern types to filter")
    pattern_names: Optional[List[str]] = Field(None, description="Specific pattern names")
    directions: Optional[List[str]] = Field(None, description="Pattern directions (bullish/bearish/neutral)")
    timeframes: Optional[List[str]] = Field(None, description="Timeframes to filter")
    min_confidence: Optional[float] = Field(None, ge=0, le=1, description="Minimum confidence score")
    max_confidence: Optional[float] = Field(None, ge=0, le=1, description="Maximum confidence score")
    min_strength: Optional[float] = Field(None, ge=0, le=1, description="Minimum strength score")
    statuses: Optional[List[str]] = Field(None, description="Pattern statuses")
    start_date: Optional[datetime] = Field(None, description="Start date for pattern detection")
    end_date: Optional[datetime] = Field(None, description="End date for pattern detection")
    tags: Optional[List[str]] = Field(None, description="Pattern tags")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    order_by: str = Field("detected_at", description="Field to order by")
    order_desc: bool = Field(True, description="Order descending")
    
    @validator('pattern_types', 'directions', 'timeframes', 'statuses', pre=True)
    def validate_enums(cls, v):
        if v:
            return [item.upper() if isinstance(item, str) else item for item in v]
        return v


class PatternPerformanceResponse(BaseModel):
    symbol: str
    pattern_type: Optional[str]
    timeframe: Optional[str]
    period_days: int
    total_patterns: int
    successful: int
    failed: int
    pending: int
    success_rate: float
    avg_return: float
    best_return: float
    worst_return: float
    avg_confidence: float
    avg_duration_hours: float
    patterns_by_type: Dict[str, int]
    patterns_by_direction: Dict[str, int]
    daily_distribution: Dict[str, int]


class PatternSubscriptionRequest(BaseModel):
    symbols: List[str]
    pattern_types: Optional[List[str]] = None
    min_confidence: float = Field(0.7, ge=0, le=1)
    notification_channels: List[str] = Field(["websocket"], description="Notification channels")
    webhook_url: Optional[str] = None
    email: Optional[str] = None


class PatternVisualizationData(BaseModel):
    pattern_id: str
    symbol: str
    pattern_type: str
    pattern_name: str
    chart_data: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    indicators: Dict[str, List[float]]
    support_resistance_levels: List[float]
    volume_profile: Optional[Dict[str, Any]]


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.subscriptions: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
        
    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
            
    async def send_pattern(self, client_id: str, pattern: Dict[str, Any]):
        if client_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_json(pattern)
                except:
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.active_connections[client_id].discard(conn)
                
    async def broadcast_pattern(self, pattern: PatternDetectionResult):
        for websocket, subscription in self.subscriptions.items():
            if self._matches_subscription(pattern, subscription):
                try:
                    await websocket.send_json(pattern.to_dict())
                except:
                    pass
                    
    def _matches_subscription(self, pattern: PatternDetectionResult, subscription: Dict[str, Any]) -> bool:
        if subscription.get('symbols') and pattern.symbol not in subscription['symbols']:
            return False
        if subscription.get('pattern_types') and pattern.pattern_type.value not in subscription['pattern_types']:
            return False
        if subscription.get('min_confidence') and pattern.confidence < subscription['min_confidence']:
            return False
        return True


websocket_manager = WebSocketManager()


def get_repository() -> PatternRepository:
    return PatternRepository("postgresql://trader:password@localhost/market_analysis")


def get_detection_service(repository: PatternRepository = Depends(get_repository)) -> PatternDetectionService:
    return PatternDetectionService(repository)


@router.get("/{symbol}", response_model=List[PatternResponse])
async def get_patterns_by_symbol(
    symbol: str,
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
    status: Optional[str] = Query(None, description="Filter by status"),
    days_back: int = Query(7, ge=1, le=365, description="Days to look back"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    repository: PatternRepository = Depends(get_repository)
) -> List[PatternResponse]:
    
    filter_params = PatternFilter(
        symbols=[symbol],
        pattern_types=[PatternType[pattern_type.upper()]] if pattern_type else None,
        timeframes=[Timeframe[timeframe.upper()]] if timeframe else None,
        min_confidence=min_confidence,
        statuses=[PatternStatus[status.upper()]] if status else None,
        start_date=datetime.utcnow() - timedelta(days=days_back),
        limit=limit,
        order_by='detected_at',
        order_desc=True
    )
    
    patterns, _ = repository.find_patterns(filter_params)
    
    return [PatternResponse.from_orm(p) for p in patterns]


@router.post("/search", response_model=Dict[str, Any])
async def search_patterns(
    request: PatternSearchRequest,
    repository: PatternRepository = Depends(get_repository)
) -> Dict[str, Any]:
    
    filter_params = PatternFilter(
        symbols=request.symbols,
        pattern_types=[PatternType[pt] for pt in request.pattern_types] if request.pattern_types else None,
        pattern_names=request.pattern_names,
        directions=[PatternDirection[d] for d in request.directions] if request.directions else None,
        timeframes=[Timeframe[tf] for tf in request.timeframes] if request.timeframes else None,
        min_confidence=request.min_confidence,
        max_confidence=request.max_confidence,
        min_strength=request.min_strength,
        statuses=[PatternStatus[s] for s in request.statuses] if request.statuses else None,
        start_date=request.start_date,
        end_date=request.end_date,
        tags=request.tags,
        limit=request.limit,
        offset=request.offset,
        order_by=request.order_by,
        order_desc=request.order_desc
    )
    
    patterns, total_count = repository.find_patterns(filter_params)
    
    return {
        'patterns': [PatternResponse.from_orm(p) for p in patterns],
        'total_count': total_count,
        'limit': request.limit,
        'offset': request.offset,
        'has_more': total_count > request.offset + request.limit
    }


@router.get("/{symbol}/performance", response_model=PatternPerformanceResponse)
async def get_pattern_performance(
    symbol: str,
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    days: int = Query(30, ge=1, le=365, description="Period in days"),
    service: PatternDetectionService = Depends(get_detection_service)
) -> PatternPerformanceResponse:
    
    performance = await service.get_pattern_performance(
        symbol,
        PatternType[pattern_type.upper()] if pattern_type else None,
        days
    )
    
    daily_distribution = {}
    start_date = datetime.utcnow() - timedelta(days=days)
    
    for i in range(days):
        date = (start_date + timedelta(days=i)).date().isoformat()
        daily_distribution[date] = 0
    
    filter_params = PatternFilter(
        symbols=[symbol],
        pattern_types=[PatternType[pattern_type.upper()]] if pattern_type else None,
        timeframes=[Timeframe[timeframe.upper()]] if timeframe else None,
        start_date=start_date
    )
    
    patterns, _ = service.repository.find_patterns(filter_params)
    
    for pattern in patterns:
        date_key = pattern.detected_at.date().isoformat()
        if date_key in daily_distribution:
            daily_distribution[date_key] += 1
    
    confidences = [p.confidence for p in patterns if p.confidence]
    durations = [p.duration_seconds for p in patterns if p.duration_seconds]
    
    return PatternPerformanceResponse(
        symbol=symbol,
        pattern_type=pattern_type,
        timeframe=timeframe,
        period_days=days,
        total_patterns=performance['total_patterns'],
        successful=performance['successful'],
        failed=performance['failed'],
        pending=performance['total_patterns'] - performance['successful'] - performance['failed'],
        success_rate=performance['success_rate'],
        avg_return=performance['avg_return'],
        best_return=performance['best_return'],
        worst_return=performance['worst_return'],
        avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
        avg_duration_hours=sum(durations) / len(durations) / 3600 if durations else 0,
        patterns_by_type=dict(performance['patterns_by_type']),
        patterns_by_direction=dict(performance['patterns_by_direction']),
        daily_distribution=daily_distribution
    )


@router.get("/statistics/{pattern_type}/{pattern_name}", response_model=Dict[str, Any])
async def get_pattern_statistics(
    pattern_type: str,
    pattern_name: str,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    repository: PatternRepository = Depends(get_repository)
) -> Dict[str, Any]:
    
    stats = repository.get_pattern_statistics(
        pattern_type=PatternType[pattern_type.upper()] if pattern_type else None,
        pattern_name=pattern_name,
        timeframe=Timeframe[timeframe.upper()] if timeframe else None,
        symbol=symbol
    )
    
    if not stats:
        raise HTTPException(status_code=404, detail="No statistics found for pattern")
    
    stat = stats[0]
    
    return {
        'pattern_type': stat.pattern_type,
        'pattern_name': stat.pattern_name,
        'symbol': stat.symbol,
        'timeframe': stat.timeframe,
        'total_occurrences': stat.total_occurrences,
        'successful_occurrences': stat.successful_occurrences,
        'failed_occurrences': stat.failed_occurrences,
        'success_rate': stat.success_rate,
        'avg_confidence': stat.avg_confidence,
        'avg_return': stat.avg_return,
        'avg_duration': stat.avg_duration,
        'avg_risk_reward': stat.avg_risk_reward,
        'best_return': stat.best_return,
        'worst_return': stat.worst_return,
        'std_dev_return': stat.std_dev_return,
        'last_occurrence': stat.last_occurrence.isoformat() if stat.last_occurrence else None,
        'last_success': stat.last_success.isoformat() if stat.last_success else None,
        'last_failure': stat.last_failure.isoformat() if stat.last_failure else None
    }


@router.post("/subscribe")
async def subscribe_to_patterns(
    request: PatternSubscriptionRequest,
    background_tasks: BackgroundTasks,
    service: PatternDetectionService = Depends(get_detection_service)
) -> Dict[str, Any]:
    
    subscription_id = f"sub_{datetime.utcnow().timestamp()}"
    
    async def notification_handler(notification):
        pattern_data = notification.pattern.to_dict()
        
        if "websocket" in request.notification_channels:
            await websocket_manager.broadcast_pattern(notification.pattern)
            
        if "webhook" in request.notification_channels and request.webhook_url:
            background_tasks.add_task(send_webhook, request.webhook_url, pattern_data)
            
        if "email" in request.notification_channels and request.email:
            background_tasks.add_task(send_email_notification, request.email, pattern_data)
    
    for channel_str in request.notification_channels:
        try:
            channel = NotificationChannel[channel_str.upper()]
            service.register_notification_handler(channel, notification_handler)
        except KeyError:
            logger.warning(f"Unknown notification channel: {channel_str}")
    
    return {
        'subscription_id': subscription_id,
        'symbols': request.symbols,
        'pattern_types': request.pattern_types,
        'min_confidence': request.min_confidence,
        'channels': request.notification_channels,
        'status': 'active'
    }


@router.get("/visualization/{pattern_id}", response_model=PatternVisualizationData)
async def get_pattern_visualization(
    pattern_id: str,
    repository: PatternRepository = Depends(get_repository)
) -> PatternVisualizationData:
    
    filter_params = PatternFilter(limit=1)
    patterns, _ = repository.find_patterns(filter_params)
    
    if not patterns:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    pattern = patterns[0]
    
    chart_data = []
    annotations = []
    
    if pattern.pattern_data:
        if 'price_points' in pattern.pattern_data:
            for point in pattern.pattern_data['price_points']:
                chart_data.append({
                    'time': point.get('time'),
                    'value': point.get('value'),
                    'type': 'price'
                })
        
        if 'pattern_points' in pattern.pattern_data:
            for point in pattern.pattern_data['pattern_points']:
                annotations.append({
                    'time': point.get('time'),
                    'price': point.get('price'),
                    'label': point.get('label', ''),
                    'color': '#00ff00' if pattern.direction == 'bullish' else '#ff0000'
                })
    
    if pattern.entry_price:
        annotations.append({
            'type': 'horizontal_line',
            'price': pattern.entry_price,
            'label': 'Entry',
            'color': '#0000ff'
        })
    
    if pattern.target_price:
        annotations.append({
            'type': 'horizontal_line',
            'price': pattern.target_price,
            'label': 'Target',
            'color': '#00ff00'
        })
    
    if pattern.stop_loss:
        annotations.append({
            'type': 'horizontal_line',
            'price': pattern.stop_loss,
            'label': 'Stop Loss',
            'color': '#ff0000'
        })
    
    indicators = {}
    if pattern.pattern_data and 'indicators' in pattern.pattern_data:
        indicators = pattern.pattern_data['indicators']
    
    support_resistance = []
    if pattern.pattern_data and 'support_resistance' in pattern.pattern_data:
        support_resistance = pattern.pattern_data['support_resistance']
    
    volume_profile = None
    if pattern.pattern_data and 'volume_profile' in pattern.pattern_data:
        volume_profile = pattern.pattern_data['volume_profile']
    
    return PatternVisualizationData(
        pattern_id=str(pattern.id),
        symbol=pattern.symbol,
        pattern_type=pattern.pattern_type,
        pattern_name=pattern.pattern_name,
        chart_data=chart_data,
        annotations=annotations,
        indicators=indicators,
        support_resistance_levels=support_resistance,
        volume_profile=volume_profile
    )


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    service: PatternDetectionService = Depends(get_detection_service)
):
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('action') == 'subscribe':
                subscription = {
                    'symbols': data.get('symbols', []),
                    'pattern_types': data.get('pattern_types'),
                    'min_confidence': data.get('min_confidence', 0.7)
                }
                websocket_manager.subscriptions[websocket] = subscription
                
                await websocket.send_json({
                    'type': 'subscription_confirmed',
                    'subscription': subscription
                })
                
            elif data.get('action') == 'unsubscribe':
                if websocket in websocket_manager.subscriptions:
                    del websocket_manager.subscriptions[websocket]
                    
                await websocket.send_json({
                    'type': 'unsubscribed'
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, client_id)
        logger.info(f"WebSocket client {client_id} disconnected")


@router.get("/discovery/trending", response_model=List[Dict[str, Any]])
async def discover_trending_patterns(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    min_occurrences: int = Query(3, ge=1, description="Minimum pattern occurrences"),
    repository: PatternRepository = Depends(get_repository)
) -> List[Dict[str, Any]]:
    
    start_date = datetime.utcnow() - timedelta(hours=hours)
    
    filter_params = PatternFilter(
        start_date=start_date,
        limit=1000,
        order_by='detected_at',
        order_desc=True
    )
    
    patterns, _ = repository.find_patterns(filter_params)
    
    pattern_counts = {}
    for pattern in patterns:
        key = (pattern.pattern_type, pattern.pattern_name)
        if key not in pattern_counts:
            pattern_counts[key] = {
                'pattern_type': pattern.pattern_type,
                'pattern_name': pattern.pattern_name,
                'count': 0,
                'symbols': set(),
                'avg_confidence': [],
                'directions': {}
            }
        
        pattern_counts[key]['count'] += 1
        pattern_counts[key]['symbols'].add(pattern.symbol)
        pattern_counts[key]['avg_confidence'].append(pattern.confidence)
        
        direction = pattern.direction
        if direction not in pattern_counts[key]['directions']:
            pattern_counts[key]['directions'][direction] = 0
        pattern_counts[key]['directions'][direction] += 1
    
    trending = []
    for key, data in pattern_counts.items():
        if data['count'] >= min_occurrences:
            trending.append({
                'pattern_type': data['pattern_type'],
                'pattern_name': data['pattern_name'],
                'occurrences': data['count'],
                'unique_symbols': len(data['symbols']),
                'symbols': list(data['symbols'])[:10],
                'avg_confidence': sum(data['avg_confidence']) / len(data['avg_confidence']),
                'dominant_direction': max(data['directions'], key=data['directions'].get),
                'direction_distribution': data['directions']
            })
    
    trending.sort(key=lambda x: x['occurrences'], reverse=True)
    
    return trending[:20]


async def send_webhook(url: str, data: Dict[str, Any]):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Webhook failed with status {response.status}")
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")


async def send_email_notification(email: str, pattern_data: Dict[str, Any]):
    logger.info(f"Would send email to {email} with pattern: {pattern_data['pattern_name']}")