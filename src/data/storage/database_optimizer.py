"""
Database optimization utilities including index creation, query optimization,
and performance monitoring. Following SOLID principles and TDD methodology.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .optimized_clients import OptimizedPostgreSQLClient, ConnectionPoolConfig

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Database index types."""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"


@dataclass
class IndexDefinition:
    """Database index definition."""
    table_name: str
    index_name: str
    columns: List[str]
    index_type: IndexType = IndexType.BTREE
    where_clause: Optional[str] = None
    include_columns: Optional[List[str]] = None
    unique: bool = False
    concurrent: bool = True


@dataclass
class QueryOptimization:
    """Query optimization configuration."""
    query_signature: str
    optimization_type: str
    before_stats: Dict[str, Any] = field(default_factory=dict)
    after_stats: Dict[str, Any] = field(default_factory=dict)
    improvement_ratio: float = 0.0


class DatabaseOptimizer:
    """
    Comprehensive database optimizer for PostgreSQL and InfluxDB.
    Handles index creation, query optimization, and performance monitoring.
    """
    
    def __init__(self, postgres_client: OptimizedPostgreSQLClient):
        self.postgres_client = postgres_client
        self.optimization_history: List[QueryOptimization] = []
        self._lock = asyncio.Lock()
    
    async def create_market_data_indexes(self) -> Dict[str, bool]:
        """Create optimized indexes for market data tables."""
        index_definitions = [
            # Primary market data table indexes
            IndexDefinition(
                table_name="market_data",
                index_name="idx_market_data_symbol_timestamp",
                columns=["symbol", "timestamp"],
                index_type=IndexType.BTREE
            ),
            IndexDefinition(
                table_name="market_data",
                index_name="idx_market_data_timestamp",
                columns=["timestamp"],
                index_type=IndexType.BRIN  # Efficient for time-series data
            ),
            IndexDefinition(
                table_name="market_data",
                index_name="idx_market_data_symbol",
                columns=["symbol"],
                index_type=IndexType.HASH  # Fast equality lookups
            ),
            IndexDefinition(
                table_name="market_data",
                index_name="idx_market_data_volume",
                columns=["volume"],
                index_type=IndexType.BTREE,
                where_clause="volume > 0"  # Partial index for non-zero volumes
            ),
            
            # Technical indicators table indexes
            IndexDefinition(
                table_name="technical_indicators",
                index_name="idx_technical_indicators_symbol_type_timestamp",
                columns=["symbol", "indicator_type", "timestamp"],
                index_type=IndexType.BTREE
            ),
            IndexDefinition(
                table_name="technical_indicators",
                index_name="idx_technical_indicators_timestamp",
                columns=["timestamp"],
                index_type=IndexType.BRIN
            ),
            
            # Pattern detection table indexes
            IndexDefinition(
                table_name="pattern_detections",
                index_name="idx_pattern_detections_symbol_pattern_timestamp",
                columns=["symbol", "pattern_type", "timestamp"],
                index_type=IndexType.BTREE
            ),
            IndexDefinition(
                table_name="pattern_detections",
                index_name="idx_pattern_detections_confidence",
                columns=["confidence_score"],
                index_type=IndexType.BTREE,
                where_clause="confidence_score >= 0.7"  # High confidence patterns
            ),
            
            # User activity indexes
            IndexDefinition(
                table_name="user_sessions",
                index_name="idx_user_sessions_user_timestamp",
                columns=["user_id", "created_at"],
                index_type=IndexType.BTREE
            ),
            
            # Composite indexes for common query patterns
            IndexDefinition(
                table_name="market_data",
                index_name="idx_market_data_symbol_timestamp_covering",
                columns=["symbol", "timestamp"],
                index_type=IndexType.BTREE,
                include_columns=["open_price", "high_price", "low_price", "close_price", "volume"]
            )
        ]
        
        results = {}
        
        for index_def in index_definitions:
            try:
                success = await self._create_index(index_def)
                results[index_def.index_name] = success
                
                if success:
                    logger.info(f"Created index: {index_def.index_name}")
                else:
                    logger.warning(f"Failed to create index: {index_def.index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating index {index_def.index_name}: {e}")
                results[index_def.index_name] = False
        
        return results
    
    async def _create_index(self, index_def: IndexDefinition) -> bool:
        """Create a single index."""
        try:
            # Check if index already exists
            check_query = """
                SELECT indexname FROM pg_indexes 
                WHERE indexname = $1 AND tablename = $2
            """
            
            existing = await self.postgres_client.execute_query(
                check_query, 
                index_def.index_name, 
                index_def.table_name,
                fetch_mode="one"
            )
            
            if existing:
                logger.info(f"Index {index_def.index_name} already exists")
                return True
            
            # Build CREATE INDEX statement
            sql_parts = ["CREATE"]
            
            if index_def.unique:
                sql_parts.append("UNIQUE")
            
            sql_parts.extend(["INDEX"])
            
            if index_def.concurrent:
                sql_parts.append("CONCURRENTLY")
            
            sql_parts.append(f"{index_def.index_name}")
            sql_parts.append(f"ON {index_def.table_name}")
            
            if index_def.index_type != IndexType.BTREE:
                sql_parts.append(f"USING {index_def.index_type.value}")
            
            # Add columns
            columns_str = ", ".join(index_def.columns)
            sql_parts.append(f"({columns_str})")
            
            # Add INCLUDE columns for covering indexes
            if index_def.include_columns:
                include_str = ", ".join(index_def.include_columns)
                sql_parts.append(f"INCLUDE ({include_str})")
            
            # Add WHERE clause for partial indexes
            if index_def.where_clause:
                sql_parts.append(f"WHERE {index_def.where_clause}")
            
            create_sql = " ".join(sql_parts)
            
            # Execute index creation
            await self.postgres_client.execute_query(create_sql, fetch_mode="none")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index {index_def.index_name}: {e}")
            return False
    
    async def analyze_query_performance(self, query: str, *args) -> Dict[str, Any]:
        """Analyze query performance using EXPLAIN ANALYZE."""
        try:
            # Run EXPLAIN ANALYZE
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            result = await self.postgres_client.execute_query(
                explain_query, 
                *args,
                fetch_mode="one"
            )
            
            if result and result.get('QUERY PLAN'):
                plan = result['QUERY PLAN'][0]
                
                return {
                    'execution_time_ms': plan.get('Execution Time', 0),
                    'planning_time_ms': plan.get('Planning Time', 0),
                    'total_cost': plan.get('Plan', {}).get('Total Cost', 0),
                    'actual_rows': plan.get('Plan', {}).get('Actual Rows', 0),
                    'buffer_usage': self._extract_buffer_usage(plan),
                    'index_usage': self._extract_index_usage(plan),
                    'full_plan': plan
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Query performance analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_buffer_usage(self, plan: Dict[str, Any]) -> Dict[str, int]:
        """Extract buffer usage statistics from execution plan."""
        def extract_recursive(node, stats):
            if isinstance(node, dict):
                if 'Shared Hit Blocks' in node:
                    stats['shared_hit_blocks'] += node['Shared Hit Blocks']
                if 'Shared Read Blocks' in node:
                    stats['shared_read_blocks'] += node['Shared Read Blocks']
                if 'Shared Dirtied Blocks' in node:
                    stats['shared_dirtied_blocks'] += node['Shared Dirtied Blocks']
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        extract_recursive(value, stats)
            elif isinstance(node, list):
                for item in node:
                    extract_recursive(item, stats)
        
        buffer_stats = {
            'shared_hit_blocks': 0,
            'shared_read_blocks': 0,
            'shared_dirtied_blocks': 0
        }
        
        extract_recursive(plan, buffer_stats)
        return buffer_stats
    
    def _extract_index_usage(self, plan: Dict[str, Any]) -> List[str]:
        """Extract index usage information from execution plan."""
        indexes_used = []
        
        def extract_recursive(node):
            if isinstance(node, dict):
                if node.get('Node Type') in ['Index Scan', 'Index Only Scan', 'Bitmap Index Scan']:
                    index_name = node.get('Index Name')
                    if index_name:
                        indexes_used.append(index_name)
                
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        extract_recursive(value)
            elif isinstance(node, list):
                for item in node:
                    extract_recursive(item)
        
        extract_recursive(plan)
        return list(set(indexes_used))  # Remove duplicates
    
    async def optimize_common_queries(self) -> List[QueryOptimization]:
        """Optimize common query patterns."""
        optimizations = []
        
        # Define common query patterns to optimize
        common_queries = [
            {
                'name': 'market_data_by_symbol_timerange',
                'query': """
                    SELECT symbol, timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data 
                    WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
                    ORDER BY timestamp DESC
                """,
                'test_params': ['AAPL', '2024-01-01 00:00:00+00', '2024-01-02 00:00:00+00']
            },
            {
                'name': 'recent_market_data_multiple_symbols',
                'query': """
                    SELECT symbol, timestamp, close_price, volume
                    FROM market_data 
                    WHERE symbol = ANY($1) AND timestamp >= $2
                    ORDER BY symbol, timestamp DESC
                """,
                'test_params': [['AAPL', 'GOOGL', 'MSFT'], '2024-01-01 00:00:00+00']
            },
            {
                'name': 'technical_indicators_latest',
                'query': """
                    SELECT DISTINCT ON (symbol, indicator_type) 
                           symbol, indicator_type, value, timestamp
                    FROM technical_indicators
                    WHERE symbol = $1 AND indicator_type = $2
                    ORDER BY symbol, indicator_type, timestamp DESC
                """,
                'test_params': ['AAPL', 'sma_20']
            },
            {
                'name': 'high_confidence_patterns',
                'query': """
                    SELECT symbol, pattern_type, confidence_score, timestamp
                    FROM pattern_detections
                    WHERE confidence_score >= $1 AND timestamp >= $2
                    ORDER BY confidence_score DESC, timestamp DESC
                """,
                'test_params': [0.8, '2024-01-01 00:00:00+00']
            }
        ]
        
        for query_info in common_queries:
            try:
                optimization = await self._optimize_query(
                    query_info['name'],
                    query_info['query'],
                    query_info['test_params']
                )
                optimizations.append(optimization)
                
            except Exception as e:
                logger.error(f"Failed to optimize query {query_info['name']}: {e}")
        
        return optimizations
    
    async def _optimize_query(
        self, 
        query_name: str, 
        query: str, 
        test_params: List[Any]
    ) -> QueryOptimization:
        """Optimize a specific query and measure improvement."""
        
        # Analyze performance before optimization
        before_stats = await self.analyze_query_performance(query, *test_params)
        
        optimization = QueryOptimization(
            query_signature=query_name,
            optimization_type="index_and_rewrite",
            before_stats=before_stats
        )
        
        # Apply query optimizations
        optimized_query = await self._apply_query_optimizations(query)
        
        # Analyze performance after optimization
        after_stats = await self.analyze_query_performance(optimized_query, *test_params)
        optimization.after_stats = after_stats
        
        # Calculate improvement
        before_time = before_stats.get('execution_time_ms', 0)
        after_time = after_stats.get('execution_time_ms', 0)
        
        if before_time > 0 and after_time > 0:
            optimization.improvement_ratio = (before_time - after_time) / before_time
        
        # Store optimization history
        async with self._lock:
            self.optimization_history.append(optimization)
        
        logger.info(f"Query {query_name} optimization: {optimization.improvement_ratio:.2%} improvement")
        
        return optimization
    
    async def _apply_query_optimizations(self, query: str) -> str:
        """Apply common query optimizations."""
        optimized = query
        
        # Add query hints and optimizations
        optimizations = [
            # Use explicit JOIN syntax instead of WHERE clauses
            self._convert_implicit_joins,
            # Add LIMIT for large result sets
            self._add_reasonable_limits,
            # Optimize ORDER BY clauses
            self._optimize_order_by
        ]
        
        for optimization_func in optimizations:
            try:
                optimized = optimization_func(optimized)
            except Exception as e:
                logger.warning(f"Query optimization step failed: {e}")
        
        return optimized
    
    def _convert_implicit_joins(self, query: str) -> str:
        """Convert implicit JOINs to explicit JOINs where possible."""
        # This is a simplified example - in practice, you'd use a SQL parser
        return query
    
    def _add_reasonable_limits(self, query: str) -> str:
        """Add LIMIT clauses to prevent runaway queries."""
        # Add limit if query doesn't have one and might return large result sets
        if "LIMIT" not in query.upper() and "SELECT" in query.upper():
            # Only add limit for queries that might return many rows
            if any(keyword in query.upper() for keyword in ["ORDER BY", "GROUP BY"]):
                query = query.rstrip(';') + " LIMIT 10000"
        
        return query
    
    def _optimize_order_by(self, query: str) -> str:
        """Optimize ORDER BY clauses to use indexes efficiently."""
        # This is simplified - real implementation would analyze index coverage
        return query
    
    async def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table statistics."""
        try:
            stats_query = """
                SELECT 
                    schemaname,
                    tablename,
                    attname as column_name,
                    n_distinct,
                    correlation,
                    most_common_vals,
                    most_common_freqs
                FROM pg_stats 
                WHERE tablename = $1
            """
            
            stats = await self.postgres_client.execute_query(
                stats_query, 
                table_name,
                fetch_mode="all"
            )
            
            # Get table size information
            size_query = """
                SELECT 
                    pg_size_pretty(pg_total_relation_size($1)) as total_size,
                    pg_size_pretty(pg_relation_size($1)) as table_size,
                    pg_size_pretty(pg_total_relation_size($1) - pg_relation_size($1)) as index_size
            """
            
            size_info = await self.postgres_client.execute_query(
                size_query,
                table_name,
                fetch_mode="one"
            )
            
            # Get row count estimate
            count_query = """
                SELECT reltuples::BIGINT as estimated_row_count
                FROM pg_class 
                WHERE relname = $1
            """
            
            count_info = await self.postgres_client.execute_query(
                count_query,
                table_name,
                fetch_mode="one"
            )
            
            return {
                'table_name': table_name,
                'column_statistics': [dict(row) for row in stats] if stats else [],
                'size_info': dict(size_info) if size_info else {},
                'estimated_rows': count_info['estimated_row_count'] if count_info else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get table statistics for {table_name}: {e}")
            return {'error': str(e)}
    
    async def analyze_slow_queries(self, min_duration_ms: float = 1000) -> List[Dict[str, Any]]:
        """Analyze slow queries using pg_stat_statements."""
        try:
            # Enable pg_stat_statements if not already enabled
            await self.postgres_client.execute_query(
                "CREATE EXTENSION IF NOT EXISTS pg_stat_statements",
                fetch_mode="none"
            )
            
            slow_queries_query = """
                SELECT 
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    max_exec_time,
                    stddev_exec_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_exec_time > $1
                ORDER BY mean_exec_time DESC
                LIMIT 20
            """
            
            slow_queries = await self.postgres_client.execute_query(
                slow_queries_query,
                min_duration_ms,
                fetch_mode="all"
            )
            
            return [dict(row) for row in slow_queries] if slow_queries else []
            
        except Exception as e:
            logger.error(f"Failed to analyze slow queries: {e}")
            return []
    
    async def create_partitions_for_time_series(
        self, 
        table_name: str, 
        partition_column: str = "timestamp",
        partition_type: str = "RANGE",
        interval: str = "1 MONTH"
    ) -> Dict[str, Any]:
        """Create time-based partitions for large tables."""
        try:
            # Check if table is already partitioned
            partition_check = """
                SELECT partrelid::regclass as partition_name
                FROM pg_partitioned_table 
                WHERE partrelid = $1::regclass
            """
            
            is_partitioned = await self.postgres_client.execute_query(
                partition_check,
                table_name,
                fetch_mode="one"
            )
            
            if is_partitioned:
                return {
                    'status': 'already_partitioned',
                    'table_name': table_name,
                    'message': 'Table is already partitioned'
                }
            
            # Create partitioned table structure
            # Note: This requires careful planning and data migration
            logger.info(f"Partitioning {table_name} would require data migration")
            
            return {
                'status': 'planning_required',
                'table_name': table_name,
                'message': 'Partitioning requires data migration - manual intervention needed'
            }
            
        except Exception as e:
            logger.error(f"Failed to create partitions for {table_name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def update_table_statistics(self, table_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Update table statistics for query optimization."""
        tables = table_names or ['market_data', 'technical_indicators', 'pattern_detections']
        results = {}
        
        for table in tables:
            try:
                await self.postgres_client.execute_query(
                    f"ANALYZE {table}",
                    fetch_mode="none"
                )
                results[table] = True
                logger.info(f"Updated statistics for table: {table}")
                
            except Exception as e:
                results[table] = False
                logger.error(f"Failed to update statistics for {table}: {e}")
        
        return results
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive database optimization report."""
        try:
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'optimization_history': [
                    {
                        'query_signature': opt.query_signature,
                        'optimization_type': opt.optimization_type,
                        'improvement_ratio': opt.improvement_ratio,
                        'before_execution_time_ms': opt.before_stats.get('execution_time_ms', 0),
                        'after_execution_time_ms': opt.after_stats.get('execution_time_ms', 0)
                    }
                    for opt in self.optimization_history
                ],
                'slow_queries': await self.analyze_slow_queries(),
                'table_statistics': {},
                'index_usage': await self._get_index_usage_stats()
            }
            
            # Get statistics for important tables
            important_tables = ['market_data', 'technical_indicators', 'pattern_detections']
            for table in important_tables:
                report['table_statistics'][table] = await self.get_table_statistics(table)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate optimization report: {e}")
            return {'error': str(e)}
    
    async def _get_index_usage_stats(self) -> Dict[str, Any]:
        """Get index usage statistics."""
        try:
            index_usage_query = """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch,
                    idx_scan
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
            """
            
            index_stats = await self.postgres_client.execute_query(
                index_usage_query,
                fetch_mode="all"
            )
            
            return {
                'index_usage': [dict(row) for row in index_stats] if index_stats else [],
                'unused_indexes': await self._find_unused_indexes()
            }
            
        except Exception as e:
            logger.error(f"Failed to get index usage stats: {e}")
            return {'error': str(e)}
    
    async def _find_unused_indexes(self) -> List[Dict[str, Any]]:
        """Find indexes that are never used."""
        try:
            unused_indexes_query = """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0 AND schemaname = 'public'
                ORDER BY pg_relation_size(indexrelid) DESC
            """
            
            unused_indexes = await self.postgres_client.execute_query(
                unused_indexes_query,
                fetch_mode="all"
            )
            
            return [dict(row) for row in unused_indexes] if unused_indexes else []
            
        except Exception as e:
            logger.error(f"Failed to find unused indexes: {e}")
            return []


# Factory function
def create_database_optimizer(postgres_dsn: str) -> DatabaseOptimizer:
    """Create database optimizer with optimized PostgreSQL client."""
    pool_config = ConnectionPoolConfig(
        min_connections=10,
        max_connections=25,
        max_queries=100000,
        statement_cache_size=200
    )
    
    postgres_client = OptimizedPostgreSQLClient(
        dsn=postgres_dsn,
        pool_config=pool_config,
        enable_query_caching=True
    )
    
    return DatabaseOptimizer(postgres_client)