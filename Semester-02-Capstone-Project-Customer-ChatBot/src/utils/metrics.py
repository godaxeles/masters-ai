"""Metrics collection and monitoring utilities."""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    timestamp: datetime
    processing_time: float
    retrieval_time: float
    generation_time: float
    retrieved_docs: int
    max_similarity_score: float
    response_length: int
    citation_count: int
    user_satisfied: Optional[bool] = None


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    total_queries: int = 0
    avg_processing_time: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_similarity_score: float = 0.0
    ticket_creation_rate: float = 0.0
    user_satisfaction_rate: Optional[float] = None
    error_rate: float = 0.0
    query_history: List[QueryMetrics] = field(default_factory=list)


class MetricsCollector:
    """Collect and analyze system metrics."""

    def __init__(self):
        self.metrics = SystemMetrics()
        self._errors = 0

    def record_query(self, query_metrics: QueryMetrics) -> None:
        """Record metrics for a query.

        Args:
            query_metrics: Metrics for the query
        """
        self.metrics.query_history.append(query_metrics)
        self.metrics.total_queries += 1

        # Update averages
        self._update_averages()

        logger.info(
            "Query processed",
            query=query_metrics.query[:50] + "..." if len(query_metrics.query) > 50 else query_metrics.query,
            processing_time=query_metrics.processing_time,
            retrieved_docs=query_metrics.retrieved_docs,
            max_similarity=query_metrics.max_similarity_score
        )

    def record_error(self, error_type: str, error_message: str) -> None:
        """Record an error.

        Args:
            error_type: Type of error
            error_message: Error message
        """
        self._errors += 1
        self.metrics.error_rate = self._errors / max(self.metrics.total_queries, 1)

        logger.error(
            "Error recorded",
            error_type=error_type,
            error_message=error_message,
            error_rate=self.metrics.error_rate
        )

    def record_ticket_creation(self) -> None:
        """Record when a support ticket is created."""
        tickets_created = sum(
            1 for q in self.metrics.query_history
            if q.user_satisfied is False
        )
        self.metrics.ticket_creation_rate = tickets_created / max(self.metrics.total_queries, 1)

        logger.info(
            "Ticket created",
            ticket_creation_rate=self.metrics.ticket_creation_rate
        )

    def record_user_feedback(self, query_id: int, satisfied: bool) -> None:
        """Record user satisfaction feedback.

        Args:
            query_id: Index of the query in history
            satisfied: Whether user was satisfied
        """
        if 0 <= query_id < len(self.metrics.query_history):
            self.metrics.query_history[query_id].user_satisfied = satisfied

            # Update satisfaction rate
            satisfied_queries = [
                q for q in self.metrics.query_history
                if q.user_satisfied is not None
            ]
            if satisfied_queries:
                self.metrics.user_satisfaction_rate = (
                    sum(1 for q in satisfied_queries if q.user_satisfied) /
                    len(satisfied_queries)
                )

            logger.info(
                "User feedback recorded",
                satisfied=satisfied,
                satisfaction_rate=self.metrics.user_satisfaction_rate
            )

    def get_metrics_summary(self) -> Dict:
        """Get a summary of current metrics.

        Returns:
            Dictionary with metrics summary
        """
        recent_queries = self.metrics.query_history[-100:] if self.metrics.query_history else []

        return {
            "total_queries": self.metrics.total_queries,
            "avg_processing_time": round(self.metrics.avg_processing_time, 3),
            "avg_retrieval_time": round(self.metrics.avg_retrieval_time, 3),
            "avg_generation_time": round(self.metrics.avg_generation_time, 3),
            "avg_similarity_score": round(self.metrics.avg_similarity_score, 3),
            "ticket_creation_rate": round(self.metrics.ticket_creation_rate * 100, 1),
            "user_satisfaction_rate": (
                round(self.metrics.user_satisfaction_rate * 100, 1)
                if self.metrics.user_satisfaction_rate is not None else None
            ),
            "error_rate": round(self.metrics.error_rate * 100, 1),
            "recent_avg_processing_time": (
                round(sum(q.processing_time for q in recent_queries) / len(recent_queries), 3)
                if recent_queries else 0
            )
        }

    def _update_averages(self) -> None:
        """Update running averages."""
        if not self.metrics.query_history:
            return

        queries = self.metrics.query_history
        n = len(queries)

        self.metrics.avg_processing_time = sum(q.processing_time for q in queries) / n
        self.metrics.avg_retrieval_time = sum(q.retrieval_time for q in queries) / n
        self.metrics.avg_generation_time = sum(q.generation_time for q in queries) / n
        self.metrics.avg_similarity_score = sum(q.max_similarity_score for q in queries) / n

    @contextmanager
    def timing_context(self, operation: str):
        """Context manager for timing operations.

        Args:
            operation: Name of the operation being timed

        Yields:
            Dictionary to store timing results
        """
        start_time = time.time()
        timing_data = {}

        try:
            yield timing_data
        finally:
            timing_data[operation] = time.time() - start_time


# Global metrics collector
metrics_collector = MetricsCollector()