"""
Structured JSON Request Logging Middleware
SaaS Revenue Intelligence API - Week 5

Logs per request:
    timestamp, method, endpoint,
    status_code, latency_ms, client_ip
"""

import json
import time
import logging
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Emits one structured JSON log line per HTTP request.
    Compatible with Grafana, ELK Stack, Datadog, CloudWatch.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        # Process request
        response = await call_next(request)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Structured log record
        log_record = {
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "method":      request.method,
            "endpoint":    str(request.url.path),
            "query":       str(request.url.query) if request.url.query else None,
            "status_code": response.status_code,
            "latency_ms":  latency_ms,
            "client_ip":   self._get_client_ip(request),
        }

        # Remove None values for clean output
        log_record = {k: v for k, v in log_record.items() if v is not None}

        logger.info(json.dumps(log_record))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP, handling proxies."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
