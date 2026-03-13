import uuid
from aiohttp import web
import structlog

logger = structlog.get_logger()


@web.middleware
async def request_id_middleware(request: web.Request, handler) -> web.StreamResponse:
    """
    Assign a request ID from X-Request-ID header (or generate one).
    Store it on the request for downstream use.
    Log request start/end with timing.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request["request_id"] = request_id

    # Don't log /health or /metrics (too noisy)
    skip_logging = request.path in ("/health", "/metrics")

    if not skip_logging:
        logger.info(
            "request_start",
            request_id=request_id,
            method=request.method,
            path=request.path,
        )

    response = await handler(request)

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id

    if not skip_logging:
        logger.info(
            "request_end",
            request_id=request_id,
            method=request.method,
            path=request.path,
            status=response.status,
        )

    return response
