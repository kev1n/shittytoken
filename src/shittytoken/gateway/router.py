"""
Custom router entry point.

Usage:
    python -m shittytoken.gateway.router

Or via the installed entry point:
    shittytoken-router
"""
import asyncio
import structlog
from aiohttp import web
from ..log import configure_logging
from ..config import cfg
from .router_app import create_router_app


async def _run() -> None:
    configure_logging()
    logger = structlog.get_logger()

    port = cfg["gateway"]["router"]["port"]  # 8001
    admin_token = cfg["gateway"]["router"].get("admin_token")

    app = await create_router_app(admin_token=admin_token)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info("router_started", port=port)

    # Run until interrupted
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
