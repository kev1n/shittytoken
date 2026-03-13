"""
ShittyToken web application — user-facing dashboard, auth, billing UI.

Usage:
    python -m shittytoken.web
    shittytoken-web
"""
import asyncio

import structlog
from aiohttp import web

from ..log import configure_logging
from ..config import cfg
from .app import create_web_app


async def _run() -> None:
    configure_logging()
    logger = structlog.get_logger()

    billing_cfg = cfg["gateway"]["billing"]
    postgres_dsn = billing_cfg["postgres_dsn"]
    redis_url = billing_cfg["redis_url"]

    app = await create_web_app(postgres_dsn=postgres_dsn, redis_url=redis_url)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()

    logger.info("web_started", port=8080)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
