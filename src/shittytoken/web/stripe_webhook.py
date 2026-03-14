"""
Stripe webhook handler — processes checkout.session.completed events.
"""
from __future__ import annotations

import structlog
import stripe
from aiohttp import web

from ..billing.postgres import BillingPostgres
from ..billing.redis_cache import BillingRedis

logger = structlog.get_logger(__name__)


async def stripe_webhook(request: web.Request) -> web.Response:
    """Handle Stripe webhook events.

    POST /webhook/stripe
    Verifies the webhook signature, then processes checkout.session.completed
    events by adding credit blocks to the user's account.
    """
    settings = request.app["settings"]
    payload = await request.read()
    sig_header = request.headers.get("Stripe-Signature", "")

    if not settings.stripe_webhook_secret:
        logger.warning("stripe_webhook.no_secret_configured")
        return web.json_response({"error": "Webhook not configured"}, status=500)

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except ValueError:
        logger.warning("stripe_webhook.invalid_payload")
        return web.json_response({"error": "Invalid payload"}, status=400)
    except stripe.SignatureVerificationError:
        logger.warning("stripe_webhook.invalid_signature")
        return web.json_response({"error": "Invalid signature"}, status=400)

    if event["type"] == "checkout.session.completed":
        await _handle_checkout_completed(request, event["data"]["object"])
    else:
        logger.debug("stripe_webhook.unhandled_event", event_type=event["type"])

    return web.json_response({"status": "ok"})


async def _handle_checkout_completed(
    request: web.Request, session: dict
) -> None:
    """Process a completed checkout session: add credits to user account."""
    metadata = session.get("metadata", {})
    user_id = metadata.get("user_id")
    amount_cents_str = metadata.get("amount_cents")

    if not user_id or not amount_cents_str:
        logger.error(
            "stripe_webhook.missing_metadata",
            session_id=session.get("id"),
        )
        return

    amount_cents = int(amount_cents_str)
    payment_intent_id = session.get("payment_intent")

    pg: BillingPostgres = request.app["billing_pg"]
    redis: BillingRedis = request.app["billing_redis"]

    # Idempotency: check ALL blocks (including exhausted/expired)
    already_credited = await pg.has_credit_block_for_payment(payment_intent_id)

    if already_credited:
        logger.info(
            "stripe_webhook.already_credited",
            user_id=user_id,
            payment_intent_id=payment_intent_id,
        )
        return

    # Add credit block
    await pg.create_credit_block(
        user_id=user_id,
        amount_cents=amount_cents,
        source="stripe_checkout",
        stripe_payment_intent_id=payment_intent_id,
    )

    # Update Redis balance cache
    new_balance = await pg.get_balance(user_id)
    await redis.set_balance(user_id, new_balance)

    logger.info(
        "stripe_webhook.credits_added",
        user_id=user_id,
        amount_cents=amount_cents,
        new_balance=new_balance,
    )


def setup_webhook_routes(app: web.Application) -> None:
    app.router.add_post("/webhook/stripe", stripe_webhook)
