"""
Route handlers for the ShittyToken web application.
"""
from __future__ import annotations

import hashlib
import secrets
from functools import wraps
from typing import Any

import aiohttp_jinja2
import bcrypt
import structlog
import stripe
from aiohttp import web
from aiohttp_session import get_session

from ..billing.postgres import BillingPostgres
from ..billing.redis_cache import BillingRedis
from ..config import cfg

logger = structlog.get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _flash(session: Any, category: str, message: str) -> None:
    """Store a flash message in the session."""
    flashes = session.setdefault("_flashes", [])
    flashes.append({"category": category, "message": message})


def _pop_flashes(session: Any) -> list[dict]:
    """Retrieve and clear flash messages."""
    flashes = session.pop("_flashes", [])
    return flashes


def _billing_pg(request: web.Request) -> BillingPostgres:
    return request.app["billing_pg"]


def _billing_redis(request: web.Request) -> BillingRedis:
    return request.app["billing_redis"]


def _gateway_base_url() -> str:
    """Return the public gateway base URL for API usage."""
    port = cfg["gateway"]["router"]["port"]
    return f"http://localhost:{port}"


def login_required(handler):
    """Decorator: redirect to /login if not authenticated."""
    @wraps(handler)
    async def wrapper(request: web.Request) -> web.StreamResponse:
        session = await get_session(request)
        if "user_id" not in session:
            raise web.HTTPFound("/login")
        return await handler(request)
    return wrapper


# ── Public pages ─────────────────────────────────────────────────────────

@aiohttp_jinja2.template("index.html")
async def index(request: web.Request) -> dict:
    session = await get_session(request)
    return {
        "user_id": session.get("user_id"),
        "flashes": _pop_flashes(session),
    }


@aiohttp_jinja2.template("signup.html")
async def signup_form(request: web.Request) -> dict:
    session = await get_session(request)
    return {"flashes": _pop_flashes(session)}


async def signup_submit(request: web.Request) -> web.Response:
    session = await get_session(request)
    data = await request.post()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    password_confirm = data.get("password_confirm", "")

    if not email or not password:
        _flash(session, "error", "Email and password are required.")
        raise web.HTTPFound("/signup")

    if password != password_confirm:
        _flash(session, "error", "Passwords do not match.")
        raise web.HTTPFound("/signup")

    if len(password) < 8:
        _flash(session, "error", "Password must be at least 8 characters.")
        raise web.HTTPFound("/signup")

    pg = _billing_pg(request)

    # Check if user exists
    existing = await pg.get_user_by_email(email)
    if existing:
        _flash(session, "error", "An account with that email already exists.")
        raise web.HTTPFound("/signup")

    # Hash password and create user
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user = await pg.create_user_with_password(email, pw_hash)

    # Log in immediately
    session["user_id"] = user.id
    session["email"] = user.email
    _flash(session, "success", "Account created! Welcome to ShittyToken.")
    raise web.HTTPFound("/dashboard")


@aiohttp_jinja2.template("login.html")
async def login_form(request: web.Request) -> dict:
    session = await get_session(request)
    return {"flashes": _pop_flashes(session)}


async def login_submit(request: web.Request) -> web.Response:
    session = await get_session(request)
    data = await request.post()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        _flash(session, "error", "Email and password are required.")
        raise web.HTTPFound("/login")

    pg = _billing_pg(request)
    result = await pg.get_user_by_email_with_password(email)

    if result is None:
        _flash(session, "error", "Invalid email or password.")
        raise web.HTTPFound("/login")

    user, pw_hash = result
    if not pw_hash or not bcrypt.checkpw(password.encode(), pw_hash.encode()):
        _flash(session, "error", "Invalid email or password.")
        raise web.HTTPFound("/login")

    session["user_id"] = user.id
    session["email"] = user.email
    _flash(session, "success", "Logged in.")
    raise web.HTTPFound("/dashboard")


async def logout(request: web.Request) -> web.Response:
    session = await get_session(request)
    session.invalidate()
    raise web.HTTPFound("/")


# ── Dashboard ────────────────────────────────────────────────────────────

@login_required
@aiohttp_jinja2.template("dashboard.html")
async def dashboard(request: web.Request) -> dict:
    session = await get_session(request)
    user_id = session["user_id"]
    pg = _billing_pg(request)

    balance = await pg.get_balance(user_id)
    api_keys = await pg.list_api_keys_for_user(user_id)
    recent_usage = await pg.get_recent_usage(user_id, limit=20)

    # Convert usage records to dicts for template
    usage_list = []
    for row in recent_usage:
        usage_list.append({
            "model": row["model"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "total_tokens": row["total_tokens"],
            "cost_cents": float(row["cost_cents"]),
            "created_at": row["created_at"],
        })

    return {
        "user_id": user_id,
        "email": session.get("email", ""),
        "balance_cents": balance,
        "balance_dollars": balance / 100,
        "api_keys": api_keys,
        "recent_usage": usage_list,
        "gateway_base_url": _gateway_base_url(),
        "flashes": _pop_flashes(session),
        "new_api_key": session.pop("new_api_key", None),
    }


# ── API Key management ───────────────────────────────────────────────────

@login_required
async def create_api_key(request: web.Request) -> web.Response:
    session = await get_session(request)
    user_id = session["user_id"]
    data = await request.post()
    name = data.get("name", "").strip() or None

    pg = _billing_pg(request)

    # Generate a random API key: sk-st-<random hex>
    plaintext = f"sk-st-{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()

    await pg.create_api_key(key_hash=key_hash, user_id=user_id, name=name)

    # Store plaintext in session so dashboard can show it ONCE
    session["new_api_key"] = plaintext
    _flash(session, "success", "API key created. Copy it now — you won't see it again!")
    raise web.HTTPFound("/dashboard")


@login_required
async def revoke_api_key(request: web.Request) -> web.Response:
    session = await get_session(request)
    key_hash = request.match_info["key_hash"]

    pg = _billing_pg(request)
    redis = _billing_redis(request)

    await pg.deactivate_api_key(key_hash)
    await redis.invalidate_api_key(key_hash)

    _flash(session, "success", "API key revoked.")
    raise web.HTTPFound("/dashboard")


# ── Billing ──────────────────────────────────────────────────────────────

@login_required
@aiohttp_jinja2.template("billing.html")
async def billing_page(request: web.Request) -> dict:
    session = await get_session(request)
    user_id = session["user_id"]
    pg = _billing_pg(request)

    balance = await pg.get_balance(user_id)
    blocks = await pg.get_active_blocks(user_id)
    ledger = await pg.get_ledger(user_id, limit=50)

    return {
        "user_id": user_id,
        "email": session.get("email", ""),
        "balance_cents": balance,
        "balance_dollars": balance / 100,
        "credit_blocks": blocks,
        "ledger_events": ledger,
        "flashes": _pop_flashes(session),
    }


@login_required
async def billing_topup(request: web.Request) -> web.Response:
    session = await get_session(request)
    user_id = session["user_id"]
    data = await request.post()

    try:
        amount_dollars = int(data.get("amount", 0))
    except (TypeError, ValueError):
        _flash(session, "error", "Invalid amount.")
        raise web.HTTPFound("/billing")

    if amount_dollars not in (5, 10, 25, 50):
        _flash(session, "error", "Invalid top-up amount.")
        raise web.HTTPFound("/billing")

    settings = request.app["settings"]
    if not settings.stripe_secret_key:
        _flash(session, "error", "Stripe is not configured.")
        raise web.HTTPFound("/billing")

    stripe.api_key = settings.stripe_secret_key

    # Build success/cancel URLs
    host = request.host
    scheme = request.scheme
    base = f"{scheme}://{host}"

    checkout_session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": f"ShittyToken Credits - ${amount_dollars}",
                    "description": f"${amount_dollars} in API credits",
                },
                "unit_amount": amount_dollars * 100,
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{base}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/billing/cancel",
        metadata={
            "user_id": user_id,
            "amount_cents": str(amount_dollars * 100),
        },
    )

    raise web.HTTPFound(checkout_session.url)


@login_required
async def billing_success(request: web.Request) -> web.Response:
    """Handle Stripe checkout success redirect.

    Credit fulfillment is handled by the Stripe webhook (stripe_webhook.py),
    NOT here.  This endpoint only verifies the session is valid and shows a
    confirmation message.  This prevents double-crediting from URL replays.
    """
    session = await get_session(request)
    user_id = session["user_id"]
    checkout_session_id = request.query.get("session_id")

    if not checkout_session_id:
        _flash(session, "error", "Invalid callback.")
        raise web.HTTPFound("/billing")

    settings = request.app["settings"]
    stripe.api_key = settings.stripe_secret_key

    try:
        cs = stripe.checkout.Session.retrieve(checkout_session_id)
    except stripe.StripeError as e:
        logger.error("stripe_session_retrieve_failed", error=str(e))
        _flash(session, "error", "Could not verify payment.")
        raise web.HTTPFound("/billing")

    if cs.payment_status != "paid":
        _flash(session, "error", "Payment not completed.")
        raise web.HTTPFound("/billing")

    meta_user_id = cs.metadata.get("user_id", "")
    amount_cents = int(cs.metadata.get("amount_cents", "0"))

    if meta_user_id != user_id or amount_cents <= 0:
        _flash(session, "error", "Payment verification failed.")
        raise web.HTTPFound("/billing")

    _flash(session, "success", f"Payment received! ${amount_cents / 100:.2f} will be added to your balance shortly.")
    raise web.HTTPFound("/billing")


@login_required
async def billing_cancel(request: web.Request) -> web.Response:
    session = await get_session(request)
    _flash(session, "info", "Payment cancelled. No charges were made.")
    raise web.HTTPFound("/billing")


# ── Route setup ──────────────────────────────────────────────────────────

def setup_routes(app: web.Application) -> None:
    app.router.add_get("/", index)
    app.router.add_get("/signup", signup_form)
    app.router.add_post("/signup", signup_submit)
    app.router.add_get("/login", login_form)
    app.router.add_post("/login", login_submit)
    app.router.add_post("/logout", logout)
    app.router.add_get("/dashboard", dashboard)
    app.router.add_post("/api-keys/create", create_api_key)
    app.router.add_post("/api-keys/{key_hash}/revoke", revoke_api_key)
    app.router.add_get("/billing", billing_page)
    app.router.add_post("/billing/topup", billing_topup)
    app.router.add_get("/billing/success", billing_success)
    app.router.add_get("/billing/cancel", billing_cancel)
