"""
Billing package — credit blocks, ledger, Redis hot-path, usage pipeline.

PostgreSQL is the system of record (append-only ledger, credit blocks).
Redis provides sub-millisecond balance checks and rate limiting on the hot path.
Periodic reconciliation catches any drift between the two.
"""
