"""
Unit tests for gateway/nginx_config.py.

Pure string-level assertions — no filesystem access, no network.
"""

import pytest

from shittytoken.gateway.nginx_config import render_nginx_config


class TestRenderNginxConfig:
    """Tests for render_nginx_config() output correctness."""

    def test_proxy_buffering_off(self):
        cfg = render_nginx_config()
        assert "proxy_buffering off" in cfg

    def test_x_accel_buffering_no(self):
        cfg = render_nginx_config()
        assert "add_header X-Accel-Buffering no" in cfg

    def test_proxy_ignore_client_abort_off(self):
        cfg = render_nginx_config()
        assert "proxy_ignore_client_abort off" in cfg

    def test_default_read_timeout(self):
        cfg = render_nginx_config()
        assert "proxy_read_timeout 120s" in cfg

    def test_default_send_timeout(self):
        cfg = render_nginx_config()
        assert "proxy_send_timeout 120s" in cfg

    def test_keepalive_64(self):
        cfg = render_nginx_config()
        assert "keepalive 64" in cfg

    def test_health_return_200(self):
        cfg = render_nginx_config()
        assert "return 200" in cfg

    def test_custom_read_timeout(self):
        cfg = render_nginx_config(read_timeout_sec=60)
        assert "proxy_read_timeout 60s" in cfg
        # Default (120s) must NOT appear in the timeout directive for read
        # (send timeout is still 120 by default, so we test the read line specifically).
        assert "proxy_read_timeout 120s" not in cfg

    def test_custom_send_timeout(self):
        cfg = render_nginx_config(send_timeout_sec=90)
        assert "proxy_send_timeout 90s" in cfg
        assert "proxy_send_timeout 120s" not in cfg

    def test_custom_router_port(self):
        cfg = render_nginx_config(router_port=9999)
        assert "127.0.0.1:9999" in cfg

    def test_custom_listen_port(self):
        cfg = render_nginx_config(listen_port=8080)
        assert "listen 8080" in cfg

    def test_proxy_http_version_11(self):
        cfg = render_nginx_config()
        assert "proxy_http_version 1.1" in cfg

    def test_completions_location_present(self):
        cfg = render_nginx_config()
        assert "location /v1/chat/completions" in cfg

    def test_models_location_present(self):
        cfg = render_nginx_config()
        assert "location /v1/models" in cfg

    def test_health_location_present(self):
        cfg = render_nginx_config()
        assert "location /health" in cfg

    def test_upstream_block_present(self):
        cfg = render_nginx_config()
        assert "upstream vllm_router" in cfg

    def test_models_timeout_30s(self):
        # /v1/models uses a 30 s timeout regardless of the SSE timeout setting.
        cfg = render_nginx_config(read_timeout_sec=120, send_timeout_sec=120)
        assert "proxy_read_timeout 30s" in cfg

    def test_content_type_text_plain_for_health(self):
        cfg = render_nginx_config()
        assert "Content-Type text/plain" in cfg
