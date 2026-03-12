"""
Nginx configuration rendering for the ShittyToken gateway.

All rendering is done via f-string templates — no Jinja2 dependency.
nginx literal braces ({ }) are escaped as {{ and }} in the template.
"""


def render_nginx_config(
    router_port: int = 8001,
    listen_port: int = 80,
    read_timeout_sec: int = 120,
    send_timeout_sec: int = 120,
) -> str:
    """
    Renders a complete nginx.conf string.

    Required directives present in the output:
    - proxy_buffering off
    - add_header X-Accel-Buffering no
    - proxy_read_timeout {read_timeout_sec}s
    - proxy_send_timeout {send_timeout_sec}s
    - proxy_ignore_client_abort off
    - proxy_http_version 1.1
    - keepalive 64 (on upstream block)

    Routes:
      /v1/chat/completions  → vllm_router (full SSE directives)
      /v1/models            → vllm_router (30s timeout, no SSE directives)
      /health               → return 200 "ok\\n" with Content-Type text/plain
    """
    return f"""\
worker_processes auto;

events {{
    worker_connections 4096;
}}

http {{
    upstream vllm_router {{
        server 127.0.0.1:{router_port};
        keepalive 64;
    }}

    log_format main '$remote_addr - $request_time $status "$request" '
                    '$body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'req_id=$http_x_request_id';

    access_log /var/log/nginx/access.log main;
    error_log  /var/log/nginx/error.log warn;

    # Never hold SSE chunks in nginx memory.
    proxy_buffering off;

    server {{
        listen {listen_port};

        # ----------------------------------------------------------------
        # SSE inference endpoint — full streaming directives
        # ----------------------------------------------------------------
        location /v1/chat/completions {{
            proxy_pass http://vllm_router;

            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            # Belt-and-suspenders SSE: nginx layer + downstream layer.
            proxy_buffering off;
            add_header X-Accel-Buffering no;

            proxy_read_timeout {read_timeout_sec}s;
            proxy_send_timeout {send_timeout_sec}s;

            # Forward client disconnects upstream so vLLM can cancel
            # generation and free KV cache immediately.
            proxy_ignore_client_abort off;
        }}

        # ----------------------------------------------------------------
        # Model listing — short timeout, no SSE directives needed
        # ----------------------------------------------------------------
        location /v1/models {{
            proxy_pass http://vllm_router;

            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            proxy_read_timeout 30s;
            proxy_send_timeout 30s;
        }}

        # ----------------------------------------------------------------
        # Health probe — answered locally, no upstream hop
        # ----------------------------------------------------------------
        location /health {{
            add_header Content-Type text/plain;
            return 200 "ok\\n";
        }}
    }}
}}
"""


def write_nginx_config(path: str, **kwargs) -> None:
    """
    Renders the nginx config with the given keyword arguments and writes
    it to *path*.  Accepts the same keyword arguments as render_nginx_config().
    """
    config = render_nginx_config(**kwargs)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(config)
