#!/usr/bin/env bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit",
    "messages": [{"role": "user", "content": "What is the meaning of life?"}],
    "max_tokens": 10000
  }'
