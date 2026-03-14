#!/bin/bash
# Binary search for max gpu_memory_utilization on a running Vast.ai instance.
# Usage: ./gpu_util_sweep.sh <ssh_host> <ssh_port>
#
# Kills vLLM, restarts with new value, sends a test request, checks if it crashed.

SSH_HOST="${1:-ssh5.vast.ai}"
SSH_PORT="${2:-29946}"
VLLM_PORT=8080

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@${SSH_HOST}" -p "$SSH_PORT" "$@" 2>/dev/null
}

# Get the base vllm command (without gpu_memory_utilization)
BASE_CMD=$(ssh_cmd "ps aux | grep 'vllm serve' | grep -v grep | head -1 | sed 's/.*vllm serve/vllm serve/' | sed 's/--gpu-memory-utilization [0-9.]*//'")
echo "Base command: $BASE_CMD"

test_util() {
    local val=$1
    echo ""
    echo "=== Testing gpu_memory_utilization=$val ==="

    # Kill existing vLLM
    ssh_cmd "pkill -f 'vllm serve' ; sleep 2 ; pkill -9 -f 'vllm serve' 2>/dev/null; sleep 1"

    # Start vLLM with new value (background, redirect to log)
    ssh_cmd "nohup $BASE_CMD --gpu-memory-utilization $val > /tmp/vllm_test.log 2>&1 &"

    # Wait for startup (poll /health)
    echo "  Waiting for vLLM to start..."
    for i in $(seq 1 60); do
        sleep 5
        status=$(ssh_cmd "curl -s -o /dev/null -w '%{http_code}' http://localhost:${VLLM_PORT}/health" 2>/dev/null)
        if [ "$status" = "200" ]; then
            echo "  vLLM healthy after ${i}x5s"

            # Check KV cache size
            kv_info=$(ssh_cmd "grep 'GPU KV cache size' /tmp/vllm_test.log | tail -1")
            mem_info=$(ssh_cmd "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1")
            echo "  $kv_info"
            echo "  GPU memory: $mem_info"

            # Send a test request to make sure it actually works
            echo "  Sending test request..."
            test_result=$(ssh_cmd "curl -s -X POST http://localhost:${VLLM_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":5}' | head -c 200")
            if echo "$test_result" | grep -q '"choices"'; then
                echo "  ✓ PASS: gpu_memory_utilization=$val works"
                return 0
            else
                echo "  ✗ FAIL: request failed: $test_result"
                return 1
            fi
        fi

        # Check if vLLM crashed during startup
        crashed=$(ssh_cmd "grep -c 'Error\|OOM\|CUDA out of memory\|RuntimeError' /tmp/vllm_test.log 2>/dev/null")
        if [ "$crashed" -gt 0 ] 2>/dev/null; then
            echo "  ✗ CRASH during startup at $val"
            ssh_cmd "grep -E 'Error|OOM|CUDA|RuntimeError' /tmp/vllm_test.log | tail -3"
            return 1
        fi
    done

    echo "  ✗ TIMEOUT: vLLM didn't start in 300s at $val"
    return 1
}

# Binary search between 0.90 and 0.99
# We know 0.90 works
lo=90
hi=99

echo "Binary search: gpu_memory_utilization 0.${lo} to 0.${hi}"
echo ""

results=""

while [ $((hi - lo)) -gt 1 ]; do
    mid=$(( (lo + hi) / 2 ))
    val="0.${mid}"

    if test_util "$val"; then
        lo=$mid
        results="${results}  0.${mid}: PASS\n"
    else
        hi=$mid
        results="${results}  0.${mid}: FAIL\n"
    fi
    echo "  Bounds: lo=0.${lo}, hi=0.${hi}"
done

# Test the boundary
test_util "0.${hi}"
if [ $? -eq 0 ]; then
    results="${results}  0.${hi}: PASS\n"
    lo=$hi
else
    results="${results}  0.${hi}: FAIL\n"
fi

echo ""
echo "================================================"
echo "RESULT: max gpu_memory_utilization = 0.${lo}"
echo ""
echo "All results:"
echo -e "$results"
echo ""

# Restore to the winning value
echo "Restoring vLLM with gpu_memory_utilization=0.${lo}..."
test_util "0.${lo}"
