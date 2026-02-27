#!/bin/bash

# Claude Code Proxy 启动脚本
# 功能：关闭已有进程后重新启动

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Claude Code Proxy 启动脚本 ==="

# 查找并关闭已有的代理进程
echo "检查已有进程..."
PIDS=$(pgrep -f "claude-code-proxy|start_proxy.py|uvicorn.*src.main")

if [ -n "$PIDS" ]; then
    echo "发现已有进程: $PIDS"
    echo "正在关闭..."
    kill $PIDS 2>/dev/null
    sleep 1
    # 如果还没关闭，强制关闭
    REMAINING=$(pgrep -f "claude-code-proxy|start_proxy.py|uvicorn.*src.main")
    if [ -n "$REMAINING" ]; then
        echo "强制关闭进程: $REMAINING"
        kill -9 $REMAINING 2>/dev/null
    fi
    echo "已关闭旧进程"
else
    echo "没有发现已有进程"
fi

# 环境变量配置
export REQUEST_TIMEOUT=300        # 上游API超时(秒)，GLM-4.7响应较慢建议300
export LOG_LEVEL=INFO              # 日志级别: DEBUG/INFO/WARNING/ERROR

# 启动新进程
echo "启动代理服务器..."
echo "  REQUEST_TIMEOUT=${REQUEST_TIMEOUT}s, LOG_LEVEL=${LOG_LEVEL}"
if command -v uv &> /dev/null; then
    uv run claude-code-proxy
else
    python start_proxy.py
fi
