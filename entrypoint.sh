#!/bin/bash
set -e

# Link nginx logs to stdout/stderr
# ln -sf /dev/stdout /var/log/nginx/access.log
# ln -sf /dev/stderr /var/log/nginx/error.log

# Function to handle shutdown
cleanup() {
    echo "Received shutdown signal, stopping services..."
    # Use SIGQUIT for nginx (more graceful), SIGTERM for Python API
    kill -QUIT "$NGINX_PID" 2>/dev/null || true
    kill -TERM "$PYTHON_PID" 2>/dev/null || true
    wait "$NGINX_PID" 2>/dev/null || true
    wait "$PYTHON_PID" 2>/dev/null || true
    exit 0
}

# Trap both signals
trap cleanup SIGTERM SIGINT

# Start nginx in background
nginx -g "daemon off;" &
NGINX_PID=$!

# Start Python API server in background
cd /app
python server.py &
PYTHON_PID=$!

# Wait for either process to exit
wait -n

# Cleanup if either exits
cleanup