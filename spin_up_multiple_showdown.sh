#!/bin/bash

# Configuration
START_PORT=8000
N_SERVERS=4
SHOWDOWN_DIR="./pokemon-showdown"

# Check for node
if ! command -v node &> /dev/null; then
    echo "❌ Error: 'node' command not found. Please ensure Node.js is installed."
    exit 1
fi

if [ ! -d "$SHOWDOWN_DIR" ]; then
    echo "❌ Error: Directory $SHOWDOWN_DIR not found."
    exit 1
fi

echo "Starting $N_SERVERS Pokemon Showdown servers (Ports $START_PORT-$((START_PORT + N_SERVERS - 1)))..."

pids=()

for ((i=0; i<N_SERVERS; i++)); do
    port=$((START_PORT + i))
    
    # Check if port is open (simple check using timeout + bash tcp)
    if (echo > /dev/tcp/localhost/$port) >/dev/null 2>&1; then
        echo "⚠️  Port $port is already in use. Skipping..."
        continue
    fi
    
    echo "   Starting server on port $port..."
    
    # Start in background, logging to file
    (cd "$SHOWDOWN_DIR" && node pokemon-showdown $port > "../showdown_server_$port.log" 2>&1) &
    pids+=($!)
done

if [ ${#pids[@]} -eq 0 ]; then
    echo "No new servers started."
else
    echo "Servers started with PIDs: ${pids[*]}"
    echo "(Logs are in showdown_server_PORT.log)"
    echo "Use 'pkill -f pokemon-showdown' to stop them later."
fi
