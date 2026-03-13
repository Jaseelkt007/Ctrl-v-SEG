#!/bin/bash
# Frontend startup script that automatically detects backend server

# Check if backend job is running and get the node
BACKEND_JOB=$(squeue -u s1492 -n ctrlv_api -h -o "%N %i" | head -n1)

if [ -z "$BACKEND_JOB" ]; then
    echo "Warning: No backend job found running. Using default backend URL."
    BACKEND_URL="http://129.69.32.93:8000"
else
    NODE=$(echo $BACKEND_JOB | awk '{print $1}')
    JOB_ID=$(echo $BACKEND_JOB | awk '{print $2}')
    echo "Found backend running on node: $NODE (Job ID: $JOB_ID)"
    
    # Get IP address of the node
    NODE_IP=$(host $NODE | awk '/has address/ { print $4 }' | head -n1)
    
    if [ -z "$NODE_IP" ]; then
        echo "Warning: Could not resolve IP for $NODE. Using default."
        BACKEND_URL="http://129.69.32.93:8000"
    else
        BACKEND_URL="http://${NODE_IP}:8000"
        echo "Backend URL: $BACKEND_URL"
    fi
fi

# Export and run frontend
export BACKEND_URL
cd "$(dirname "$0")"
npm run dev -- -p 3000 -H 0.0.0.0
