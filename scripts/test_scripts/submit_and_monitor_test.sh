#!/bin/bash
# Helper script to submit test and monitor logs

echo "Submitting Semantic VAE Integration Test..."
JOB_ID=$(sbatch /usrhomes/s1492/Ctrl-V-seg/scripts/test_scripts/test_semantic_vae_integration.sh | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit job"
    exit 1
fi

echo "✓ Job submitted: $JOB_ID"
echo ""
echo "Monitoring job status..."
echo "Press Ctrl+C to stop monitoring (job will continue running)"
echo ""

LOG_FILE="/no_backups/s1492/Ctrl-V/logs/test_semantic_vae_${JOB_ID}.out"
ERR_FILE="/no_backups/s1492/Ctrl-V/logs/test_semantic_vae_${JOB_ID}.err"

# Wait for log file to be created
while [ ! -f "$LOG_FILE" ]; do
    sleep 2
    echo -n "."
done
echo ""
echo "Log file created: $LOG_FILE"
echo ""

# Tail the log file
tail -f "$LOG_FILE" &
TAIL_PID=$!

# Monitor job status
while true; do
    STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
    
    if [ -z "$STATUS" ]; then
        echo ""
        echo "Job $JOB_ID completed"
        sleep 2
        kill $TAIL_PID 2>/dev/null
        break
    fi
    
    sleep 5
done

echo ""
echo "========================================="
echo "Final Output:"
echo "========================================="
cat "$LOG_FILE"

if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
    echo ""
    echo "========================================="
    echo "Errors (if any):"
    echo "========================================="
    cat "$ERR_FILE"
fi

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo "Output log: $LOG_FILE"
echo "Error log: $ERR_FILE"
