#!/bin/bash

###############################################################################
# Quick check of test results
# Shows file sizes of predicted semantic images
###############################################################################

OUTPUT_DIR="${1:-/no_backups/s1492/Ctrl-V/outputs/kitti360_sem_instant_test}"

echo "=========================================="
echo "Checking test results"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

IMAGE_DIR="$OUTPUT_DIR/wandb/latest-run/files/media/images"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Image directory not found: $IMAGE_DIR"
    echo ""
    echo "Have you run the test yet?"
    echo "  ./scripts/eval_scripts/quick_test_single_clip.sh"
    exit 1
fi

cd "$IMAGE_DIR"

echo "Predicted semantic images (*_pred_sem.png):"
echo ""
ls -lh *_pred_sem.png* 2>/dev/null | head -30

echo ""
echo "=========================================="
echo "Quick Analysis:"
echo "=========================================="

# Count files
TOTAL=$(ls *_pred_sem.png* 2>/dev/null | wc -l)
BLACK=$(find . -name "*_pred_sem.png*" -size 271c 2>/dev/null | wc -l)
GOOD=$((TOTAL - BLACK))

echo "Total _pred_sem.png files: $TOTAL"
echo "Black frames (271 bytes):  $BLACK"
echo "Good frames (>10KB):       $GOOD"
echo ""

if [ $BLACK -eq 0 ]; then
    echo "✅ SUCCESS! No black frames detected!"
    echo "   The fix is working correctly."
    echo ""
    echo "You can now run the full evaluation:"
    echo "  sbatch scripts/eval_scripts/eval_kitti360_sem_overall.sh"
elif [ $BLACK -lt $((TOTAL / 10)) ]; then
    echo "⚠️  Few black frames detected ($BLACK out of $TOTAL)"
    echo "   This might be acceptable for edge cases."
else
    echo "❌ PROBLEM: Many black frames still detected!"
    echo "   The fix may not be working correctly."
    echo ""
    echo "Please check:"
    echo "  1. Is the fix applied in tools/eval_overall.py?"
    echo "  2. Did you activate the correct conda environment?"
    echo "  3. Are you using the updated code?"
fi

echo ""
echo "For detailed analysis, run:"
echo "  python /usrhomes/s1492/Ctrl-V-seg/scripts/eval_scripts/verify_fix.py $OUTPUT_DIR"
echo ""
