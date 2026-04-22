#!/bin/bash
# Test script to verify receiver handles delayed transmission starts correctly
#
# This script tests the fix for the issue where decoding degrades when
# transmission starts long after receiver launch.
#
# Usage: ./test_delayed_start.sh [delay_seconds]

set -e

DELAY=${1:-60}  # Default: 60 second delay
TEST_MESSAGE="Hello World! This is a test message to verify the delayed start fix works correctly. 123456789"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "==================================="
echo "Delayed Start Test"
echo "==================================="
echo "Test message: $TEST_MESSAGE"
echo "Delay before transmission: ${DELAY} seconds"
echo ""

# Test 1: Quick start (baseline)
echo "[Test 1] Quick start (baseline)..."
OUTPUT_FILE="$TEMP_DIR/quick_output.txt"

# Launch receiver in background with file output
./receive --device 16 -m 4 -e -d 32 -o "$OUTPUT_FILE" > /dev/null 2>&1 &
RECEIVER_PID=$!

# Small delay to ensure receiver is ready
sleep 2

# Start transmission immediately
echo "  Starting transmission (quick)..."
echo "$TEST_MESSAGE" | ./transmit -m 4 -e -d 32 > /dev/null 2>&1

# Wait for transmission to complete
sleep 8

# Stop receiver
kill $RECEIVER_PID 2>/dev/null || true
wait $RECEIVER_PID 2>/dev/null || true

# Check output
if [ -f "$OUTPUT_FILE" ]; then
    QUICK_OUTPUT=$(cat "$OUTPUT_FILE")
    echo "  Decoded: $QUICK_OUTPUT"
else
    echo "  ERROR: No output file generated"
    QUICK_OUTPUT=""
fi

# Test 2: Delayed start
echo ""
echo "[Test 2] Delayed start (${DELAY}s wait)..."
OUTPUT_FILE="$TEMP_DIR/delayed_output.txt"

# Launch receiver in background with file output
./receive --device 16 -m 4 -e -d 32 -o "$OUTPUT_FILE" > /dev/null 2>&1 &
RECEIVER_PID=$!

# Wait for the specified delay
echo "  Receiver launched. Waiting ${DELAY} seconds..."
sleep "$DELAY"

# Start transmission after delay
echo "  Starting transmission (delayed)..."
echo "$TEST_MESSAGE" | ./transmit -m 4 -e -d 32 > /dev/null 2>&1

# Wait for transmission to complete
sleep 8

# Stop receiver
kill $RECEIVER_PID 2>/dev/null || true
wait $RECEIVER_PID 2>/dev/null || true

# Check output
if [ -f "$OUTPUT_FILE" ]; then
    DELAYED_OUTPUT=$(cat "$OUTPUT_FILE")
    echo "  Decoded: $DELAYED_OUTPUT"
else
    echo "  ERROR: No output file generated"
    DELAYED_OUTPUT=""
fi

# Compare results
echo ""
echo "==================================="
echo "Results Summary"
echo "==================================="
echo "Expected:  $TEST_MESSAGE"
echo "Quick:     $QUICK_OUTPUT"
echo "Delayed:   $DELAYED_OUTPUT"
echo ""

# Simple quality check (count matching characters)
count_matches() {
    local expected="$1"
    local actual="$2"
    local matches=0
    local total=${#expected}

    for ((i=0; i<${#expected} && i<${#actual}; i++)); do
        if [ "${expected:$i:1}" = "${actual:$i:1}" ]; then
            ((matches++))
        fi
    done

    echo $((matches * 100 / total))
}

if [ -n "$QUICK_OUTPUT" ] && [ -n "$DELAYED_OUTPUT" ]; then
    QUICK_MATCH=$(count_matches "$TEST_MESSAGE" "$QUICK_OUTPUT")
    DELAYED_MATCH=$(count_matches "$TEST_MESSAGE" "$DELAYED_OUTPUT")

    echo "Quick start accuracy:   ${QUICK_MATCH}%"
    echo "Delayed start accuracy: ${DELAYED_MATCH}%"
    echo ""

    if [ "$DELAYED_MATCH" -ge 90 ]; then
        echo "✓ PASS: Delayed start decoded successfully (≥90% match)"
        exit 0
    elif [ "$DELAYED_MATCH" -ge "$QUICK_MATCH" ]; then
        echo "⚠ PARTIAL: Delayed matches quick start but accuracy is low"
        exit 0
    else
        echo "✗ FAIL: Delayed start significantly worse than quick start"
        echo "  This suggests the state reset fix may not be working correctly"
        exit 1
    fi
else
    echo "✗ FAIL: One or both tests did not produce output"
    exit 1
fi
