#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./test_orchestrate.sh [options]

Run a small matrix of transmitter/receiver integration tests.

Options:
  --device IDX        Pass --device IDX to receive
  --name SUBSTR       Pass -n SUBSTR to receive (device name filter)
  --warmup SEC        Seconds to wait after starting receiver (default: 2)
  --post-wait SEC     Seconds to wait after transmit completes (default: 8)
  --message TEXT      Payload text to transmit (default: built-in)
  --no-build          Skip "make transmit receive"
  --keep-temp         Keep temp directory and logs/output files
  -h, --help          Show this help
EOF
}

DEVICE_ARG=()
NAME_ARG=()
WARMUP=2
POST_WAIT=8
DO_BUILD=1
KEEP_TEMP=0
MESSAGE="Hello from test_orchestrate.sh - transmitter/receiver option matrix"

while (($# > 0)); do
  case "$1" in
    --device)
      [[ $# -ge 2 ]] || { echo "Error: --device requires a value" >&2; exit 2; }
      DEVICE_ARG=(--device "$2")
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "Error: --name requires a value" >&2; exit 2; }
      NAME_ARG=(-n "$2")
      shift 2
      ;;
    --warmup)
      [[ $# -ge 2 ]] || { echo "Error: --warmup requires a value" >&2; exit 2; }
      WARMUP="$2"
      shift 2
      ;;
    --post-wait)
      [[ $# -ge 2 ]] || { echo "Error: --post-wait requires a value" >&2; exit 2; }
      POST_WAIT="$2"
      shift 2
      ;;
    --message)
      [[ $# -ge 2 ]] || { echo "Error: --message requires a value" >&2; exit 2; }
      MESSAGE="$2"
      shift 2
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "[setup] Building transmit/receive..."
  make transmit receive
fi

if [[ ! -x ./transmit || ! -x ./receive ]]; then
  echo "Error: ./transmit and/or ./receive not found or not executable" >&2
  exit 1
fi

TEMP_DIR=$(mktemp -d)
if [[ "$KEEP_TEMP" -eq 1 ]]; then
  echo "[setup] Keeping temp dir: $TEMP_DIR"
else
  trap 'rm -rf "$TEMP_DIR"' EXIT
fi

PAYLOAD_FILE="$TEMP_DIR/payload.txt"
printf '%s' "$MESSAGE" > "$PAYLOAD_FILE"

count_matches_pct() {
  local expected="$1"
  local actual="$2"
  local matches=0
  local total=${#expected}

  if (( total == 0 )); then
    echo 0
    return
  fi

  local i=0
  local max_i=${#actual}
  if (( total < max_i )); then
    max_i=$total
  fi

  for ((i=0; i<max_i; i++)); do
    if [[ "${expected:i:1}" == "${actual:i:1}" ]]; then
      ((matches++))
    fi
  done

  echo $((matches * 100 / total))
}

cleanup_receiver() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

run_case() {
  local case_name="$1"
  local tx_opts="$2"
  local rx_opts="$3"

  local safe_name
  safe_name=$(echo "$case_name" | tr ' /' '__')

  local out_file="$TEMP_DIR/${safe_name}.out"
  local rx_log="$TEMP_DIR/${safe_name}.receive.log"
  local tx_log="$TEMP_DIR/${safe_name}.transmit.log"

  local receiver_pid
  echo "[case] $case_name"
  echo "  RX opts: $rx_opts"
  echo "  TX opts: $tx_opts"

  set +e
  ./receive "${DEVICE_ARG[@]}" "${NAME_ARG[@]}" $rx_opts -o "$out_file" >"$rx_log" 2>&1 &
  receiver_pid=$!
  set -e

  sleep "$WARMUP"

  local tx_status=0
  if ! ./transmit $tx_opts -i "$PAYLOAD_FILE" >"$tx_log" 2>&1; then
    tx_status=$?
  fi

  sleep "$POST_WAIT"
  cleanup_receiver "$receiver_pid"

  local actual=""
  if [[ -f "$out_file" ]]; then
    actual=$(cat "$out_file")
  fi

  local acc
  acc=$(count_matches_pct "$MESSAGE" "$actual")

  local result="PASS"
  if [[ "$tx_status" -ne 0 ]]; then
    result="FAIL (tx exit $tx_status)"
  elif [[ -z "$actual" ]]; then
    result="FAIL (no decoded output)"
  elif (( acc < 80 )); then
    result="FAIL (accuracy ${acc}%)"
  fi

  echo "  Result: $result"
  echo "  Accuracy: ${acc}%"
  echo "  Logs: $rx_log | $tx_log"
  echo

  CASE_NAMES+=("$case_name")
  CASE_RESULTS+=("$result")
  CASE_ACCURACY+=("$acc")
}

declare -a CASE_NAMES
declare -a CASE_RESULTS
declare -a CASE_ACCURACY

run_case "default_qpsk" "-m 4 -c 750" "-m 4 -c 750"
run_case "faster_symbol_qpsk" "-m 4 -c 750 -s 48" "-m 4 -c 750 -s 48"
run_case "higher_carrier_8psk" "-m 8 -c 1125" "-m 8 -c 1125"
run_case "fec_16psk" "-m 16 -c 1500 -e -d 32" "-m 16 -c 1500 -e -d 32"

echo "===================================="
echo "Orchestration Summary"
echo "===================================="

overall_fail=0
for i in "${!CASE_NAMES[@]}"; do
  printf '%-20s  %-26s  %3s%%\n' "${CASE_NAMES[$i]}" "${CASE_RESULTS[$i]}" "${CASE_ACCURACY[$i]}"
  if [[ "${CASE_RESULTS[$i]}" == FAIL* ]]; then
    overall_fail=1
  fi
done

echo
if [[ "$overall_fail" -eq 0 ]]; then
  echo "All cases passed."
  exit 0
else
  echo "One or more cases failed."
  exit 1
fi
