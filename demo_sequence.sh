#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./demo_sequence.sh [options]

Runs a fixed sequence of over-the-air transmitter/receiver demos:
  - PSK targets: 50 / 500 / 5000 bps
  - FSK targets: 50 / 500 / 5000 bps (5000 is mapped to nearest feasible)
  - Auto-search for highest observed rate with error probability < 0.01

Options:
  --device IDX        Pass --device IDX to receive
  --name SUBSTR       Pass -n SUBSTR to receive (device name filter)
  --warmup SEC        Wait after starting receiver (default: 2)
  --post-wait SEC     Wait after transmit completes (default: 8)
  --message TEXT      Payload text (default: built-in)
  --no-build          Skip "make transmit receive"
  --keep-temp         Keep temp directory with logs and outputs
  -h, --help          Show this help
EOF
}

DEVICE_ARG=()
NAME_ARG=()
WARMUP=2
POST_WAIT=8
DO_BUILD=1
KEEP_TEMP=0
MESSAGE="ENGN1580 final demo payload: reproducible text for symbol error estimation over acoustic channel."

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

PAYLOAD_FILE="./input.txt"

cleanup_receiver() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

compute_rate_bps() {
  local m="$1"
  local s="$2"
  local k="$3"
  local bits_per_symbol=0

  while (( m > 1 )); do
    m=$((m / 2))
    bits_per_symbol=$((bits_per_symbol + 1))
  done

  awk -v bpsym="$bits_per_symbol" -v bufs="$s" -v carr="$k" 'BEGIN { printf "%.2f", (48000.0/256.0) * bpsym * carr / bufs }'
}

calc_char_error_probability() {
  local expected="$1"
  local actual="$2"

  awk -v e="$expected" -v a="$actual" '
    BEGIN {
      le = length(e)
      la = length(a)
      max = (le > la) ? le : la
      if (max == 0) { printf "1.0000"; exit }
      mism = 0
      for (i = 1; i <= max; i++) {
        ce = (i <= le) ? substr(e, i, 1) : ""
        ca = (i <= la) ? substr(a, i, 1) : ""
        if (ce != ca) mism++
      }
      printf "%.4f", mism / max
    }'
}

run_demo_case() {
  local label="$1"
  local modulation="$2"
  local m="$3"
  local s="$4"
  local k="$5"
  local c="$6"
  local target_desc="$7"

  local safe_name
  safe_name=$(echo "$label" | tr ' /' '__')

  local out_file="$TEMP_DIR/${safe_name}.out"
  local rx_log="$TEMP_DIR/${safe_name}.receive.log"
  local tx_log="$TEMP_DIR/${safe_name}.transmit.log"

  local actual_rate
  actual_rate=$(compute_rate_bps "$m" "$s" "$k")

  local tx_opts="-t $modulation -m $m -s $s -c $c"
  local rx_opts="-t $modulation -m $m -s $s -c $c"
  if [[ "$k" -gt 1 ]]; then
    tx_opts+=" -k $k"
    rx_opts+=" -k $k"
  fi

  echo "[demo] $label"
  echo "  Target: $target_desc"
  echo "  Config: $tx_opts"
  echo "  Nominal rate: ${actual_rate} bps"

  set +e
  ./receive "${DEVICE_ARG[@]}" "${NAME_ARG[@]}" $rx_opts -o "$out_file" >"$rx_log" 2>&1 &
  local receiver_pid=$!
  set -e

  sleep "$WARMUP"

  local tx_status=0
  if ! ./transmit $tx_opts -i "$PAYLOAD_FILE" >"$tx_log" 2>&1; then
    tx_status=$?
  fi

  sleep "$POST_WAIT"
  cleanup_receiver "$receiver_pid"

  local decoded=""
  if [[ -f "$out_file" ]]; then
    decoded=$(cat "$out_file")
  fi

  local err_prob="1.0000"
  if [[ -n "$decoded" ]]; then
    err_prob=$(calc_char_error_probability "$MESSAGE" "$decoded")
  fi

  local result="PASS"
  if [[ "$tx_status" -ne 0 ]]; then
    result="FAIL (tx exit $tx_status)"
    err_prob="1.0000"
  elif [[ -z "$decoded" ]]; then
    result="FAIL (no decoded output)"
    err_prob="1.0000"
  fi

  echo "  Result: $result"
  echo "  Observed error probability: $err_prob"
  echo "  Logs: $rx_log | $tx_log"
  echo

  CASE_LABELS+=("$label")
  CASE_TARGETS+=("$target_desc")
  CASE_RATES+=("$actual_rate")
  CASE_ERRORS+=("$err_prob")
  CASE_RESULTS+=("$result")
}

declare -a CASE_LABELS
declare -a CASE_TARGETS
declare -a CASE_RATES
declare -a CASE_ERRORS
declare -a CASE_RESULTS

echo "===================================="
echo "Demo Sequence"
echo "===================================="

run_demo_case "PSK_50bps"   "psk" 32 16 1 10125   "PSK target 50 bps"
run_demo_case "PSK_500bps"  "psk" 32 16  8 10125   "PSK target 500 bps"
run_demo_case "PSK_5000bps" "psk" 256 8 32 10125   "PSK target 5000 bps"

run_demo_case "FSK_50bps"   "fsk" 16 16 1 750   "FSK target 50 bps"
run_demo_case "FSK_500bps"  "fsk" 64 8 1 750   "FSK target 500 bps"
run_demo_case "FSK_5000bps_nearest" "fsk" 64 8 1 750 "FSK target 5000 bps (nearest feasible in this implementation)"

echo "[search] Highest observed rate under error probability < 0.01"

BEST_LABEL=""
BEST_RATE="0.00"
BEST_ERR="1.0000"

# Ordered high to low nominal rate.
search_case() {
  local label="$1"
  local mod="$2"
  local m="$3"
  local s="$4"
  local k="$5"
  local c="$6"

  run_demo_case "$label" "$mod" "$m" "$s" "$k" "$c" "auto-search"

  local idx=$(( ${#CASE_LABELS[@]} - 1 ))
  local case_result="${CASE_RESULTS[$idx]}"
  local case_err="${CASE_ERRORS[$idx]}"
  local case_rate="${CASE_RATES[$idx]}"

  if [[ "$case_result" == "PASS" ]]; then
    if awk -v e="$case_err" 'BEGIN { exit !(e < 0.01) }'; then
      BEST_LABEL="$label"
      BEST_RATE="$case_rate"
      BEST_ERR="$case_err"
      return 0
    fi
  fi

  return 1
}

search_case "AUTO_psk_5000" "psk" 16 8 20 10125 || \
search_case "AUTO_psk_3000" "psk" 16 8 16 10125 || \
search_case "AUTO_psk_2000" "psk" 16 8 8 10125 || \
search_case "AUTO_fsk_1500" "fsk" 64 8 1 10125 || \
search_case "AUTO_psk_1000" "psk" 16 8 4 10125 || \
search_case "AUTO_fsk_500"  "fsk" 64 8 1 10125 || \
search_case "AUTO_psk_500"  "psk" 16 8 2 10125 || true

echo "===================================="
echo "Summary"
echo "===================================="

for i in "${!CASE_LABELS[@]}"; do
  printf '%-24s  %-10s  %-9s bps  p_err=%s  %s\n' \
    "${CASE_LABELS[$i]}" "${CASE_RESULTS[$i]}" "${CASE_RATES[$i]}" "${CASE_ERRORS[$i]}" "${CASE_TARGETS[$i]}"
done

echo
if [[ -n "$BEST_LABEL" ]]; then
  echo "Best observed config with p_err < 0.01: $BEST_LABEL (${BEST_RATE} bps, p_err=${BEST_ERR})"
else
  echo "No tested config met p_err < 0.01 in this run."
fi
