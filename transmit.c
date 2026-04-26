/*
 * transmit.c — M-PSK audio transmitter
 *
 * Modulation: M-PSK  (Phase Shift Keying)
 * ─────────────────────────────────────────────────────────────────────
 * A single carrier at 750 Hz carries all data.  Each symbol encodes
 * log2(M) bits by setting the carrier phase to
 *
 *     phase = symbol_index × (2π / M)
 *
 * where symbol_index is the Gray-coded representation of the raw bit
 * group.  Gray coding ensures that adjacent constellation points differ
 * by exactly one bit, minimising bit errors near decision boundaries.
 *
 * A preamble of PREAMBLE_SYMBOLS phase-0° symbols is prepended to every
 * transmission so the receiver can measure the timing-dependent phase
 * offset and calibrate itself before decoding data.
 *
 * Usage
 * ─────
 *   echo "Hello" | ./transmit [-m M] [-c FREQ_HZ] [-s BUFS]
 *
 *   -m M   PSK order — must be a power of 2, minimum 2.
 *           2  = BPSK  (1 bit/symbol)
 *           4  = QPSK  (2 bits/symbol)   ← default
 *           8  = 8-PSK (3 bits/symbol)
 *          16  = 16-PSK …  etc.
 *
 *   -c HZ  carrier frequency in Hz (integer, < 20000)
 *           Must align with sample timing so the carrier completes
 *           an integer number of cycles per 256-sample buffer.
 *
 *   -s N   TX buffers per symbol (default 64; must be >= 32 and
 *           divisible by 16 so RX windows per symbol is integer)
 *
 * Timing
 * ──────
 *   Carrier     : CARRIER_CYCLES cycles per FRAMES_PER_BUFFER samples
 *               = 4 × 48000 / 256 = 750 Hz
 *   Symbol dur. : BUFFERS_PER_SYMBOL × FRAMES_PER_BUFFER / SAMPLE_RATE
 *               = 64 × 256 / 48000 ≈ 341 ms
 */

#include <math.h>
#include <portaudio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "psk_common.h"

/* ═══════════════════════════════════════════════════════════════════
 * Timing / carrier constants
 * ═══════════════════════════════════════════════════════════════════ */
#define SAMPLE_RATE          48000
#define FRAMES_PER_BUFFER    256
#define DEFAULT_BUFFERS_PER_SYMBOL 64
#define MIN_BUFFERS_PER_SYMBOL     32
#define DEFAULT_CARRIER_CYCLES 4
#define DEFAULT_NUM_CARRIERS   1
#define MAX_NUM_CARRIERS      32
#define CARRIER_SPACING_HZ     375
#define MAX_CARRIER_HZ       20000
#define PREAMBLE_SYMBOLS     20       /* phase-0° symbols sent before data       */

/* Carrier: exactly 4 cycles per 256-sample buffer → 750 Hz.
 * Because the carrier completes an integer number of cycles per buffer,
 * the phase accumulator resets cleanly at every buffer boundary —
 * meaning each buffer can independently specify an absolute carrier phase
 * without any inter-buffer continuity bookkeeping. */
/* Preamble length increased to 16 symbols (≈ 5.46 s) so that even when
 * the receiver starts up to 4 s after the transmitter, calibration
 * always lands entirely within the zero-phase preamble region and
 * never accidentally captures the 180° alignment marker. */
#undef  PREAMBLE_SYMBOLS
#define PREAMBLE_SYMBOLS    16

static const float TWO_PI          = 2.0f * (float)M_PI;

/* ═══════════════════════════════════════════════════════════════════
 * Helper functions
 * ═══════════════════════════════════════════════════════════════════ */

/* Returns non-zero if n is a power of two and n ≥ 2. */
static int is_power_of_2(int n)
{
    return (n >= 2) && ((n & (n - 1)) == 0);
}

/* Integer log base-2 (valid only for exact powers of two). */
static int ilog2(int n)
{
    int k = 0;
    while (n > 1) { n >>= 1; ++k; }
    return k;
}

static int gcd_int(int a, int b)
{
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return (a < 0) ? -a : a;
}

static int nearest_compatible_hz(int requested_hz)
{
    const int step = SAMPLE_RATE / gcd_int(SAMPLE_RATE, FRAMES_PER_BUFFER);
    const int min_hz = step;
    const int max_hz = ((MAX_CARRIER_HZ - 1) / step) * step;

    if (requested_hz <= min_hz) return min_hz;
    if (requested_hz >= max_hz) return max_hz;

    int lower = (requested_hz / step) * step;
    int upper = lower + step;
    if (lower < min_hz) lower = min_hz;
    if (upper > max_hz) upper = max_hz;

    return (requested_hz - lower <= upper - requested_hz) ? lower : upper;
}

static int parse_carrier_hz(const char *s, int *out_hz)
{
    char *ep;
    long v = strtol(s, &ep, 10);
    if (*ep != '\0') return 0;
    if (v <= 0 || v >= MAX_CARRIER_HZ) return 0;
    *out_hz = (int)v;
    return 1;
}

static int parse_buffers_per_symbol(const char *s, int *out_bpsym)
{
    char *ep;
    long v = strtol(s, &ep, 10);
    if (*ep != '\0') return 0;
    if (v < MIN_BUFFERS_PER_SYMBOL) return 0;
    if ((v % 16) != 0) return 0;
    *out_bpsym = (int)v;
    return 1;
}

static int parse_num_carriers(const char *s, int *out_n)
{
    char *ep;
    long v = strtol(s, &ep, 10);
    if (*ep != '\0') return 0;
    if (v < 1 || v > MAX_NUM_CARRIERS) return 0;
    *out_n = (int)v;
    return 1;
}

static int carrier_offset_index(int idx)
{
    if (idx <= 0) return 0;
    int step = (idx + 1) / 2;
    return (idx & 1) ? step : -step;
}

static void print_base_range_hint(int num_carriers)
{
    int min_off = 0;
    int max_off = 0;
    for (int c = 0; c < num_carriers; c++) {
        int off = carrier_offset_index(c);
        if (off < min_off) min_off = off;
        if (off > max_off) max_off = off;
    }

    int base_min = 1 - min_off * CARRIER_SPACING_HZ;
    int base_max = (MAX_CARRIER_HZ - 1) - max_off * CARRIER_SPACING_HZ;

    const int step = SAMPLE_RATE / gcd_int(SAMPLE_RATE, FRAMES_PER_BUFFER);
    int min_compat = ((base_min + step - 1) / step) * step;
    int max_compat = (base_max / step) * step;

    if (min_compat <= max_compat) {
        fprintf(stderr,
                "       For -k %d, valid base carrier range is %d..%d Hz (compatible step %d Hz).\n",
                num_carriers, min_compat, max_compat, step);
    } else {
        fprintf(stderr,
                "       No compatible base frequency exists for -k %d with spacing %d Hz under %d Hz limit.\n",
                num_carriers, CARRIER_SPACING_HZ, MAX_CARRIER_HZ - 1);
    }
}

/* Binary-reflected Gray encoding: maps a natural binary index n to the
 * corresponding Gray code word so that adjacent symbols differ by 1 bit. */
static int gray_encode(int n)
{
    return n ^ (n >> 1);
}

/* ═══════════════════════════════════════════════════════════════════
 * bytes_to_symbols
 *
 * Convert raw data bytes into an array of Gray-coded M-PSK symbol
 * indices (each in [0, M-1]).  The bit stream is read MSB-first from
 * each byte and grouped into chunks of bits_per_symbol (= log2(M)).
 * The final group is zero-padded on the right if necessary.
 *
 * Returns a heap-allocated array; caller must free().
 * *out_count is set to the number of symbols produced.
 * Returns NULL on allocation failure.
 * ═══════════════════════════════════════════════════════════════════ */
static uint8_t *bytes_to_symbols_mc(const uint8_t *data, size_t data_len,
                                    int bps, int num_carriers,
                                    size_t *out_symbol_count)
{
    size_t total_bits = data_len * 8;
    size_t bits_per_timeslot = (size_t)bps * (size_t)num_carriers;
    size_t n_symbols = (total_bits + bits_per_timeslot - 1) / bits_per_timeslot;
    size_t total_syms = n_symbols * (size_t)num_carriers;

    uint8_t *syms = calloc(total_syms, 1);
    if (!syms) return NULL;

    for (size_t s = 0; s < n_symbols; s++) {
        for (int c = 0; c < num_carriers; c++) {
            int raw = 0;
            for (int b = 0; b < bps; b++) {
                size_t bit_idx = s * bits_per_timeslot
                               + (size_t)c * (size_t)bps
                               + (size_t)b;
                int bit = 0;
                if (bit_idx < total_bits) {
                    size_t byte_i = bit_idx / 8;
                    int bit_i = 7 - (int)(bit_idx % 8);
                    bit = (data[byte_i] >> bit_i) & 1;
                }
                raw = (raw << 1) | bit;
            }
            syms[s * (size_t)num_carriers + (size_t)c] = (uint8_t)gray_encode(raw);
        }
    }

    *out_symbol_count = n_symbols;
    return syms;
}

/* ═══════════════════════════════════════════════════════════════════
 * PortAudio callback state
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    const uint8_t *syms;       /* full symbol array (preamble + data)      */
    size_t         n_syms;     /* total number of symbols                  */
    size_t         sym_idx;    /* index of symbol currently being sent     */
    size_t         buf_count;  /* number of buffers sent for current symbol*/
    int            buffers_per_symbol;
    int            num_carriers;
    float          two_pi_fc_over_n[MAX_NUM_CARRIERS];
    int            M;          /* PSK order                                */
} cb_state_t;

/* ═══════════════════════════════════════════════════════════════════
 * audio_callback
 *
 * Called by PortAudio each time it needs FRAMES_PER_BUFFER more samples.
 *
 * Generates:  out[t] = sin(2π · CARRIER_CYCLES · t / N  +  φ_sym)
 *
 * where φ_sym = sym_idx_value × (2π / M).  Because CARRIER_CYCLES is
 * an exact integer and N = FRAMES_PER_BUFFER, the carrier completes
 * exactly CARRIER_CYCLES full cycles per buffer and the phase term
 * resets cleanly — each buffer encodes a pure, unambiguous phase.
 * ═══════════════════════════════════════════════════════════════════ */
static int audio_callback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo *timeInfo,
                           PaStreamCallbackFlags statusFlags, void *userData)
{
    (void)inputBuffer; (void)framesPerBuffer;
    (void)timeInfo;    (void)statusFlags;

    cb_state_t *d = (cb_state_t *)userData;
    float      *o = (float *)outputBuffer;

    /* All symbols sent — signal stream completion. */
    if (d->sym_idx >= d->n_syms)
        return paComplete;

    /* Fill the output buffer */
    for (int t = 0; t < FRAMES_PER_BUFFER; t++) {
        float sample = 0.0f;
        for (int c = 0; c < d->num_carriers; c++) {
            size_t idx = d->sym_idx * (size_t)d->num_carriers + (size_t)c;
            float phase = (float)d->syms[idx] * TWO_PI / (float)d->M;
            sample += sinf(d->two_pi_fc_over_n[c] * (float)t + phase);
        }
        o[t] = sample / (float)d->num_carriers;
    }

    /* Advance buffer counter; roll over to next symbol when due */
    if (++d->buf_count >= (size_t)d->buffers_per_symbol) {
        d->buf_count = 0;
        d->sym_idx++;
    }

    return paContinue;
}



/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /* ── parse arguments ──────────────────────────────────────────── */
    int         M          = 4;    /* default: QPSK                   */
    int         fec        = 0;    /* FEC disabled by default         */
    int         depth      = PSK_DEFAULT_ILVE_DEPTH;
    int         carrier_hz = DEFAULT_CARRIER_CYCLES * SAMPLE_RATE / FRAMES_PER_BUFFER;
    int         buffers_per_symbol = DEFAULT_BUFFERS_PER_SYMBOL;
    int         num_carriers = DEFAULT_NUM_CARRIERS;
    const char *input_file = NULL; /* NULL → read from stdin          */

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-i") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -i requires a filename.\n");
                return 1;
            }
            input_file = argv[++a];
        } else if (strcmp(argv[a], "-m") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -m requires an argument.\n");
                return 1;
            }
            M = atoi(argv[++a]);
            if (!is_power_of_2(M)) {
                fprintf(stderr,
                    "Error: M=%d is not a power of 2 (must be ≥ 2).\n"
                    "  Valid values: 2 (BPSK), 4 (QPSK), 8, 16, 32 …\n",
                    M);
                return 1;
            }
        } else if (strcmp(argv[a], "-c") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -c requires carrier frequency in Hz.\n");
                return 1;
            }
            if (!parse_carrier_hz(argv[++a], &carrier_hz)) {
                char *ep;
                long raw = strtol(argv[a], &ep, 10);
                fprintf(stderr,
                        "Error: invalid carrier frequency '%s' Hz (must be integer in 1..%d).\n",
                        argv[a], MAX_CARRIER_HZ - 1);
                if (ep != argv[a]) {
                    int nearest = nearest_compatible_hz((int)raw);
                    fprintf(stderr,
                            "       Nearest compatible frequency: %d Hz.\n",
                            nearest);
                }
                fprintf(stderr,
                        "       Example values: 750, 1125, 1500.\n");
                return 1;
            }
        } else if (strcmp(argv[a], "-s") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -s requires buffers-per-symbol value.\n");
                return 1;
            }
            if (!parse_buffers_per_symbol(argv[++a], &buffers_per_symbol)) {
                fprintf(stderr,
                        "Error: invalid -s value '%s' (must be integer >= 32 and divisible by 16).\n",
                        argv[a]);
                fprintf(stderr,
                        "       Example values: 32, 48, 64.\n");
                return 1;
            }
        } else if (strcmp(argv[a], "-k") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -k requires a carrier-count value.\n");
                return 1;
            }
            if (!parse_num_carriers(argv[++a], &num_carriers)) {
                fprintf(stderr,
                        "Error: invalid -k value '%s' (must be integer in 1..%d).\n",
                        argv[a], MAX_NUM_CARRIERS);
                return 1;
            }
        } else if (strcmp(argv[a], "-e") == 0) {
            fec = 1;
        } else if (strcmp(argv[a], "-d") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -d requires a depth argument.\n");
                return 1;
            }
            depth = atoi(argv[++a]);
            if (depth < 1) {
                fprintf(stderr, "Error: depth must be ≥ 1.\n");
                return 1;
            }
        } else {
            fprintf(stderr,
                "Unknown argument: %s\n"
                "Usage: %s [-m M] [-c FREQ_HZ] [-s BUFS] [-k CARRIERS] [-e] [-d DEPTH] [-i INPUT_FILE]\n"
                "  -c HZ     carrier frequency in Hz (integer, < %d)\n"
                "  -s N      TX buffers/symbol (>= %d, divisible by 16)\n"
                "  -k N      number of parallel carriers (1..%d, spacing %d Hz)\n"
                "  -i FILE   read payload from FILE instead of stdin\n",
                argv[a], argv[0], MAX_CARRIER_HZ, MIN_BUFFERS_PER_SYMBOL,
                MAX_NUM_CARRIERS, CARRIER_SPACING_HZ);
            return 1;
        }
    }

    if (((long)carrier_hz * FRAMES_PER_BUFFER) % SAMPLE_RATE != 0) {
        int nearest = nearest_compatible_hz(carrier_hz);
        fprintf(stderr,
                "Error: carrier %d Hz is incompatible with %d Hz sample rate and %d-sample TX buffers.\n",
                carrier_hz, SAMPLE_RATE, FRAMES_PER_BUFFER);
        fprintf(stderr,
                "       Choose a frequency where (Hz * %d) is divisible by %d.\n",
                FRAMES_PER_BUFFER, SAMPLE_RATE);
        fprintf(stderr,
                "       Nearest compatible frequency: %d Hz.\n",
                nearest);
        fprintf(stderr,
                "       Examples: 750 Hz, 1125 Hz, 1500 Hz.\n");
        return 1;
    }

    int carrier_cycles = carrier_hz * FRAMES_PER_BUFFER / SAMPLE_RATE;
    int spacing_cycles = CARRIER_SPACING_HZ * FRAMES_PER_BUFFER / SAMPLE_RATE;
    for (int c = 0; c < num_carriers; c++) {
        int off = carrier_offset_index(c);
        int hz_c = carrier_hz + off * CARRIER_SPACING_HZ;
        if (hz_c <= 0 || hz_c >= MAX_CARRIER_HZ) {
            fprintf(stderr,
                    "Error: -k %d with base carrier %d Hz yields out-of-range carrier %d (%d Hz).\n",
                    num_carriers, carrier_hz, MAX_CARRIER_HZ - 1, c + 1, hz_c);
            print_base_range_hint(num_carriers);
            return 1;
        }
        if (((long)hz_c * FRAMES_PER_BUFFER) % SAMPLE_RATE != 0) {
            fprintf(stderr,
                    "Error: derived carrier %d Hz is incompatible with sample timing.\n",
                    hz_c);
            return 1;
        }
    }

    int bps = ilog2(M);
    float actual_carrier_hz =
        (float)carrier_cycles * SAMPLE_RATE / FRAMES_PER_BUFFER;
    float raw_rate_bps =
        (float)(bps * num_carriers) * (float)SAMPLE_RATE
        / ((float)buffers_per_symbol * (float)FRAMES_PER_BUFFER);
    float sym_ms = 1000.0f * (float)buffers_per_symbol
                 * (float)FRAMES_PER_BUFFER / (float)SAMPLE_RATE;

    fprintf(stderr,
            "%d-PSK transmitter — %d carrier%s, %d bit%s/symbol each, base %.1f Hz, symbol %.1f ms, raw %.1f bps%s\n",
            M, num_carriers, num_carriers == 1 ? "" : "s",
            bps, bps == 1 ? "" : "s", actual_carrier_hz,
            sym_ms, raw_rate_bps,
            fec ? " [FEC on]" : "");

    if (buffers_per_symbol < 64) {
        fprintf(stderr,
                "Note: -s %d increases throughput but may raise BER on noisy channels.\n",
                buffers_per_symbol);
    }

    /* ── read input data (file or stdin) ─────────────────────────── */
    size_t   data_len = 0;
    uint8_t *data     = NULL;

    if (input_file) {
        data = psk_read_file(input_file, &data_len);
        if (!data) return 1;   /* psk_read_file printed the error    */
    } else {
        data = psk_read_stream(stdin, &data_len);
        if (!data) {
            fprintf(stderr, "Error: failed to read from stdin.\n");
            return 1;
        }
    }
    if (data_len == 0) {
        fprintf(stderr, "Error: no input data%s.\n",
                input_file ? " in file" : " on stdin");
        free(data);
        return 1;
    }

    /* ── optional FEC encode (Hamming(7,4) + block interleave) ──────── */
    if (fec) {
        size_t   enc_len;
        uint8_t *enc = psk_fec_encode(data, data_len, depth, &enc_len);
        free(data);
        if (!enc) {
            fprintf(stderr, "Error: FEC encode allocation failed.\n");
            return 1;
        }
        data     = enc;
        data_len = enc_len;
        fprintf(stderr,
                "FEC: Hamming(7,4) + interleave depth %d  "
                "(%zu → %zu bytes, rate %.0f%%)\n",
                depth, data_len * 4 / 7, data_len,
                100.0f * 4.0f / 7.0f);
    }

    /* ── convert bytes → Gray-coded M-PSK symbols ─────────────────── */
    size_t   n_data;
    uint8_t *data_syms = bytes_to_symbols_mc(data, data_len, bps, num_carriers, &n_data);
    free(data);
    if (!data_syms) {
        fprintf(stderr, "Error: allocation failed in bytes_to_symbols_mc.\n");
        return 1;
    }

    /* ── build full symbol array: preamble + alignment marker + data ─
     *
     * Layout
     * ──────
     * Indices 0 … PREAMBLE_SYMBOLS-1  : phase 0°  (8 calibration symbols)
     * Index   PREAMBLE_SYMBOLS        : phase 180° = symbol M/2
     *                                   (alignment marker — separate from
     *                                    preamble so calibration never
     *                                    captures the wrong phase)
     * Indices PREAMBLE_SYMBOLS+1 …    : Gray-coded data symbols
     *
     * Why the marker is placed AFTER the preamble
     * ────────────────────────────────────────────
     * With sleep 1 between transmitter and receiver start, the carrier
     * arrives ≈ 3 symbols into the transmission.  If the marker were
     * the last preamble symbol (index PREAMBLE_SYMBOLS-1), calibration
     * would land on the 180° marker and produce a wrong phase offset
     * (observed: 51.8° instead of the correct −90°).
     *
     * By placing the marker after all 8 zero-phase preamble symbols,
     * even a receiver that starts 2 s late calibrates entirely within
     * the zero-phase region and gets phase_offset = −90° reliably.
     *
     * The receiver detects the 0°→180° jump, sets timing_locked,
     * and uses skip_after_lock=1 to discard the marker before
     * accumulating data bits.
     */
    size_t   n_total  = (size_t)PREAMBLE_SYMBOLS + 1 + n_data;
    uint8_t *all_syms = calloc(n_total * (size_t)num_carriers, 1);   /* calloc zeroes preamble */
    if (!all_syms) {
        fprintf(stderr, "Error: allocation failed for symbol array.\n");
        free(data_syms);
        return 1;
    }
    /* Alignment marker immediately after the all-zero preamble */
    for (int c = 0; c < num_carriers; c++)
        all_syms[PREAMBLE_SYMBOLS * (size_t)num_carriers + (size_t)c] = (uint8_t)(M / 2);
    memcpy(all_syms + ((size_t)PREAMBLE_SYMBOLS + 1) * (size_t)num_carriers,
           data_syms,
           n_data * (size_t)num_carriers);
    free(data_syms);

    float tx_seconds = (float)n_total
                     * (float)buffers_per_symbol
                     * FRAMES_PER_BUFFER
                     / SAMPLE_RATE;
    fprintf(stderr,
            "Transmitting %d preamble + 1 marker + %zu data symbols/time"
            "  (%.1f s total).\n",
            PREAMBLE_SYMBOLS, n_data, tx_seconds);

    /* ── PortAudio ────────────────────────────────────────────────── */
    PaError pe = Pa_Initialize();
    if (pe != paNoError) {
        fprintf(stderr, "Pa_Initialize: %s\n", Pa_GetErrorText(pe));
        free(all_syms);
        return 1;
    }

    PaStream  *stream;
    cb_state_t cb;
    memset(&cb, 0, sizeof(cb));
    cb.syms = all_syms;
    cb.n_syms = n_total;
    cb.sym_idx = 0;
    cb.buf_count = 0;
    cb.buffers_per_symbol = buffers_per_symbol;
    cb.num_carriers = num_carriers;
    cb.M = M;
    for (int c = 0; c < num_carriers; c++) {
        int off = carrier_offset_index(c);
        int cyc = carrier_cycles + off * spacing_cycles;
        cb.two_pi_fc_over_n[c] = 2.0f * (float)M_PI * (float)cyc / (float)FRAMES_PER_BUFFER;
    }

    pe = Pa_OpenDefaultStream(&stream,
                              0, 1,           /* no input, 1 output channel  */
                              paFloat32,
                              SAMPLE_RATE, FRAMES_PER_BUFFER,
                              audio_callback, &cb);
    if (pe != paNoError) {
        fprintf(stderr, "Pa_OpenDefaultStream: %s\n", Pa_GetErrorText(pe));
        Pa_Terminate();
        free(all_syms);
        return 1;
    }

    pe = Pa_StartStream(stream);
    if (pe != paNoError) {
        fprintf(stderr, "Pa_StartStream: %s\n", Pa_GetErrorText(pe));
        Pa_CloseStream(stream);
        Pa_Terminate();
        free(all_syms);
        return 1;
    }

    while (Pa_IsStreamActive(stream))
        Pa_Sleep(100);

    Pa_CloseStream(stream);
    Pa_Terminate();
    free(all_syms);

    fprintf(stderr, "Transmission complete.\n");
    return 0;
}
