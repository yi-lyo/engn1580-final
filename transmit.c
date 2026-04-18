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
 *   echo "Hello" | ./transmit [-m M]
 *
 *   -m M   PSK order — must be a power of 2, minimum 2.
 *           2  = BPSK  (1 bit/symbol)
 *           4  = QPSK  (2 bits/symbol)   ← default
 *           8  = 8-PSK (3 bits/symbol)
 *          16  = 16-PSK …  etc.
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

/* ═══════════════════════════════════════════════════════════════════
 * Timing / carrier constants
 * ═══════════════════════════════════════════════════════════════════ */
#define SAMPLE_RATE          48000
#define FRAMES_PER_BUFFER    256
#define BUFFERS_PER_SYMBOL   64      /* symbol duration ≈ 341 ms               */
#define PREAMBLE_SYMBOLS     4       /* phase-0° symbols sent before data       */

/* Carrier: exactly 4 cycles per 256-sample buffer → 750 Hz.
 * Because the carrier completes an integer number of cycles per buffer,
 * the phase accumulator resets cleanly at every buffer boundary —
 * meaning each buffer can independently specify an absolute carrier phase
 * without any inter-buffer continuity bookkeeping. */
#define CARRIER_CYCLES       4

static const float TWO_PI          = 2.0f * (float)M_PI;
static const float TWO_PI_FC_OVER_N =
    2.0f * (float)M_PI * CARRIER_CYCLES / FRAMES_PER_BUFFER;

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
static uint8_t *bytes_to_symbols(const uint8_t *data, size_t data_len,
                                  int bps, size_t *out_count)
{
    size_t total_bits = data_len * 8;
    size_t n_syms     = (total_bits + (size_t)bps - 1) / (size_t)bps;

    uint8_t *syms = calloc(n_syms, 1);
    if (!syms) return NULL;

    for (size_t s = 0; s < n_syms; s++) {
        int raw = 0;
        for (int b = 0; b < bps; b++) {
            size_t bit_idx = s * (size_t)bps + (size_t)b;
            int    bit     = 0;
            if (bit_idx < total_bits) {
                size_t byte_i = bit_idx / 8;
                int    bit_i  = 7 - (int)(bit_idx % 8);  /* MSB first */
                bit = (data[byte_i] >> bit_i) & 1;
            }
            raw = (raw << 1) | bit;
        }
        syms[s] = (uint8_t)gray_encode(raw);
    }

    *out_count = n_syms;
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

    /* Carrier phase for the current symbol */
    float phase = (float)d->syms[d->sym_idx] * TWO_PI / (float)d->M;

    /* Fill the output buffer */
    for (int t = 0; t < FRAMES_PER_BUFFER; t++)
        o[t] = sinf(TWO_PI_FC_OVER_N * (float)t + phase);

    /* Advance buffer counter; roll over to next symbol when due */
    if (++d->buf_count >= BUFFERS_PER_SYMBOL) {
        d->buf_count = 0;
        d->sym_idx++;
    }

    return paContinue;
}

/* ═══════════════════════════════════════════════════════════════════
 * read_stdin — read all of stdin into a heap buffer
 * ═══════════════════════════════════════════════════════════════════ */
static uint8_t *read_stdin(size_t *out_len)
{
    size_t   cap = 4096, len = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) return NULL;

    int c;
    while ((c = getchar()) != EOF) {
        if (len == cap) {
            uint8_t *tmp = realloc(buf, cap *= 2);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
        buf[len++] = (uint8_t)c;
    }
    *out_len = len;
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /* ── parse -m M ───────────────────────────────────────────────── */
    int M = 4;   /* default: QPSK */

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-m") == 0) {
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
        } else {
            fprintf(stderr,
                "Unknown argument: %s\n"
                "Usage: %s [-m M]   (reads from stdin)\n",
                argv[a], argv[0]);
            return 1;
        }
    }

    int bps = ilog2(M);
    float carrier_hz =
        (float)CARRIER_CYCLES * SAMPLE_RATE / FRAMES_PER_BUFFER;

    fprintf(stderr,
            "%d-PSK transmitter — %d bit%s/symbol, carrier %.0f Hz\n",
            M, bps, bps == 1 ? "" : "s", carrier_hz);

    /* ── read input data ──────────────────────────────────────────── */
    size_t   data_len;
    uint8_t *data = read_stdin(&data_len);
    if (!data) {
        fprintf(stderr, "Error: failed to allocate read buffer.\n");
        return 1;
    }
    if (data_len == 0) {
        fprintf(stderr, "Error: no input data on stdin.\n");
        free(data);
        return 1;
    }

    /* ── convert bytes → Gray-coded M-PSK symbols ─────────────────── */
    size_t   n_data;
    uint8_t *data_syms = bytes_to_symbols(data, data_len, bps, &n_data);
    free(data);
    if (!data_syms) {
        fprintf(stderr, "Error: allocation failed in bytes_to_symbols.\n");
        return 1;
    }

    /* ── build full symbol array: preamble (symbol 0) + data ─────── */
    size_t   n_total  = (size_t)PREAMBLE_SYMBOLS + n_data;
    uint8_t *all_syms = calloc(n_total, 1);   /* calloc zeroes preamble */
    if (!all_syms) {
        fprintf(stderr, "Error: allocation failed for symbol array.\n");
        free(data_syms);
        return 1;
    }
    memcpy(all_syms + PREAMBLE_SYMBOLS, data_syms, n_data);
    free(data_syms);

    float tx_seconds = (float)n_total
                     * BUFFERS_PER_SYMBOL
                     * FRAMES_PER_BUFFER
                     / SAMPLE_RATE;
    fprintf(stderr,
            "Transmitting %d preamble + %zu data symbols  (%.1f s total).\n",
            PREAMBLE_SYMBOLS, n_data, tx_seconds);

    /* ── PortAudio ────────────────────────────────────────────────── */
    PaError pe = Pa_Initialize();
    if (pe != paNoError) {
        fprintf(stderr, "Pa_Initialize: %s\n", Pa_GetErrorText(pe));
        free(all_syms);
        return 1;
    }

    PaStream  *stream;
    cb_state_t cb = { all_syms, n_total, 0, 0, M };

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