/*
 * transmit_cdma.c — CDMA-enhanced PSK audio transmitter
 *
 * Modulation: BPSK with CDMA spreading (Walsh-Hadamard codes)
 * ─────────────────────────────────────────────────────────────────────
 * Data is first converted to CDMA symbols (each carrying log2(SF) bits).
 * Each symbol is spread using Walsh-Hadamard codes into SF chips (+1/-1).
 * Each chip is transmitted as a BPSK symbol:
 *   +1 → 0° phase (symbol 0)
 *   -1 → 180° phase (symbol 1)
 *
 * Processing Gain
 * ───────────────
 * With spreading factor SF=16, we get ~12 dB processing gain, meaning
 * the system can tolerate 12 dB more noise than non-spread PSK.
 *
 * Usage
 * ─────
 *   echo "Hello" | ./transmit_cdma [-e]
 *
 *   -e         Enable FEC (Hamming codes + interleaving) before CDMA
 *   -i FILE    Read payload from FILE instead of stdin
 *
 * Timing
 * ──────
 *   Carrier     : 750 Hz (same as regular PSK transmitter)
 *   Chip rate   : ~2.9 chips/sec (symbol duration ≈ 341 ms)
 *   Data rate   : ~11.7 bits/sec with SF=16 (16x slower than 4-PSK)
 *   Processing gain: ~12 dB (with SF=16)
 */

#include <math.h>
#include <portaudio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "psk_common.h"
#include "cdma_common.h"

/* ═══════════════════════════════════════════════════════════════════
 * Timing / carrier constants
 * ═══════════════════════════════════════════════════════════════════ */
#define SAMPLE_RATE          48000
#define FRAMES_PER_BUFFER    256
#define BUFFERS_PER_CHIP     64      /* chip duration ≈ 341 ms (same as symbol) */
#define PREAMBLE_CHIPS       32      /* phase-0° chips for calibration */
#define CARRIER_CYCLES       4       /* 750 Hz carrier */

static const float TWO_PI          = 2.0f * (float)M_PI;
static const float TWO_PI_FC_OVER_N =
    2.0f * (float)M_PI * CARRIER_CYCLES / FRAMES_PER_BUFFER;

/* ═══════════════════════════════════════════════════════════════════
 * Callback state
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    int8_t *chips;         /* array of chips (+1/-1) to transmit */
    size_t  n_chips;       /* total number of chips */
    size_t  chip_idx;      /* current chip index */
    size_t  buf_count;     /* buffers sent for current chip */
} cb_state_t;

/* ═══════════════════════════════════════════════════════════════════
 * Audio callback — BPSK transmission of chips
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

    /* All chips sent — signal stream completion */
    if (d->chip_idx >= d->n_chips)
        return paComplete;

    /* Convert chip to BPSK phase: +1 → 0°, -1 → 180° */
    int bpsk_sym = cdma_chip_to_psk_symbol(d->chips[d->chip_idx]);
    float phase = (float)bpsk_sym * (float)M_PI;  /* 0 or π */

    /* Fill the output buffer with carrier at this phase */
    for (int t = 0; t < FRAMES_PER_BUFFER; t++)
        o[t] = sinf(TWO_PI_FC_OVER_N * (float)t + phase);

    /* Advance buffer counter; roll over to next chip when due */
    if (++d->buf_count >= BUFFERS_PER_CHIP) {
        d->buf_count = 0;
        d->chip_idx++;
    }

    return paContinue;
}

/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /* ── Parse arguments ─────────────────────────────────────────── */
    int         fec        = 0;    /* FEC disabled by default */
    int         depth      = PSK_DEFAULT_ILVE_DEPTH;
    const char *input_file = NULL; /* NULL → read from stdin */

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-i") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -i requires a filename.\n");
                return 1;
            }
            input_file = argv[++a];
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
                "Usage: %s [-e] [-d DEPTH] [-i INPUT_FILE]\n"
                "  -i FILE   read payload from FILE instead of stdin\n"
                "  -e        enable FEC (Hamming + interleaving)\n"
                "  -d DEPTH  interleave depth (default %d)\n",
                argv[a], argv[0], PSK_DEFAULT_ILVE_DEPTH);
            return 1;
        }
    }

    int SF = CDMA_SPREADING_FACTOR;
    float carrier_hz =
        (float)CARRIER_CYCLES * SAMPLE_RATE / FRAMES_PER_BUFFER;
    float chip_duration_ms =
        (float)BUFFERS_PER_CHIP * FRAMES_PER_BUFFER * 1000.0f / SAMPLE_RATE;
    float processing_gain_db = cdma_calculate_processing_gain_db(SF);

    fprintf(stderr,
        "CDMA Transmitter\n"
        "────────────────────────────────────────────────────\n"
        "  Carrier       : %.1f Hz\n"
        "  Spreading     : SF=%d (Walsh-Hadamard codes)\n"
        "  Bits/symbol   : %d\n"
        "  Chip duration : %.1f ms\n"
        "  Processing gain: %.1f dB\n"
        "  FEC           : %s\n",
        carrier_hz, SF, CDMA_BITS_PER_SYMBOL, chip_duration_ms,
        processing_gain_db, fec ? "enabled" : "disabled");

    if (fec) {
        fprintf(stderr, "  Interleave    : depth %d\n", depth);
    }

    /* ── Read input data ─────────────────────────────────────────── */
    size_t   data_len = 0;
    uint8_t *data     = NULL;

    if (input_file) {
        data = psk_read_file(input_file, &data_len);
        if (!data) {
            fprintf(stderr, "Error: could not read '%s'\n", input_file);
            return 1;
        }
    } else {
        data = psk_read_stream(stdin, &data_len);
        if (!data) {
            fprintf(stderr, "Error: could not read from stdin\n");
            return 1;
        }
    }

    fprintf(stderr, "  Payload       : %zu bytes\n", data_len);

    /* ── Apply FEC if requested ─────────────────────────────────── */
    uint8_t *encoded_data = data;
    size_t   encoded_len  = data_len;

    if (fec) {
        encoded_data = psk_fec_encode(data, data_len, depth, &encoded_len);
        if (!encoded_data) {
            fprintf(stderr, "Error: FEC encoding failed\n");
            free(data);
            return 1;
        }
        fprintf(stderr, "  After FEC     : %zu bytes (rate %.2f)\n",
                encoded_len, (float)data_len / (float)encoded_len);
    }

    /* ── Convert to CDMA symbols ────────────────────────────────── */
    size_t   n_symbols = 0;
    uint8_t *symbols   = cdma_bytes_to_symbols(encoded_data, encoded_len,
                                                &n_symbols);
    if (!symbols) {
        fprintf(stderr, "Error: CDMA symbol conversion failed\n");
        if (fec && encoded_data != data) free(encoded_data);
        free(data);
        return 1;
    }

    fprintf(stderr, "  CDMA symbols  : %zu\n", n_symbols);

    /* ── Spread to chips ────────────────────────────────────────── */
    size_t total_chips = PREAMBLE_CHIPS + (n_symbols * SF);
    int8_t *all_chips  = (int8_t *)malloc(total_chips * sizeof(int8_t));
    if (!all_chips) {
        fprintf(stderr, "Error: chip allocation failed\n");
        free(symbols);
        if (fec && encoded_data != data) free(encoded_data);
        free(data);
        return 1;
    }

    /* Preamble: all +1 chips (0° phase) for receiver calibration */
    for (size_t i = 0; i < PREAMBLE_CHIPS; i++) {
        all_chips[i] = 1;
    }

    /* Spread data symbols to chips using Walsh codes */
    for (size_t s = 0; s < n_symbols; s++) {
        int8_t chips[64];  /* Max SF = 64 */
        cdma_spread_symbol(symbols[s], chips, SF);
        
        /* Copy to output chip array */
        for (int c = 0; c < SF; c++) {
            all_chips[PREAMBLE_CHIPS + s * SF + c] = chips[c];
        }
    }

    fprintf(stderr, "  Total chips   : %zu (%zu preamble + %zu data)\n",
            total_chips, (size_t)PREAMBLE_CHIPS, n_symbols * SF);

    float duration_sec = (float)total_chips * chip_duration_ms / 1000.0f;
    fprintf(stderr, "  Duration      : %.1f seconds\n", duration_sec);
    fprintf(stderr, "────────────────────────────────────────────────────\n");

    /* ── Initialize PortAudio ───────────────────────────────────── */
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "PortAudio init error: %s\n", Pa_GetErrorText(err));
        free(all_chips);
        free(symbols);
        if (fec && encoded_data != data) free(encoded_data);
        free(data);
        return 1;
    }

    /* ── Set up callback state ──────────────────────────────────── */
    cb_state_t cb_state = {
        .chips     = all_chips,
        .n_chips   = total_chips,
        .chip_idx  = 0,
        .buf_count = 0
    };

    /* ── Open audio stream ──────────────────────────────────────── */
    PaStream *stream = NULL;
    err = Pa_OpenDefaultStream(&stream, 0, 1, paFloat32, SAMPLE_RATE,
                               FRAMES_PER_BUFFER, audio_callback,
                               &cb_state);
    if (err != paNoError) {
        fprintf(stderr, "PortAudio open error: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        free(all_chips);
        free(symbols);
        if (fec && encoded_data != data) free(encoded_data);
        free(data);
        return 1;
    }

    /* ── Start playback ─────────────────────────────────────────── */
    fprintf(stderr, "Transmitting...\n");

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "PortAudio start error: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        free(all_chips);
        free(symbols);
        if (fec && encoded_data != data) free(encoded_data);
        free(data);
        return 1;
    }

    /* ── Wait for completion ────────────────────────────────────── */
    while (Pa_IsStreamActive(stream) == 1) {
        Pa_Sleep(100);
    }

    fprintf(stderr, "Transmission complete.\n");

    /* ── Cleanup ────────────────────────────────────────────────── */
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    free(all_chips);
    free(symbols);
    if (fec && encoded_data != data) free(encoded_data);
    free(data);

    return 0;
}