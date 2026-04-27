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
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
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

#define UI_WIN_W 1100
#define UI_WIN_H 680
#define UI_FFT_BARS 96
#define UI_FREQ_MIN_HZ 20.0f
#define UI_FREQ_MAX_HZ 20000.0f

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

typedef struct {
    volatile int   tone_enabled;
    volatile float tone_freq_hz;
    volatile float tone_intensity;
    float          tone_phase;
    volatile int   noise_enabled;
    volatile float noise_intensity;
    uint32_t       noise_rng_state;
    int            noise_has_spare;
    float          noise_spare;
    volatile uint32_t latest_seq;
    volatile int   buffer_ready;
    float          latest_buffer[FRAMES_PER_BUFFER];
} tx_ui_state_t;

static void draw_text(SDL_Renderer *ren, TTF_Font *font,
                      const char *text, int x, int y, SDL_Color col)
{
    if (!font || !text || !text[0]) return;
    SDL_Surface *surf = TTF_RenderUTF8_Blended(font, text, col);
    if (!surf) return;
    SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, surf);
    SDL_FreeSurface(surf);
    if (!tex) return;
    int w = 0, h = 0;
    SDL_QueryTexture(tex, NULL, NULL, &w, &h);
    SDL_Rect dst = {x, y, w, h};
    SDL_RenderCopy(ren, tex, NULL, &dst);
    SDL_DestroyTexture(tex);
}

static int text_width(TTF_Font *font, const char *text)
{
    if (!font || !text || !text[0])
        return (int)(text ? strlen(text) * 9 : 0);
    int w = 0;
    int h = 0;
    if (TTF_SizeUTF8(font, text, &w, &h) < 0)
        return (int)strlen(text) * 9;
    return w;
}

static int point_in_rect(int x, int y, SDL_Rect r)
{
    return x >= r.x && x < (r.x + r.w) && y >= r.y && y < (r.y + r.h);
}

static float clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float noise_sigma_from_intensity(float intensity)
{
    float x = clampf(intensity, 0.0f, 1.0f);
    return 0.35f * x * x;
}

static uint32_t lcg_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float gaussian_randn(tx_ui_state_t *ui)
{
    if (ui->noise_has_spare) {
        ui->noise_has_spare = 0;
        return ui->noise_spare;
    }

    float u1 = ((float)((lcg_next(&ui->noise_rng_state) >> 8) & 0x00FFFFFFu) + 1.0f)
             / 16777217.0f;
    float u2 = ((float)((lcg_next(&ui->noise_rng_state) >> 8) & 0x00FFFFFFu) + 1.0f)
             / 16777217.0f;

    float mag = sqrtf(-2.0f * logf(u1));
    float ang = TWO_PI * u2;
    ui->noise_spare = mag * sinf(ang);
    ui->noise_has_spare = 1;
    return mag * cosf(ang);
}

static float slider_value_from_x(int mx, SDL_Rect track, float min_v, float max_v)
{
    float t = (float)(mx - track.x) / (float)SDL_max(track.w, 1);
    t = clampf(t, 0.0f, 1.0f);
    return min_v + t * (max_v - min_v);
}

static int slider_knob_x(SDL_Rect track, float value, float min_v, float max_v)
{
    float span = max_v - min_v;
    float t = (span > 0.0f) ? (value - min_v) / span : 0.0f;
    t = clampf(t, 0.0f, 1.0f);
    return track.x + (int)lroundf(t * (float)track.w);
}

static void compute_fft_bars(const float *samples, size_t n, float *bars)
{
    static float win[FRAMES_PER_BUFFER];
    static int win_init = 0;
    if (!win_init) {
        for (size_t t = 0; t < FRAMES_PER_BUFFER; t++) {
            win[t] = 0.5f - 0.5f * cosf(TWO_PI * (float)t / (float)(FRAMES_PER_BUFFER - 1));
        }
        win_init = 1;
    }

    const float hz_per_bin = (float)SAMPLE_RATE / (float)n;
    const float span_hz = UI_FREQ_MAX_HZ - UI_FREQ_MIN_HZ;

    for (int i = 0; i < UI_FFT_BARS; i++) {
        float frac = (UI_FFT_BARS > 1)
                   ? (float)i / (float)(UI_FFT_BARS - 1)
                   : 0.0f;
        float f_hz = UI_FREQ_MIN_HZ + frac * span_hz;
        int   bin  = (int)lroundf(f_hz / hz_per_bin);
        if (bin < 0) bin = 0;
        if (bin >= (int)n / 2) bin = (int)n / 2 - 1;

        float best_mag = 0.0f;
        for (int db = -1; db <= 1; db++) {
            int bb = bin + db;
            if (bb < 0) bb = 0;
            if (bb >= (int)n / 2) bb = (int)n / 2 - 1;

            float re = 0.0f;
            float im = 0.0f;
            float w = TWO_PI * (float)bb / (float)n;
            for (size_t t = 0; t < n; t++) {
                float a = w * (float)t;
                float s = samples[t] * win[t];
                re += s * cosf(a);
                im -= s * sinf(a);
            }
            float mag = sqrtf(re * re + im * im);
            if (mag > best_mag) best_mag = mag;
        }
        bars[i] = best_mag;
    }
}

static void render_tx_ui(SDL_Renderer *ren, TTF_Font *font,
                         const float *bars,
                         SDL_Rect chart,
                         SDL_Rect panel,
                         SDL_Rect tone_checkbox,
                         SDL_Rect noise_checkbox,
                         SDL_Rect freq_track,
                         SDL_Rect intensity_track,
                         SDL_Rect noise_track,
                         int tone_enabled,
                         float tone_freq_hz,
                         float tone_intensity,
                         int noise_enabled,
                         float noise_intensity,
                         float base_carrier_hz,
                         size_t sym_idx,
                         size_t n_syms)
{
    SDL_SetRenderDrawColor(ren, 7, 10, 18, 255);
    SDL_RenderClear(ren);

    SDL_SetRenderDrawColor(ren, 18, 24, 38, 255);
    SDL_RenderFillRect(ren, &chart);
    SDL_SetRenderDrawColor(ren, 38, 48, 73, 255);
    SDL_RenderDrawRect(ren, &chart);

    float max_mag = 1e-6f;
    for (int i = 0; i < UI_FFT_BARS; i++)
        if (bars[i] > max_mag) max_mag = bars[i];

    int bar_w = SDL_max(2, chart.w / UI_FFT_BARS);
    for (int i = 0; i < UI_FFT_BARS; i++) {
        float nrm = bars[i] / max_mag;
        int h = (int)(nrm * (float)(chart.h - 36));
        if (h < 1) h = 1;
        SDL_Rect br = {
            chart.x + i * bar_w,
            chart.y + chart.h - h - 1,
            SDL_max(1, bar_w - 1),
            h
        };
        SDL_SetRenderDrawColor(ren, 40, 175, 230, 255);
        SDL_RenderFillRect(ren, &br);
    }

    SDL_Color text = {220, 228, 240, 255};
    SDL_Color dim  = {120, 132, 150, 255};
    char line[128];
    int value_right_x = chart.x + chart.w - 14;

    draw_text(ren, font, "Transmitter FFT", 42, 18, text);
    snprintf(line, sizeof(line), "Base carrier: %.1f Hz", base_carrier_hz);
    draw_text(ren, font, line, 42, 50, dim);
    char sym_line[128];
    snprintf(sym_line, sizeof(sym_line), "Symbols: %zu / %zu", sym_idx, n_syms);
    int sym_x = chart.x + chart.w - text_width(font, sym_line) - 12;
    int sym_y = 50;
    if (sym_x < 330) {
        sym_x = 42;
        sym_y = 74;
    }
    draw_text(ren, font, sym_line, sym_x, sym_y, dim);

    draw_text(ren, font, "Spectrum (20 Hz - 20 kHz)", chart.x + 8, chart.y + 8, dim);

    SDL_SetRenderDrawColor(ren, 24, 31, 46, 255);
    SDL_RenderDrawLine(ren, chart.x, chart.y + 28, chart.x + chart.w, chart.y + 28);

    SDL_SetRenderDrawColor(ren, 30, 37, 54, 255);
    SDL_RenderFillRect(ren, &panel);
    SDL_SetRenderDrawColor(ren, 56, 67, 95, 255);
    SDL_RenderDrawRect(ren, &panel);

    draw_text(ren, font, "Tone Controls", panel.x + 12, panel.y + 8, text);
    snprintf(line, sizeof(line), "Tone: %s", tone_enabled ? "ON" : "OFF");
    draw_text(ren, font, line,
              panel.x + panel.w - text_width(font, line) - 14,
              panel.y + 8,
              tone_enabled ? text : dim);
    SDL_SetRenderDrawColor(ren, 46, 56, 82, 255);
    SDL_RenderDrawLine(ren, panel.x + 10, panel.y + 34,
                       panel.x + panel.w - 10, panel.y + 34);

    SDL_SetRenderDrawColor(ren, 235, 235, 235, 255);
    SDL_RenderDrawRect(ren, &tone_checkbox);
    if (tone_enabled) {
        SDL_RenderDrawLine(ren, tone_checkbox.x + 3, tone_checkbox.y + tone_checkbox.h / 2,
                           tone_checkbox.x + tone_checkbox.w / 2, tone_checkbox.y + tone_checkbox.h - 4);
        SDL_RenderDrawLine(ren, tone_checkbox.x + tone_checkbox.w / 2, tone_checkbox.y + tone_checkbox.h - 4,
                           tone_checkbox.x + tone_checkbox.w - 3, tone_checkbox.y + 3);
    }
    draw_text(ren, font, "Enable pure sinusoid", tone_checkbox.x + tone_checkbox.w + 10, tone_checkbox.y - 2, text);

    draw_text(ren, font, "Sinusoid frequency (20 Hz - 20 kHz)", freq_track.x, freq_track.y - 24,
              tone_enabled ? text : dim);
    SDL_SetRenderDrawColor(ren,
                           tone_enabled ? 80 : 55,
                           tone_enabled ? 140 : 70,
                           tone_enabled ? 235 : 95,
                           255);
    SDL_Rect freq_line = {freq_track.x, freq_track.y + freq_track.h / 2 - 2, freq_track.w, 4};
    SDL_RenderFillRect(ren, &freq_line);

    int freq_knob_x = slider_knob_x(freq_track, tone_freq_hz, UI_FREQ_MIN_HZ, UI_FREQ_MAX_HZ);
    SDL_Rect freq_knob = {freq_knob_x - 7, freq_track.y - 4, 14, freq_track.h + 8};
    SDL_SetRenderDrawColor(ren,
                           tone_enabled ? 240 : 110,
                           tone_enabled ? 245 : 120,
                           tone_enabled ? 255 : 140,
                           255);
    SDL_RenderFillRect(ren, &freq_knob);
    char freq_line_text[32];
    snprintf(freq_line_text, sizeof(freq_line_text), "%.0f Hz", tone_freq_hz);
    draw_text(ren, font,
              freq_line_text,
              value_right_x - text_width(font, freq_line_text),
              freq_track.y - 3,
              tone_enabled ? text : dim);

    draw_text(ren, font, "Sinusoid intensity", intensity_track.x, intensity_track.y - 24, text);
    SDL_SetRenderDrawColor(ren, 90, 185, 120, 255);
    SDL_Rect int_line = {intensity_track.x, intensity_track.y + intensity_track.h / 2 - 2,
                         intensity_track.w, 4};
    SDL_RenderFillRect(ren, &int_line);

    int int_knob_x = slider_knob_x(intensity_track, tone_intensity, 0.0f, 1.0f);
    SDL_Rect int_knob = {int_knob_x - 7, intensity_track.y - 4, 14, intensity_track.h + 8};
    SDL_SetRenderDrawColor(ren, 220, 255, 230, 255);
    SDL_RenderFillRect(ren, &int_knob);
    char intensity_line_text[32];
    snprintf(intensity_line_text, sizeof(intensity_line_text), "%.2f", tone_intensity);
    draw_text(ren, font,
              intensity_line_text,
              value_right_x - text_width(font, intensity_line_text),
              intensity_track.y - 3,
              text);

    SDL_SetRenderDrawColor(ren, 235, 235, 235, 255);
    SDL_RenderDrawRect(ren, &noise_checkbox);
    if (noise_enabled) {
        SDL_RenderDrawLine(ren, noise_checkbox.x + 3, noise_checkbox.y + noise_checkbox.h / 2,
                           noise_checkbox.x + noise_checkbox.w / 2, noise_checkbox.y + noise_checkbox.h - 4);
        SDL_RenderDrawLine(ren, noise_checkbox.x + noise_checkbox.w / 2, noise_checkbox.y + noise_checkbox.h - 4,
                           noise_checkbox.x + noise_checkbox.w - 3, noise_checkbox.y + 3);
    }
    draw_text(ren, font, "Enable Gaussian noise", noise_checkbox.x + noise_checkbox.w + 10,
              noise_checkbox.y - 2, text);

    draw_text(ren, font, "Noise intensity", noise_track.x, noise_track.y - 24,
              noise_enabled ? text : dim);
    SDL_SetRenderDrawColor(ren,
                           noise_enabled ? 210 : 95,
                           noise_enabled ? 125 : 95,
                           noise_enabled ?  70 : 95,
                           255);
    SDL_Rect noise_line = {noise_track.x, noise_track.y + noise_track.h / 2 - 2,
                           noise_track.w, 4};
    SDL_RenderFillRect(ren, &noise_line);

    int noise_knob_x = slider_knob_x(noise_track, noise_intensity, 0.0f, 1.0f);
    SDL_Rect noise_knob = {noise_knob_x - 7, noise_track.y - 4, 14, noise_track.h + 8};
    SDL_SetRenderDrawColor(ren,
                           noise_enabled ? 250 : 135,
                           noise_enabled ? 205 : 135,
                           noise_enabled ? 180 : 135,
                           255);
    SDL_RenderFillRect(ren, &noise_knob);
    float noise_sigma = noise_sigma_from_intensity(noise_intensity);
    char noise_line_text[32];
    snprintf(noise_line_text, sizeof(noise_line_text), "sigma %.3f", noise_sigma);
    draw_text(ren, font,
              noise_line_text,
              value_right_x - text_width(font, noise_line_text),
              noise_track.y - 3,
              noise_enabled ? text : dim);

    SDL_SetRenderDrawColor(ren, 46, 56, 82, 255);
    SDL_RenderDrawLine(ren, panel.x + 10, panel.y + panel.h - 24,
                       panel.x + panel.w - 10, panel.y + panel.h - 24);
    draw_text(ren, font,
              "Noise is full-band (20 Hz-20 kHz); value is Gaussian sigma per sample.",
              42, panel.y + panel.h - 18, dim);
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
    tx_ui_state_t *ui;
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
    tx_ui_state_t *ui = d->ui;

    /* All symbols sent — signal stream completion. */
    if (d->sym_idx >= d->n_syms)
        return paComplete;

    if (ui)
        ui->latest_seq++;

    /* Fill the output buffer */
    for (int t = 0; t < FRAMES_PER_BUFFER; t++) {
        float sample = 0.0f;
        for (int c = 0; c < d->num_carriers; c++) {
            size_t idx = d->sym_idx * (size_t)d->num_carriers + (size_t)c;
            float phase = (float)d->syms[idx] * TWO_PI / (float)d->M;
            sample += sinf(d->two_pi_fc_over_n[c] * (float)t + phase);
        }
        sample /= (float)d->num_carriers;

        if (ui && ui->tone_enabled) {
            float tone = sinf(ui->tone_phase);
            float tone_amp = clampf(ui->tone_intensity, 0.0f, 1.0f);
            sample = (1.0f - tone_amp) * sample + tone_amp * tone;

            ui->tone_phase += TWO_PI * ui->tone_freq_hz / (float)SAMPLE_RATE;
            if (ui->tone_phase > TWO_PI)
                ui->tone_phase -= TWO_PI;
        }

        if (ui && ui->noise_enabled) {
            float sigma = noise_sigma_from_intensity(ui->noise_intensity);
            sample += sigma * gaussian_randn(ui);
        }

        o[t] = clampf(sample, -1.0f, 1.0f);
        if (ui)
            ui->latest_buffer[t] = o[t];
    }

    if (ui)
        ui->latest_seq++;

    if (ui)
        ui->buffer_ready = 1;

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
                    "Error: -k %d with base carrier %d Hz yields out-of-range carrier #%d (%d Hz, limit 1..%d).\n",
                    num_carriers, carrier_hz, c + 1, hz_c, MAX_CARRIER_HZ - 1);
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

    tx_ui_state_t ui_state;
    memset(&ui_state, 0, sizeof(ui_state));
    ui_state.tone_enabled = 0;
    ui_state.tone_freq_hz = clampf(actual_carrier_hz, UI_FREQ_MIN_HZ, UI_FREQ_MAX_HZ);
    ui_state.tone_intensity = 0.5f;
    ui_state.tone_phase = 0.0f;
    ui_state.noise_enabled = 0;
    ui_state.noise_intensity = 0.15f;
    ui_state.noise_rng_state = 0x1234A5C3u;
    ui_state.noise_has_spare = 0;
    ui_state.noise_spare = 0.0f;
    cb.ui = &ui_state;

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

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        free(all_syms);
        return 1;
    }
    if (TTF_Init() < 0) {
        fprintf(stderr, "TTF_Init: %s\n", TTF_GetError());
        SDL_Quit();
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        free(all_syms);
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow(
        "M-PSK Transmitter — FFT + Sinusoid Controls",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        UI_WIN_W, UI_WIN_H,
        SDL_WINDOW_SHOWN);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        free(all_syms);
        return 1;
    }

    SDL_Renderer *ren = SDL_CreateRenderer(win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) {
        fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        TTF_Quit();
        SDL_Quit();
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        free(all_syms);
        return 1;
    }

    TTF_Font *font = NULL;
    static const char *const font_paths[] = {
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        NULL
    };
    for (int fp = 0; font_paths[fp] && !font; fp++)
        font = TTF_OpenFont(font_paths[fp], 20);
    if (!font)
        fprintf(stderr, "Warning: no font found — UI text labels will be absent.\n");

    SDL_SetWindowMinimumSize(win, 900, 760);

    SDL_Rect chart = {0, 0, 0, 0};
    SDL_Rect panel = {0, 0, 0, 0};
    SDL_Rect tone_checkbox = {0, 0, 0, 0};
    SDL_Rect freq_track = {0, 0, 0, 0};
    SDL_Rect intensity_track = {0, 0, 0, 0};
    SDL_Rect noise_checkbox = {0, 0, 0, 0};
    SDL_Rect noise_track = {0, 0, 0, 0};

    float bars[UI_FFT_BARS] = {0.0f};
    float bars_smooth[UI_FFT_BARS] = {0.0f};
    int dragging_freq = 0;
    int dragging_intensity = 0;
    int dragging_noise = 0;
    int running = 1;

    while (running) {
        int win_w = UI_WIN_W;
        int win_h = UI_WIN_H;
        SDL_GetWindowSize(win, &win_w, &win_h);

        int margin_x = SDL_max(20, win_w / 28);
        int top_y = SDL_max(76, win_h / 14);
        int gap = SDL_max(14, win_h / 44);
        int bottom_margin = SDL_max(16, win_h / 44);
        int panel_h = SDL_max(300, win_h / 3);
        int chart_h = win_h - top_y - gap - panel_h - bottom_margin;
        if (chart_h < 220) {
            chart_h = 220;
            panel_h = win_h - top_y - gap - chart_h - bottom_margin;
            if (panel_h < 260) panel_h = 260;
        }

        chart.x = margin_x;
        chart.y = top_y;
        chart.w = win_w - 2 * margin_x;
        chart.h = chart_h;

        panel.x = margin_x;
        panel.y = chart.y + chart.h + gap;
        panel.w = chart.w;
        panel.h = win_h - panel.y - bottom_margin;
        if (panel.h < 260) panel.h = 260;

        int track_x = panel.x + 24;
        int track_w = panel.w - 140;
        if (track_w < 260) track_w = 260;

        int row0 = panel.y + 42;
        int row1 = row0 + 42;
        int row2 = row1 + 42;
        int row3 = row2 + 54;
        int row4 = row3 + 36;

        tone_checkbox = (SDL_Rect){track_x, row0, 20, 20};
        freq_track = (SDL_Rect){track_x, row1, track_w, 14};
        intensity_track = (SDL_Rect){track_x, row2, track_w, 14};
        noise_checkbox = (SDL_Rect){track_x, row3, 20, 20};
        noise_track = (SDL_Rect){track_x, row4, track_w, 14};

        int stream_active = Pa_IsStreamActive(stream);
        if (stream_active < 0) {
            fprintf(stderr, "Pa_IsStreamActive: %s\n", Pa_GetErrorText(stream_active));
            break;
        }
        if (stream_active == 0)
            break;

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = 0;
            } else if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT) {
                int mx = ev.button.x;
                int my = ev.button.y;
                if (point_in_rect(mx, my, tone_checkbox)) {
                    ui_state.tone_enabled = !ui_state.tone_enabled;
                } else if (point_in_rect(mx, my, freq_track) && ui_state.tone_enabled) {
                    dragging_freq = 1;
                    ui_state.tone_freq_hz = slider_value_from_x(mx, freq_track,
                                                                UI_FREQ_MIN_HZ, UI_FREQ_MAX_HZ);
                } else if (point_in_rect(mx, my, intensity_track)) {
                    dragging_intensity = 1;
                    ui_state.tone_intensity = slider_value_from_x(mx, intensity_track, 0.0f, 1.0f);
                } else if (point_in_rect(mx, my, noise_checkbox)) {
                    ui_state.noise_enabled = !ui_state.noise_enabled;
                } else if (point_in_rect(mx, my, noise_track) && ui_state.noise_enabled) {
                    dragging_noise = 1;
                    ui_state.noise_intensity = slider_value_from_x(mx, noise_track, 0.0f, 1.0f);
                }
            } else if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) {
                dragging_freq = 0;
                dragging_intensity = 0;
                dragging_noise = 0;
            } else if (ev.type == SDL_MOUSEMOTION) {
                int mx = ev.motion.x;
                if (dragging_freq && ui_state.tone_enabled) {
                    ui_state.tone_freq_hz = slider_value_from_x(mx, freq_track,
                                                                UI_FREQ_MIN_HZ, UI_FREQ_MAX_HZ);
                }
                if (dragging_intensity) {
                    ui_state.tone_intensity = slider_value_from_x(mx, intensity_track, 0.0f, 1.0f);
                }
                if (dragging_noise && ui_state.noise_enabled) {
                    ui_state.noise_intensity = slider_value_from_x(mx, noise_track, 0.0f, 1.0f);
                }
            }
        }

        if (ui_state.buffer_ready) {
            float snap[FRAMES_PER_BUFFER];
            uint32_t s0, s1;
            do {
                s0 = ui_state.latest_seq;
                if (s0 & 1u) continue;
                memcpy(snap, ui_state.latest_buffer, sizeof(snap));
                s1 = ui_state.latest_seq;
            } while ((s0 != s1) || (s0 & 1u));

            compute_fft_bars(snap, FRAMES_PER_BUFFER, bars);
            for (int i = 0; i < UI_FFT_BARS; i++)
                bars_smooth[i] = 0.88f * bars_smooth[i] + 0.12f * bars[i];
            ui_state.buffer_ready = 0;
        }

        render_tx_ui(ren, font,
                     bars_smooth,
                     chart,
                     panel,
                     tone_checkbox,
                     noise_checkbox,
                     freq_track,
                     intensity_track,
                     noise_track,
                     ui_state.tone_enabled,
                     ui_state.tone_freq_hz,
                     ui_state.tone_intensity,
                     ui_state.noise_enabled,
                     ui_state.noise_intensity,
                     actual_carrier_hz,
                     cb.sym_idx,
                     cb.n_syms);
        SDL_RenderPresent(ren);
        SDL_Delay(16);
    }

    if (!running)
        Pa_StopStream(stream);

    if (font) TTF_CloseFont(font);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    TTF_Quit();
    SDL_Quit();

    Pa_CloseStream(stream);
    Pa_Terminate();
    free(all_syms);

    fprintf(stderr, "Transmission complete.\n");
    return 0;
}
