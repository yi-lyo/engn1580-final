/*
 * receive.c — M-PSK audio receiver with SDL2 real-time FFT + IQ display
 *
 * Modulation: M-PSK  (Phase Shift Keying), M = power of 2, default QPSK
 * ─────────────────────────────────────────────────────────────────────
 * A single carrier at 750 Hz carries all data.  The carrier phase encodes
 * log2(M) bits per symbol:  phase = symbol_index × (2π / M).
 * Gray coding is used so adjacent constellation points differ by 1 bit.
 *
 * The transmitter prepends PREAMBLE_SYMBOLS phase-0° symbols before data.
 * The receiver uses those to measure the timing-dependent phase offset and
 * enter the DECODING state automatically.
 *
 * State machine (main thread)
 * ────────────────────────────
 *   WAITING     — no carrier detected
 *   CALIBRATING — carrier appeared; accumulating preamble phase measurements
 *                 to compute the constant phase offset
 *   DECODING    — offset known; mapping received phase → nearest symbol
 *
 * Architecture
 * ────────────
 *   PortAudio thread   — projects audio onto carrier phasor (fast O(N))
 *                        copies samples + complex carrier value to shared buf
 *   Main thread        — computes 100-bin DFT, decodes PSK phase, renders
 *
 * Command-line flags
 * ──────────────────
 *   (none)              auto-select best input device
 *   -m M                PSK order (power of 2, ≥ 2; default 4 = QPSK)
 *   -d <index>          force a specific PortAudio device index
 *   -n <substr>         use first input device whose name contains substr
 *   --list-devices      print all input-capable devices and exit
 */

#include <complex.h>
#include <limits.h>
#include <math.h>
#include <portaudio.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "psk_common.h"

/* ═══════════════════════════════════════════════════════════════════
 * Signal / demodulation constants
 * ═══════════════════════════════════════════════════════════════════ */
#define SAMPLE_RATE              48000
#define FRAMES_PER_BUFFER        0x1000          /* 4096 samples/callback  */

/* Carrier bin mapping
 *   transmitter cycles/buffer = c (for 256-sample TX buffer)
 *   receiver carrier bin       = 16c (for 4096-sample RX window)
 * Default c=4 → bin 64 → 750 Hz. */
#define DEFAULT_CARRIER_CYCLES   4
#define MAX_CARRIER_HZ           20000
#define CARRIER_BIN_PER_CYCLE   (FRAMES_PER_BUFFER / 256)
#define MIN_BUFFERS_PER_SYMBOL   32
#define DEFAULT_NUM_CARRIERS      1
#define MAX_NUM_CARRIERS         32
#define CARRIER_SPACING_HZ      375

static int   g_carrier_bin = DEFAULT_CARRIER_CYCLES * CARRIER_BIN_PER_CYCLE;
static float g_carrier_hz  = 750.0f;
static int   g_carrier_bins[MAX_NUM_CARRIERS];
static float g_carrier_hzs[MAX_NUM_CARRIERS];
static int   g_num_carriers = 1;

/* Carrier DFT magnitude for a full-scale sine wave ≈ FRAMES_PER_BUFFER/2.
 * SIG_THRESHOLD is a low absolute floor (≈ 1% of full scale) so that
 * over-the-air reception works even when the acoustic path attenuates
 * the signal well below the software-loopback level.
 *
 * False positives are blocked by CARRIER_SNR_MIN_DB: the carrier bin
 * must also be at least 6 dB above the mean noise floor before the
 * receiver enters calibration.  This SNR gate is scale-independent and
 * is the primary carrier-present criterion for OTA use.
 */
#define SIG_THRESHOLD            5.0f

/* Minimum carrier SNR (dB) to consider the carrier present.
 * Uses the snr_db value already computed from the FFT magnitude array.
 * 6 dB ≈ carrier power 4× the average noise-bin power — clearly visible
 * in the FFT display while still rejecting broadband noise. */
#define CARRIER_SNR_MIN_DB       1.0f

/* Number of receiver windows used for preamble calibration.
 * Each window is FRAMES_PER_BUFFER samples.  With BUFFERS_PER_SYMBOL=64
 * in the transmitter (symbol = 16384 samples) we get 4 windows/symbol.
 * 32 windows = 8 preamble symbols — gives receiver ample time to lock
 * even when started 3-4 seconds after transmitter begins. */
#define CAL_WINDOWS             32

/* ═══════════════════════════════════════════════════════════════════
 * Window / chart geometry
 * ═══════════════════════════════════════════════════════════════════ */
#define WIN_W     2100
#define WIN_H     1320   /* includes decoded-output panel at bottom           */

#define CHART_X     50
#define CHART_Y    112
#define CHART_W   1100   /* = BAR_WIDTH × BAR_PX_W = 100 × 11 */
#define CHART_H    700
#define BAR_WIDTH  100
#define BAR_PX_W    11

#define FFT_DISPLAY_MIN_HZ 20.0f
#define FFT_DISPLAY_MAX_HZ 20000.0f

#define AXIS_Y   (CHART_Y + CHART_H + 12)   /* Hz tick labels               */
#define INFO_Y   (AXIS_Y + 42)              /* carrier magnitude/phase row  */
#define INFO2_Y  (INFO_Y + 34)              /* symbol/state row             */
#define SYNC_Y   (INFO2_Y + 40)             /* timing + PLL row             */
#define STAT_Y   (SYNC_Y + 36)              /* SNR / FEC / interference row */
#define HINT_Y   (STAT_Y + 36)              /* keyboard hint                */

/* ── Decoded output panel (full-width strip at the bottom) ─────── */
#define OUTPUT_X   CHART_X                  /* left-aligned with FFT chart   */
#define OUTPUT_W   (CHART_W + 30 + IQ_SIZE) /* spans both panels  (= 1030)   */
#define OUTPUT_Y   (HINT_Y + 20)            /* below keyboard hint           */
#define OUTPUT_H   (WIN_H - OUTPUT_Y - 10)  /* fills to window bottom (~148) */

/* ── Timing / carrier-recovery parameters ─────────────────────────── */
/*
 * SYM_WINDOWS  — number of receiver windows per transmitter symbol.
 *   Transmitter symbol = BUFFERS_PER_SYMBOL × FRAMES_PER_BUFFER_TX
 *                      = 64 × 256 = 16 384 samples.
 *   Receiver window    = FRAMES_PER_BUFFER = 4 096 samples.
 *   16 384 / 4 096 = 4 → exactly 4 receiver windows per symbol.
 *
 * PLL_ALPHA — first-order decision-directed carrier PLL loop gain.
 *   After every clean (non-boundary) symbol decision the PLL adjusts
 *   the phase offset by  α × phase_error  to track slow drift.
 *   Small α ⟹ slow but noise-robust tracking.
 */
#define PLL_ALPHA    0.025f

/* ── IQ constellation panel (right of FFT chart) ─────────────────── */
#define IQ_X      1200  /* = CHART_X + CHART_W + 50                  */
#define IQ_Y      112
#define IQ_SIZE   820   /* square panel                               */
#define IQ_HISTORY 200  /* frames of phase history (~16 s at ~12 Hz) */

/* ═══════════════════════════════════════════════════════════════════
 * IQ history point
 *   is_bnd = 1 when this point was captured during a boundary window
 *   (the window that straddles two symbols).  These points are drawn
 *   in a different colour so the user can see the inter-symbol smear.
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct { float re, im; int is_bnd; } IQ_Point;

/* ═══════════════════════════════════════════════════════════════════
 * Thread-shared audio buffer
 * ═══════════════════════════════════════════════════════════════════ */
static pthread_mutex_t s_mutex    = PTHREAD_MUTEX_INITIALIZER;
static float           s_samples[FRAMES_PER_BUFFER];
static float           s_ft_re[MAX_NUM_CARRIERS];
static float           s_ft_im[MAX_NUM_CARRIERS];
static float           s_mag[MAX_NUM_CARRIERS];
static volatile int    s_new_data = 0;

/* ═══════════════════════════════════════════════════════════════════
 * Carrier phasor lookup table
 *   exptable[i] = e^{ −i 2π × CARRIER_BIN/N × i }
 *   Inner product with audio gives DFT[CARRIER_BIN].
 * ═══════════════════════════════════════════════════════════════════ */
static float complex exptable[MAX_NUM_CARRIERS][FRAMES_PER_BUFFER];

/* ═══════════════════════════════════════════════════════════════════
 * Circular audio buffer for symbol-aligned window extraction
 *   Accumulates all incoming audio samples so we can extract windows
 *   at positions aligned to detected symbol boundaries.
 * ═══════════════════════════════════════════════════════════════════ */
#define SLIDE_BUF_SIZE (FRAMES_PER_BUFFER * 32)  /* 32 windows = 8 symbols */
static pthread_mutex_t slide_mutex       = PTHREAD_MUTEX_INITIALIZER;
static float           slide_buf[SLIDE_BUF_SIZE];
static size_t          slide_write_pos   = 0;
static size_t          slide_sample_count = 0;  /* total samples written */

typedef struct {
    size_t callback_count;
    size_t max_callback_count;
} callback_data_t;

/* ═══════════════════════════════════════════════════════════════════
 * Helper functions
 * ═══════════════════════════════════════════════════════════════════ */
static int is_power_of_2(int n) { return (n >= 2) && ((n & (n - 1)) == 0); }
static int ilog2(int n)          { int k = 0; while (n > 1) { n >>= 1; k++; } return k; }

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
    const int step = SAMPLE_RATE / gcd_int(SAMPLE_RATE, 256);
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

    const int step = SAMPLE_RATE / gcd_int(SAMPLE_RATE, 256);
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

/* Gray decode: convert Gray code word back to natural binary */
static int gray_decode(int g)
{
    int n = g;
    for (int mask = g >> 1; mask; mask >>= 1)
        n ^= mask;
    return n;
}

/* ───────────────────────────────────────────────────────────────────
 * audio_callback
 *
 * Projects the audio onto the carrier phasor (O(N) inner product),
 * then copies samples + complex carrier value to the shared buffer
 * via a non-blocking mutex trylock.
 * ─────────────────────────────────────────────────────────────────── */
static int audio_callback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo *timeInfo,
                           PaStreamCallbackFlags statusFlags, void *userData)
{
    (void)outputBuffer; (void)framesPerBuffer;
    (void)timeInfo;     (void)statusFlags;

    const float *mic = (const float *)inputBuffer;

    float complex ft[MAX_NUM_CARRIERS];
    for (int c = 0; c < g_num_carriers; c++) {
        ft[c] = 0.0f;
        for (size_t i = 0; i < FRAMES_PER_BUFFER; i++)
            ft[c] += mic[i] * exptable[c][i];
    }

    if (pthread_mutex_trylock(&s_mutex) == 0) {
        memcpy(s_samples, mic, FRAMES_PER_BUFFER * sizeof(float));
        for (int c = 0; c < g_num_carriers; c++) {
            s_ft_re[c] = crealf(ft[c]);
            s_ft_im[c] = cimagf(ft[c]);
            s_mag[c] = cabsf(ft[c]);
        }
        s_new_data = 1;
        pthread_mutex_unlock(&s_mutex);
    }

    /* Append to circular buffer for symbol-aligned extraction */
    if (pthread_mutex_trylock(&slide_mutex) == 0) {
        for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
            slide_buf[slide_write_pos] = mic[i];
            slide_write_pos = (slide_write_pos + 1) % SLIDE_BUF_SIZE;
        }
        slide_sample_count += FRAMES_PER_BUFFER;
        pthread_mutex_unlock(&slide_mutex);
    }

    callback_data_t *ud = (callback_data_t *)userData;
    if (++ud->callback_count >= ud->max_callback_count) return paComplete;
    return paContinue;
}

/* ───────────────────────────────────────────────────────────────────
 * compute_dft
 *
 * Analysis DFT for bins [0 … BAR_WIDTH-1].
 * Used by SNR/interference stats and demod support logic.
 * ─────────────────────────────────────────────────────────────────── */
static void compute_dft(const float *samples, size_t n, float *magnitudes)
{
    for (int k = 0; k < BAR_WIDTH; k++) {
        float complex w  = cexpf(-I * 2.0f * M_PI * (float)k / (float)n);
        float complex wt = 1.0f;
        float complex X  = 0.0f;
        for (size_t t = 0; t < n; t++) {
            X  += samples[t] * wt;
            wt *= w;
        }
        magnitudes[k] = cabsf(X) / (float)n;
    }
}

/* ───────────────────────────────────────────────────────────────────
 * compute_display_spectrum
 *
 * Fills display_mag[0 … BAR_WIDTH-1] by sampling the spectrum over
 * 20 Hz … 20 kHz (linear in frequency) so the chart spans the
 * audible range instead of just the first 100 DFT bins.
 * ─────────────────────────────────────────────────────────────────── */
static void compute_display_spectrum(const float *samples, size_t n,
                                     float *display_mag)
{
    const float hz_per_bin = (float)SAMPLE_RATE / (float)n;
    const float span_hz = FFT_DISPLAY_MAX_HZ - FFT_DISPLAY_MIN_HZ;

    for (int i = 0; i < BAR_WIDTH; i++) {
        float frac = (BAR_WIDTH > 1) ? (float)i / (float)(BAR_WIDTH - 1) : 0.0f;
        float f_hz = FFT_DISPLAY_MIN_HZ + frac * span_hz;
        int   bin  = (int)lroundf(f_hz / hz_per_bin);

        if (bin < 0) bin = 0;
        if (bin > (int)n / 2) bin = (int)n / 2;

        float complex w  = cexpf(-I * 2.0f * M_PI * (float)bin / (float)n);
        float complex wt = 1.0f;
        float complex X  = 0.0f;
        for (size_t t = 0; t < n; t++) {
            X  += samples[t] * wt;
            wt *= w;
        }
        display_mag[i] = cabsf(X) / (float)n;
    }
}

/* ───────────────────────────────────────────────────────────────────
 * draw_text  —  render a UTF-8 string at (x, y) with the given colour
 * ─────────────────────────────────────────────────────────────────── */
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

static SDL_Color carrier_color(int idx)
{
    if (idx < 0) idx = 0;
    float h = fmodf(0.61803398875f * (float)idx, 1.0f);  /* golden-ratio hue step */
    float s = 0.65f;
    float v = 1.0f;

    float hh = h * 6.0f;
    int   i  = (int)floorf(hh);
    float f  = hh - (float)i;
    float p  = v * (1.0f - s);
    float q  = v * (1.0f - s * f);
    float t  = v * (1.0f - s * (1.0f - f));

    float r, g, b;
    switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default:r = v; g = p; b = q; break;
    }

    SDL_Color c = {
        (Uint8)(r * 255.0f),
        (Uint8)(g * 255.0f),
        (Uint8)(b * 255.0f),
        255
    };
    return c;
}

/* ───────────────────────────────────────────────────────────────────
 * draw_circle  —  midpoint / Bresenham circle outline
 * ─────────────────────────────────────────────────────────────────── */
static void draw_circle(SDL_Renderer *ren, int cx, int cy, int r)
{
    if (r <= 0) return;
    int x = r, y = 0, err = 0;
    while (x >= y) {
        SDL_RenderDrawPoint(ren, cx + x, cy - y);
        SDL_RenderDrawPoint(ren, cx + y, cy - x);
        SDL_RenderDrawPoint(ren, cx - y, cy - x);
        SDL_RenderDrawPoint(ren, cx - x, cy - y);
        SDL_RenderDrawPoint(ren, cx - x, cy + y);
        SDL_RenderDrawPoint(ren, cx - y, cy + x);
        SDL_RenderDrawPoint(ren, cx + y, cy + x);
        SDL_RenderDrawPoint(ren, cx + x, cy + y);
        if (err <= 0) { y++; err += 2 * y + 1; }
        else          { x--; err -= 2 * x + 1; }
    }
}

/* ───────────────────────────────────────────────────────────────────
 * render_iq_panel
 *
 * Draw the M-PSK constellation diagram into the right-hand IQ panel.
 *
 * Each entry in hist[] is a unit-circle point (cos φ, sin φ) where φ is
 * the calibrated received phase.  The M reference markers are drawn at
 * angles  2π k / M  (k = 0 … M−1) so that — once calibrated — the
 * received dots cluster tightly on top of the markers.
 *
 * Colour scheme
 * ─────────────
 *   Dots during CALIBRATING  : amber  (calibration in progress)
 *   Dots during DECODING     : teal   (good data)
 *   Reference markers        : subtle blue-white cross ("+")
 *   Active marker            : bright white highlight
 *   Bit label per marker     : shown for M ≤ 8
 *
 * Alpha fades quadratically from newest (opaque) to oldest (transparent).
 * The three most recent points are drawn larger for visibility.
 *
 * rx_state : 0 = WAITING, 1 = CALIBRATING, 2 = DECODING
 * ─────────────────────────────────────────────────────────────────── */
static void render_iq_panel(SDL_Renderer *ren, TTF_Font *font,
                             const IQ_Point *hist, int head, int count,
                             int M, int current_sym, int rx_state,
                             int timing_locked, int sym_window,
                             int panel_x, int panel_y,
                             int panel_w, int panel_h,
                             int carrier_idx, float carrier_hz)
{
    const int pad = 8;
    const int top_h = (font ? 24 : 10);
    const int bottom_h = (panel_h >= 160 ? 38 : 20);
    int inner_w = panel_w - 2 * pad;
    int inner_h = panel_h - top_h - bottom_h - 2 * pad;
    int plot_side = SDL_min(inner_w, inner_h);
    if (plot_side < 40) return;

    int plot_x = panel_x + (panel_w - plot_side) / 2;
    int plot_y = panel_y + top_h + pad + (inner_h - plot_side) / 2;

    const int cx     = plot_x + plot_side / 2;
    const int cy     = plot_y + plot_side / 2;
    const int plot_r = plot_side / 2 - 10;   /* unit-circle pixel radius     */
    const float two_pi = 2.0f * (float)M_PI;

    /* ── background ────────────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 6, 6, 12, 255);
    SDL_Rect bg = {panel_x, panel_y, panel_w, panel_h};
    SDL_RenderFillRect(ren, &bg);

    /* ── reference circles at 25 %, 50 %, 75 %, 100 % ──────────────── */
    SDL_SetRenderDrawColor(ren, 20, 20, 42, 255);
    for (int ring = 1; ring <= 4; ring++)
        draw_circle(ren, cx, cy, plot_r * ring / 4);

    /* ── crosshair axes ────────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 36, 36, 68, 255);
    SDL_RenderDrawLine(ren, plot_x + 3, cy, plot_x + plot_side - 3, cy);
    SDL_RenderDrawLine(ren, cx, plot_y + 3, cx, plot_y + plot_side - 3);

    /* ── M reference constellation markers ─────────────────────────── */
    for (int k = 0; k < M; k++) {
        float angle = two_pi * (float)k / (float)M;
        int   mx    = cx + (int)(cosf(angle) * (float)plot_r);
        int   my    = cy - (int)(sinf(angle) * (float)plot_r);

        int active  = (k == current_sym && rx_state == 2);
        int arm     = active ? 9 : 6;

        /* Draw "+" marker */
        if (active) {
            SDL_SetRenderDrawColor(ren, 200, 200, 255, 255);
        } else {
            SDL_SetRenderDrawColor(ren, 55, 55, 100, 255);
        }
        SDL_RenderDrawLine(ren, mx - arm, my,       mx + arm, my);
        SDL_RenderDrawLine(ren, mx,       my - arm, mx,       my + arm);

        /* Bit-pattern label for M ≤ 8 (positioned just outside the ring) */
        if (font && M <= 8 && plot_side >= 180) {
            int bps   = ilog2(M);
            int bits  = gray_decode(k);
            char lbl[16] = {0};
            for (int b = bps - 1; b >= 0; b--)
                lbl[bps - 1 - b] = '0' + ((bits >> b) & 1);

            /* Push label outward from the circle centre */
            float push   = (float)(plot_r + 14);
            int   tx     = cx + (int)(cosf(angle) * push) - 6;
            int   ty     = cy - (int)(sinf(angle) * push) - 6;

            SDL_Color lc = active
                ? (SDL_Color){220, 220, 255, 255}
                : (SDL_Color){ 70,  70, 110, 255};
            draw_text(ren, font, lbl, tx, ty, lc);
        }
    }

    /* ── history dots with quadratic alpha fade ─────────────────────── */
    if (count > 0) {
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

        for (int i = 0; i < count; i++) {
            int idx = (head - count + i + IQ_HISTORY) % IQ_HISTORY;

            float t     = (count > 1)
                        ? (float)(i + 1) / (float)count
                        : 1.0f;
            Uint8 alpha = (Uint8)(t * t * 225.0f);
            if (alpha < 6) continue;

            /* Point → screen coordinates (y flipped) */
            int px = cx + (int)(hist[idx].re * (float)plot_r);
            int py = cy - (int)(hist[idx].im * (float)plot_r);

            /* Clamp inside panel */
            px = SDL_max(plot_x + 2, SDL_min(plot_x + plot_side - 6, px));
            py = SDL_max(plot_y + 2, SDL_min(plot_y + plot_side - 6, py));

            /* Colour:
             *   amber      — still calibrating (phase offset unknown)
             *   orange-red — boundary window (inter-symbol transition smear)
             *   teal       — clean decoded sample                          */
            if (rx_state == 1) {
                SDL_SetRenderDrawColor(ren, 220, 155,   0, alpha);
            } else if (hist[idx].is_bnd) {
                SDL_SetRenderDrawColor(ren, 210,  70,  30, alpha);
            } else {
                SDL_SetRenderDrawColor(ren,   0, 200, 175, alpha);
            }

            int dot  = (i >= count - 3) ? 7 : 4;
            int half = dot / 2;
            SDL_Rect dr = {px - half, py - half, dot, dot};
            SDL_RenderFillRect(ren, &dr);
        }

        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
    }

    /* ── border ─────────────────────────────────────────────────────── */
    SDL_Color cc = carrier_color(carrier_idx - 1);
    SDL_SetRenderDrawColor(ren,
                           (Uint8)SDL_max((int)cc.r - 80, 40),
                           (Uint8)SDL_max((int)cc.g - 80, 40),
                           (Uint8)SDL_max((int)cc.b - 80, 40),
                           255);
    SDL_RenderDrawRect(ren, &bg);

    /* ── labels ─────────────────────────────────────────────────────── */
    SDL_Color lbl = {105, 105, 145, 255};

    /* Panel title */
    {
        char title[64];
        snprintf(title, sizeof(title), "C%d %.0fHz", carrier_idx, carrier_hz);
        /* Rough centering */
        int tw = (int)strlen(title) * 7;
        SDL_Color tc = carrier_color(carrier_idx - 1);
        draw_text(ren, font, title,
                  panel_x + panel_w / 2 - tw / 2, panel_y + 3, tc);
    }

    /* Axis labels */
    if (plot_side >= 100) {
        draw_text(ren, font, "I",  plot_x + plot_side - 14, cy -  9, lbl);
        draw_text(ren, font, "Q",  cx + 5,                  plot_y + 5, lbl);
    }

    /* State indicator */
    {
        const char *state_str =
            rx_state == 0 ? "WAITING" :
            rx_state == 1 ? "CALIBRATING" :
                            "DECODING";
        SDL_Color sc =
            rx_state == 0 ? (SDL_Color){ 60,  60,  80, 255} :
            rx_state == 1 ? (SDL_Color){220, 155,   0, 255} :
                            (SDL_Color){  0, 200, 140, 255};
        draw_text(ren, font, state_str, panel_x + 6, panel_y + panel_h - 18, sc);
    }

    /* Timing window indicator (bottom-right of IQ panel) */
    if (timing_locked && panel_w >= 210) {
        SDL_Color tc = {60, 60, 95, 255};
        char tw_str[32];
        /* sym_window: 0 = boundary slot, >0 = safe slots */
        snprintf(tw_str, sizeof(tw_str),
                 "Win %d  %s",
                 sym_window,
                 sym_window == 0 ? "(boundary)" : "(safe)");
        draw_text(ren, font, tw_str,
                 panel_x + 6, panel_y + panel_h - 36, tc);
    }
}

static void render_iq_panels(SDL_Renderer *ren, TTF_Font *font,
                              IQ_Point iq_hist[MAX_NUM_CARRIERS][IQ_HISTORY],
                              int iq_head, int iq_count,
                              int M, const int *current_sym_c, int rx_state,
                              int timing_locked, int sym_window,
                              int num_carriers)
{
    const int gap = 8;
    int best_cols = 1;
    int best_rows = num_carriers;
    int best_plot = -1;

    for (int cols = 1; cols <= num_carriers; cols++) {
        int rows = (num_carriers + cols - 1) / cols;
        int cell_w = (IQ_SIZE - (cols + 1) * gap) / cols;
        int cell_h = (IQ_SIZE - (rows + 1) * gap) / rows;

        int inner_w = cell_w - 16;
        int inner_h = cell_h - 24 - 38 - 16;
        int plot_side = SDL_min(inner_w, inner_h);

        if (plot_side > best_plot ||
            (plot_side == best_plot && cols * rows < best_cols * best_rows)) {
            best_plot = plot_side;
            best_cols = cols;
            best_rows = rows;
        }
    }

    int cols = best_cols;
    int rows = best_rows;
    int cell_w = (IQ_SIZE - (cols + 1) * gap) / cols;
    int cell_h = (IQ_SIZE - (rows + 1) * gap) / rows;

    for (int c = 0; c < num_carriers; c++) {
        int row = c / cols;
        int col = c % cols;

        int row_first = row * cols;
        int row_items = SDL_min(cols, num_carriers - row_first);
        int row_offset = (cols - row_items) * (cell_w + gap) / 2;

        int px = IQ_X + gap + row_offset + col * (cell_w + gap);
        int py = IQ_Y + gap + row * (cell_h + gap);

        render_iq_panel(ren, font,
                        iq_hist[c], iq_head, iq_count,
                        M, current_sym_c[c], rx_state,
                        timing_locked, sym_window,
                        px, py, cell_w, cell_h,
                        c + 1, g_carrier_hzs[c]);
    }
}

/* ───────────────────────────────────────────────────────────────────
 * render_frame
 *
 * render_decoded_output
 *
 * Draws the full-width "Decoded Output" strip at the bottom of the window.
 * The panel updates live as symbols are decoded — each newly-decoded byte
 * appears immediately, giving a typewriter-style streaming display.
 *
 * Layout
 * ──────
 *   Title bar  — "Decoded Output  [N bytes · M FEC corrections · status]"
 *   Separator  — 1-px horizontal rule
 *   Text area  — decoded bytes rendered as UTF-8 text; non-printable bytes
 *                become a middle dot (·, U+00B7).  Lines are word-wrapped
 *                at CHARS_PER_LINE bytes and the view always scrolls to show
 *                the most recently decoded content.
 *   Cursor     — a blinking block (▌) appended to the last line while the
 *                receiver is in DECODING state.
 *
 * rx_state: 0 = WAITING, 1 = CALIBRATING, 2 = DECODING
 * ─────────────────────────────────────────────────────────────────── */
static void render_decoded_output(SDL_Renderer *ren, TTF_Font *font,
                                   const char *text, int text_chars,
                                   int byte_count, int corrections,
                                   int fec_enabled, int rx_state)
{
    const int X          = OUTPUT_X;
    const int Y          = OUTPUT_Y;
    const int W          = OUTPUT_W;
    const int H          = OUTPUT_H;
    const int PAD_X      = 8;
    const int PAD_Y      = 5;
    const int LINE_H     = 22;          /* pixels per text row              */
    const int TITLE_H    = 30;          /* pixels for the title row         */
    const int SEP_Y      = Y + PAD_Y + TITLE_H;
    const int CONTENT_Y  = SEP_Y + 5;
    const int CONTENT_H  = H - (CONTENT_Y - Y) - 4;
    const int MAX_LINES  = CONTENT_H / LINE_H;
    /* Maximum bytes per display line.  At ~7 px/char and OUTPUT_W ≈ 1030 px
     * the comfortable column count is around 128 characters.              */
    const int COLS       = 108;

    /* ── panel background ──────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 6, 6, 14, 255);
    SDL_Rect bg = {X, Y, W, H};
    SDL_RenderFillRect(ren, &bg);

    /* ── title ──────────────────────────────────────────────────────── */
    {
        const char *status_str =
            rx_state == 0 ? "waiting for carrier" :
            rx_state == 1 ? "calibrating…"        :
                            "receiving";

        char title[192];
        if (fec_enabled && corrections > 0)
            snprintf(title, sizeof(title),
                     "Decoded Output  [%d byte%s  \xc2\xb7  "
                     "%d FEC correction%s  \xc2\xb7  %s]",
                     byte_count, byte_count == 1 ? "" : "s",
                     corrections, corrections == 1 ? "" : "s",
                     status_str);
        else
            snprintf(title, sizeof(title),
                     "Decoded Output  [%d byte%s  \xc2\xb7  %s]",
                     byte_count, byte_count == 1 ? "" : "s",
                     status_str);

        SDL_Color tc = {120, 120, 160, 255};
        draw_text(ren, font, title, X + PAD_X, Y + PAD_Y, tc);
    }

    /* ── separator ──────────────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 38, 38, 65, 255);
    SDL_RenderDrawLine(ren, X + 4, SEP_Y, X + W - 4, SEP_Y);

    /* ── border ─────────────────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 55, 55, 90, 255);
    SDL_RenderDrawRect(ren, &bg);

    /* ── placeholder when nothing has been decoded yet ──────────────── */
    if (byte_count == 0 || text_chars == 0) {
        const char *hint =
            rx_state == 0 ? "Waiting for carrier\xe2\x80\xa6" :
            rx_state == 1 ? "Calibrating phase reference\xe2\x80\xa6" :
                            "Receiving\xe2\x80\xa6";
        SDL_Color hc = {50, 50, 80, 255};
        draw_text(ren, font, hint, X + PAD_X, CONTENT_Y, hc);
        return;
    }

    /* ── split preview text into display lines ───────────────────────
     * We iterate byte-by-byte, counting visual characters (middle-dot
     * sequences 0xC2 0xB7 count as one column each).  A new line begins
     * when a '\n' is encountered or when the column count reaches COLS.
     * We keep at most 256 line descriptors and render the last MAX_LINES.
     */
    typedef struct { const char *start; size_t bytes; } LineSpan;
    LineSpan spans[256];
    int n_spans = 0;

    const char *p          = text;
    const char *line_start = p;
    int         col        = 0;

    while (*p && n_spans < 255) {
        unsigned char c = (unsigned char)*p;

        /* Middle dot: U+00B7, encoded as 0xC2 0xB7 — counts as 1 column */
        int is_dot = (c == 0xC2 && (unsigned char)*(p+1) == 0xB7);
        int advance = is_dot ? 2 : 1;

        if (c == '\n' || col >= COLS) {
            spans[n_spans].start = line_start;
            spans[n_spans].bytes = (size_t)(p - line_start);
            n_spans++;
            if (c == '\n') p++;
            line_start = p;
            col = 0;
            continue;
        }

        p   += advance;
        col += 1;
    }
    /* Capture the final partial line */
    if (p > line_start && n_spans < 256) {
        spans[n_spans].start = line_start;
        spans[n_spans].bytes = (size_t)(p - line_start);
        n_spans++;
    }

    /* ── blinking cursor appended to the last line while decoding ────── */
    int show_cursor = (rx_state == 2) &&
                      (((int)(SDL_GetTicks() / 400)) % 2 == 0);

    /* ── render the last MAX_LINES spans ────────────────────────────── */
    int first = n_spans - MAX_LINES;
    if (first < 0) first = 0;

    SDL_Color text_col = {190, 200, 185, 255};

    for (int li = first; li < n_spans; li++) {
        /* Copy the span into a NUL-terminated buffer */
        char line_buf[COLS * 3 + 8];   /* ×3 for UTF-8, +8 for cursor   */
        size_t blen = spans[li].bytes;
        if (blen >= sizeof(line_buf) - 4) blen = sizeof(line_buf) - 4;
        memcpy(line_buf, spans[li].start, blen);
        line_buf[blen] = '\0';

        /* Append cursor glyph to the very last line */
        if (li == n_spans - 1 && show_cursor)
            /* U+258C LEFT HALF BLOCK, UTF-8: 0xE2 0x96 0x8C */
            strncat(line_buf, "\xe2\x96\x8c", sizeof(line_buf) - blen - 1);

        int row_y = CONTENT_Y + (li - first) * LINE_H;
        draw_text(ren, font, line_buf, X + PAD_X, row_y, text_col);
    }
}

/* ───────────────────────────────────────────────────────────────────
 * render_frame
 *
 * Render the FFT bar chart and M-PSK decode readout.
 * Does NOT call SDL_RenderPresent — the caller does that after also
 * rendering the IQ panel, so both panels land in the same frame flip.
 *
 * Visual elements (top → bottom):
 *   Title bar   — Hz/bin, carrier bin/frequency, M-PSK order
 *   Chart area  — dark background, horizontal grid
 *   Guide line  — dim dotted vertical at carrier bin
 *   Bars        — gold = carrier bin, cyan-green = all others
 *   Peak-hold   — 1-px bright line per bin, decays exponentially
 *   Chart border + frequency axis tick marks and labels
 *   Decode row  — carrier magnitude, phase, decoded symbol + bits
 * ─────────────────────────────────────────────────────────────────── */
static void render_frame(SDL_Renderer *ren, TTF_Font *font,
                          const float *magnitudes, float *peaks,
                          float carrier_mag, float carrier_phase_deg,
                          int current_sym, int current_bits,
                          int rx_state, int M, float live_rate_bps,
                          int sym_windows_cfg, int decode_window_idx,
                          /* timing recovery */
                          int sym_window, int is_boundary, int timing_locked,
                          /* carrier PLL */
                          float pll_error_deg,
                          /* channel quality */
                          float snr_db, int interference, int intf_bin,
                          /* FEC */
                          int fec_enabled, int fec_corrections,
                          int fec_depth)
{
    const float span_hz = FFT_DISPLAY_MAX_HZ - FFT_DISPLAY_MIN_HZ;
    const float display_hz_per_bar =
        (BAR_WIDTH > 1) ? span_hz / (float)(BAR_WIDTH - 1) : 0.0f;
    int carrier_display_bins[MAX_NUM_CARRIERS] = {0};
    for (int c = 0; c < g_num_carriers; c++) {
        int bin = (int)lroundf((g_carrier_hzs[c] - FFT_DISPLAY_MIN_HZ)
                               / display_hz_per_bar);
        if (bin < 0) bin = 0;
        if (bin >= BAR_WIDTH) bin = BAR_WIDTH - 1;
        carrier_display_bins[c] = bin;
    }

    /* ── clear ─────────────────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 14, 14, 22, 255);
    SDL_RenderClear(ren);

    /* ── title bar ─────────────────────────────────────────────────── */
    {
        char title[200];
        snprintf(title, sizeof(title),
                 "FFT Spectrum  —  %.0f..%.0f Hz   |   "
                 "Carrier: bin %d  (~%.0f Hz)   |   "
                 "%d-PSK  (%d bit%s/symbol)  |  %d carrier%s",
                 FFT_DISPLAY_MIN_HZ, FFT_DISPLAY_MAX_HZ,
                 g_carrier_bin, g_carrier_hz,
                 M, ilog2(M), ilog2(M) == 1 ? "" : "s",
                 g_num_carriers, g_num_carriers == 1 ? "" : "s");
        SDL_Color c = {175, 175, 210, 255};
        draw_text(ren, font, title, CHART_X, 18, c);

        /* Per-carrier color legend */
        int lx = CHART_X;
        int ly = 48;
        const int line_h = 24;
        const int max_x = CHART_X + CHART_W - 20;
        for (int k = 0; k < g_num_carriers; k++) {
            char lbuf[40];
            snprintf(lbuf, sizeof(lbuf), "C%d %.0fHz", k + 1, g_carrier_hzs[k]);

            int label_w = (int)strlen(lbuf) * 9;
            if (font && TTF_SizeUTF8(font, lbuf, &label_w, NULL) != 0)
                label_w = (int)strlen(lbuf) * 9;
            int entry_w = 14 + label_w + 16;

            if (lx + entry_w > max_x) {
                lx = CHART_X;
                ly += line_h;
            }

            SDL_Color cc = carrier_color(k);
            SDL_SetRenderDrawColor(ren, cc.r, cc.g, cc.b, 255);
            SDL_Rect sw = {lx, ly + 3, 10, 10};
            SDL_RenderFillRect(ren, &sw);

            SDL_Color lc = {155, 155, 190, 255};
            draw_text(ren, font, lbuf, lx + 14, ly, lc);

            lx += entry_w;
        }
    }

    /* ── chart background ──────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 6, 6, 12, 255);
    SDL_Rect chart_bg = {CHART_X, CHART_Y, CHART_W, CHART_H};
    SDL_RenderFillRect(ren, &chart_bg);

    /* ── horizontal grid at 25 %, 50 %, 75 % ───────────────────────── */
    SDL_SetRenderDrawColor(ren, 26, 26, 48, 255);
    for (int g = 1; g <= 3; g++) {
        int gy = CHART_Y + CHART_H - CHART_H * g / 4;
        SDL_RenderDrawLine(ren, CHART_X, gy, CHART_X + CHART_W, gy);
    }

    /* ── dim dotted guide line at the carrier bin ───────────────────── */
    for (int c = 0; c < g_num_carriers; c++) {
        int carrier_display_bin = carrier_display_bins[c];
        SDL_Color cc = carrier_color(c);
        SDL_SetRenderDrawColor(ren,
                               (Uint8)SDL_max((int)cc.r / 3, 25),
                               (Uint8)SDL_max((int)cc.g / 3, 25),
                               (Uint8)SDL_max((int)cc.b / 3, 25),
                               255);
        for (int dy = CHART_Y; dy < CHART_Y + CHART_H; dy += 3)
            SDL_RenderDrawPoint(ren, CHART_X + carrier_display_bin * BAR_PX_W, dy);
    }

    /* ── normalise bars to the loudest bin in this frame ────────────── */
    float max_mag = 1e-9f;
    for (int k = 0; k < BAR_WIDTH; k++)
        if (magnitudes[k] > max_mag) max_mag = magnitudes[k];

    /* ── bars + peak-hold ───────────────────────────────────────────── */
    for (int k = 0; k < BAR_WIDTH; k++) {
        int carrier_idx = -1;
        for (int c = 0; c < g_num_carriers; c++) {
            if (k == carrier_display_bins[c]) {
                carrier_idx = c;
                break;
            }
        }

        float norm  = magnitudes[k] / max_mag;
        int   bar_h = (int)(norm * (float)CHART_H);
        int   bx    = CHART_X + k * BAR_PX_W;

        /* Colour: carrier-coded for any carrier bin, cyan-green for others */
        Uint8 r, g, b;
        if (carrier_idx >= 0) {
            SDL_Color cc = carrier_color(carrier_idx);
            r = cc.r; g = cc.g; b = cc.b;
        } else {
            r =   0; g = 185; b = 90;
        }

        if (bar_h > 0) {
            SDL_SetRenderDrawColor(ren, r, g, b, 255);
            SDL_Rect bar = {bx, CHART_Y + CHART_H - bar_h,
                            BAR_PX_W - 1, bar_h};
            SDL_RenderFillRect(ren, &bar);
        }

        /* Peak-hold: jump on new peak, decay at ~0.97/frame */
        if (magnitudes[k] > peaks[k]) peaks[k] = magnitudes[k];
        else                          peaks[k] *= 0.97f;

        int peak_h = (int)(peaks[k] / max_mag * (float)CHART_H);
        if (peak_h > 1 && peak_h < CHART_H) {
            int py = CHART_Y + CHART_H - peak_h;
            SDL_SetRenderDrawColor(ren,
                (Uint8)SDL_min((int)r + 70, 255),
                (Uint8)SDL_min((int)g + 70, 255),
                (Uint8)SDL_min((int)b + 70, 255), 255);
            SDL_RenderDrawLine(ren, bx, py, bx + BAR_PX_W - 2, py);
        }
    }

    /* ── chart border ──────────────────────────────────────────────── */
    SDL_SetRenderDrawColor(ren, 55, 55, 90, 255);
    SDL_RenderDrawRect(ren, &chart_bg);

    /* ── frequency-axis tick marks and labels ───────────────────────── */
    {
        SDL_Color axis_col = {105, 105, 145, 255};
        SDL_SetRenderDrawColor(ren,
            axis_col.r, axis_col.g, axis_col.b, axis_col.a);
        for (int q = 0; q <= 4; q++) {
            int tx = CHART_X + CHART_W * q / 4;
            SDL_RenderDrawLine(ren, tx, CHART_Y + CHART_H,
                                    tx, CHART_Y + CHART_H + 5);
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%.0f Hz", FFT_DISPLAY_MIN_HZ);
        draw_text(ren, font, buf, CHART_X - 2, AXIS_Y, axis_col);
        for (int q = 1; q <= 4; q++) {
            float freq = FFT_DISPLAY_MIN_HZ
                       + (FFT_DISPLAY_MAX_HZ - FFT_DISPLAY_MIN_HZ)
                       * (float)q / 4.0f;
            snprintf(buf, sizeof(buf), "%.0f Hz", freq);
            int tx = CHART_X + CHART_W * q / 4 - 22;
            draw_text(ren, font, buf, tx, AXIS_Y, axis_col);
        }
    }

    /* ── M-PSK decode readout ───────────────────────────────────────── */
    {
        int bps = ilog2(M);

        /* Build decoded-bits string */
        char bits_str[64] = "---";
        if (rx_state == 2) {
            for (int b = bps - 1; b >= 0; b--)
                bits_str[bps - 1 - b] = '0' + ((current_bits >> b) & 1);
            bits_str[bps] = '\0';
        }

        /* Magnitude + phase */
        {
            char buf[80];
            snprintf(buf, sizeof(buf),
                     "Carrier mag: %7.1f   Phase: %+7.2f\xc2\xb0",
                     carrier_mag, carrier_phase_deg);
            SDL_Color c = {160, 160, 190, 255};
            draw_text(ren, font, buf, CHART_X, INFO_Y, c);
        }

        /* Decoded symbol + bits */
        {
            SDL_Color sc =
                rx_state == 0 ? (SDL_Color){ 55,  55,  75, 255} :
                rx_state == 1 ? (SDL_Color){200, 140,   0, 255} :
                                (SDL_Color){  0, 210, 160, 255};
            char buf[80];
            if (rx_state < 2) {
                snprintf(buf, sizeof(buf), "%s",
                         rx_state == 0 ? "Waiting for carrier…"
                                       : "Calibrating…");
            } else {
                snprintf(buf, sizeof(buf),
                         "Symbol: %d   Bits: %s", current_sym, bits_str);
            }
            draw_text(ren, font, buf, CHART_X, INFO2_Y, sc);
        }

        /* Keyboard hint */
        {
            SDL_Color c = {48, 48, 75, 255};
            const char *hint = "Q / Esc  to quit";
            int hint_w = 120;
            if (font && TTF_SizeUTF8(font, hint, &hint_w, NULL) != 0)
                hint_w = 120;
            draw_text(ren, font, "Q / Esc  to quit",
                      CHART_X + CHART_W - hint_w, HINT_Y, c);
        }
    }

    /* ── Channel-quality / FEC status row (STAT_Y) ──────────────────
     *
     * SNR estimate
     * ────────────
     * 10·log10(carrier_power / mean_noise_power) from the DFT magnitudes.
     * Colour: green ≥ 20 dB (good), yellow ≥ 10 dB (fair), red < 10 dB.
     *
     * Interference indicator
     * ──────────────────────
     * Lit when any non-carrier bin exceeds 50% of the carrier amplitude.
     * Shows the interfering bin index and its frequency.
     *
     * FEC statistics
     * ──────────────
     * When FEC is enabled (-e flag) shows the running count of single-bit
     * corrections made by Hamming(7,4) decoding.
     * ─────────────────────────────────────────────────────────────── */
    {
        SDL_Color dim = {70, 70, 100, 255};
        int x = CHART_X;

        /* SNR bar */
        draw_text(ren, font, "SNR:", x, STAT_Y + 4, dim);
        x += 38;

        {
            const int BAR_W = 100, BAR_H = 10;
            const int bar_y = STAT_Y + (22 - BAR_H) / 2;

            SDL_SetRenderDrawColor(ren, 25, 25, 45, 255);
            SDL_Rect bg = {x, bar_y, BAR_W, BAR_H};
            SDL_RenderFillRect(ren, &bg);

            /* Fill proportional to SNR: 0 dB → empty, 30 dB → full */
            float fill = (snr_db + 0.0f) / 30.0f;
            if (fill < 0.0f) fill = 0.0f;
            if (fill > 1.0f) fill = 1.0f;
            int fill_w = (int)(fill * BAR_W);
            if (fill_w > 0) {
                Uint8 r = fill >= 0.67f ? 0 : fill >= 0.33f ? 190 : 210;
                Uint8 g = fill >= 0.67f ? 195 : fill >= 0.33f ? 175 : 55;
                Uint8 b = 0;
                SDL_SetRenderDrawColor(ren, r, g, b, 255);
                SDL_Rect bar = {x, bar_y, fill_w, BAR_H};
                SDL_RenderFillRect(ren, &bar);
            }

            SDL_SetRenderDrawColor(ren, 50, 50, 80, 255);
            SDL_RenderDrawRect(ren, &bg);
        }
        x += 106;

        {
            char snr_buf[24];
            snprintf(snr_buf, sizeof(snr_buf), "%.1f dB", snr_db);
            SDL_Color sc = snr_db >= 20.0f ? (SDL_Color){0, 200, 100, 255} :
                           snr_db >= 10.0f ? (SDL_Color){200, 170, 0, 255} :
                                             (SDL_Color){210, 60, 30, 255};
            draw_text(ren, font, snr_buf, x, STAT_Y + 4, sc);
        }
        x += 70;

        /* Interference indicator */
        if (interference) {
            float hz_per_bin = (float)SAMPLE_RATE / (float)FRAMES_PER_BUFFER;
            char intf_buf[64];
            snprintf(intf_buf, sizeof(intf_buf),
                     "\xe2\x9a\xa0 INTF @ bin %d (%.0f Hz)",
                     intf_bin, (float)intf_bin * hz_per_bin);
            SDL_Color ic = {220, 100, 20, 255};
            draw_text(ren, font, intf_buf, x, STAT_Y + 4, ic);
            x += 200;
        }

        /* Live data-rate readout */
        {
            char rate_buf[64];
            snprintf(rate_buf, sizeof(rate_buf), "Rate: %6.1f bps", live_rate_bps);
            SDL_Color rc = live_rate_bps >= 200.0f ? (SDL_Color){0, 200, 110, 255} :
                           live_rate_bps >= 50.0f  ? (SDL_Color){200, 170, 0, 255} :
                                                     (SDL_Color){210, 80, 40, 255};
            draw_text(ren, font, rate_buf, x, STAT_Y + 4, rc);

            int rate_w = 120;
            if (font && TTF_SizeUTF8(font, rate_buf, &rate_w, NULL) != 0)
                rate_w = 120;
            x += rate_w + 12;
        }

        /* FEC statistics */
        if (fec_enabled) {
            char fec_buf[64];
            snprintf(fec_buf, sizeof(fec_buf),
                     "FEC(7,4) depth=%d  corrections: %d",
                     fec_depth, fec_corrections);
            SDL_Color fc = fec_corrections > 0
                ? (SDL_Color){  0, 200, 140, 255}
                : (SDL_Color){ 80, 130, 160, 255};
            draw_text(ren, font, fec_buf, x, STAT_Y + 4, fc);
        } else {
            draw_text(ren, font, "FEC: off  (-e to enable)",
                      x, STAT_Y + 4, dim);
        }
    }

    /* ── Timing recovery + carrier PLL row (SYNC_Y) ─────────────────
     *
     * Timing recovery
     * ───────────────
     * The receiver window (4096 samples) divides the symbol period
     * (16 384 samples) exactly 4 times, so there are always 4 windows
     * per symbol.  One of them will straddle the symbol boundary and
     * therefore sees a mixture of two phases — it must not be decoded.
     *
     * We detect boundaries by comparing consecutive complex carrier
     * values:   delta = ft_curr × conj(ft_prev)
     * If |∠delta| > π/M (half the symbol spacing), a symbol transition
     * crossed this window.  The sym_window counter then resets:
     *   0 = boundary window   (orange box — skip decoding)
     *   1 = first safe window (green  box — decode here)
     *   2 = second safe       (green  box — decode here)
     *   3 = third safe        (green  box — decode here)
     *
     * Carrier Phase PLL
     * ─────────────────
     * After each clean (non-boundary) symbol decision the PLL computes
     * the residual phase error between the corrected received phase and
     * the ideal constellation angle:
     *   error = corrected_phase − (current_sym × 2π/M)
     * and folds it into the phase offset estimate:
     *   phase_offset += PLL_ALPHA × error
     * This tracks any slow drift that the one-shot preamble calibration
     * could not capture, keeping the constellation aligned over time.
     * ─────────────────────────────────────────────────────────────── */
    {
        SDL_Color dim  = { 80,  80, 110, 255};
        SDL_Color med  = {130, 130, 165, 255};

        draw_text(ren, font, "Timing:", CHART_X, SYNC_Y + 4, dim);

        /* ── 4-box symbol window indicator ──────────────────────── */
        const int BOX_W = 22, BOX_H = 22, BOX_GAP = 3;
        int bx = CHART_X + 58;

        for (int w = 0; w < sym_windows_cfg; w++) {
            int is_cur = (timing_locked && w == sym_window);

            Uint8 r, g, b;
            if (!timing_locked) {
                r = 40; g = 40; b = 60;          /* grey  — not acquired   */
            } else if (w == 0) {                  /* boundary slot          */
                /* Distinguish actively-detected boundary (bright red) from
                 * the expected counter-rollover position (dim orange).     */
                int active_bnd = (is_cur && is_boundary);
                r = active_bnd ? 220 : (is_cur ? 160 : 80);
                g = active_bnd ?  45 : (is_cur ?  90 : 45);
                b = active_bnd ?  15 : (is_cur ?  15 : 10);
            } else if (w == decode_window_idx) {  /* preferred decode slot  */
                r = is_cur ?  80 : 30;
                g = is_cur ? 170 : 90;
                b = is_cur ?  30 : 15;
            } else {                              /* clean safe slot        */
                r = is_cur ?  30 : 18;
                g = is_cur ? 190 : 110;
                b = is_cur ?  80 : 40;
            }

            SDL_SetRenderDrawColor(ren, r, g, b, 255);
            SDL_Rect box = {bx, SYNC_Y, BOX_W, BOX_H};
            SDL_RenderFillRect(ren, &box);

            /* White border on the currently active window */
            if (is_cur) {
                SDL_SetRenderDrawColor(ren, 210, 210, 210, 255);
                SDL_RenderDrawRect(ren, &box);
            }

            SDL_Color lc = {190, 190, 210, 255};
            char box_lbl[8];
            if (w == 0) {
                memcpy(box_lbl, "B", 2);
            } else {
                int n = snprintf(box_lbl, sizeof(box_lbl), "%d", w);
                if (n < 0 || n >= (int)sizeof(box_lbl)) {
                    memcpy(box_lbl, "*", 2);
                }
            }
            draw_text(ren, font, box_lbl,
                      bx + (BOX_W - 7) / 2, SYNC_Y + 4, lc);
            bx += BOX_W + BOX_GAP;
        }

        /* Timing lock status */
        {
            const char *lock_str = timing_locked ? "LOCKED" : "ACQUIRING";
            SDL_Color lc = timing_locked
                ? (SDL_Color){  0, 205, 120, 255}
                : (SDL_Color){190, 135,   0, 255};
            draw_text(ren, font, lock_str, bx + 8, SYNC_Y + 4, lc);
        }

        /* ── PLL error bar ──────────────────────────────────────── */
        int pll_x = bx + 90;
        draw_text(ren, font, "PLL:", pll_x, SYNC_Y + 4, dim);
        pll_x += 38;

        const int BAR_W = 140, BAR_H = 10;
        const int bar_cy = SYNC_Y + (BOX_H - BAR_H) / 2;
        const int bar_cx = pll_x + BAR_W / 2;

        /* Bar track */
        SDL_SetRenderDrawColor(ren, 28, 28, 48, 255);
        SDL_Rect bar_bg = {pll_x, bar_cy, BAR_W, BAR_H};
        SDL_RenderFillRect(ren, &bar_bg);

        /* Centre tick */
        SDL_SetRenderDrawColor(ren, 75, 75, 110, 255);
        SDL_RenderDrawLine(ren, bar_cx, bar_cy - 2,
                                bar_cx, bar_cy + BAR_H + 2);

        /* Error marker — range is ±(180/M)°, i.e., ±half a symbol */
        if (timing_locked && rx_state == 2) {
            float max_err_deg = 180.0f / (float)M;
            float norm = pll_error_deg / max_err_deg;
            if (norm >  1.0f) norm =  1.0f;
            if (norm < -1.0f) norm = -1.0f;

            int mx = bar_cx + (int)(norm * (float)(BAR_W / 2));

            float abs_n = norm < 0 ? -norm : norm;
            Uint8 er = abs_n < 0.3f ?   0 : abs_n < 0.6f ? 190 : 215;
            Uint8 eg = abs_n < 0.3f ? 195 : abs_n < 0.6f ? 170 :  55;
            Uint8 eb = 0;

            SDL_SetRenderDrawColor(ren, er, eg, eb, 255);
            SDL_Rect mk = {mx - 2, bar_cy - 2, 5, BAR_H + 4};
            SDL_RenderFillRect(ren, &mk);
        }

        /* Error value in degrees */
        {
            char pll_str[24];
            if (timing_locked && rx_state == 2)
                snprintf(pll_str, sizeof(pll_str),
                         "%+.1f\xc2\xb0", pll_error_deg);
            else
                snprintf(pll_str, sizeof(pll_str), "---");
            draw_text(ren, font, pll_str,
                      pll_x + BAR_W + 6, SYNC_Y + 4, med);
        }
    }

    /* Caller presents after rendering both panels */
}

/* ═══════════════════════════════════════════════════════════════════
 * find_input_device
 *
 * Enumerate PortAudio devices and pick the best input source.
 *
 * Priority:
 *   1. Non-skipped device at SAMPLE_RATE with fewest input channels.
 *      (Fewer channels ≈ specific PipeWire virtual source at 2 ch
 *       rather than a broad virtual mixer at 64–128 ch.)
 *   2. Any device at SAMPLE_RATE (including skipped-name devices).
 *   3. Any device with at least one input channel (last resort).
 *
 * "default", "dmix", "sysdefault" are skipped in pass 1 because the
 * ALSA virtual mixing layer fails or is silent when PipeWire owns the
 * hardware.
 * ═══════════════════════════════════════════════════════════════════ */
static PaDeviceIndex find_input_device(void)
{
    PaDeviceIndex num_devices = Pa_GetDeviceCount();
    if (num_devices < 0) {
        fprintf(stderr, "Pa_GetDeviceCount: %s\n",
                Pa_GetErrorText((PaError)num_devices));
        return paNoDevice;
    }

    static const char *const skip_names[] = {
        "default", "dmix", "sysdefault", NULL
    };

    PaDeviceIndex best_rate     = paNoDevice;
    int           best_rate_ch  = INT_MAX;
    PaDeviceIndex fallback_rate = paNoDevice;
    PaDeviceIndex any_input     = paNoDevice;

    for (PaDeviceIndex i = 0; i < num_devices; i++) {
        const PaDeviceInfo *info = Pa_GetDeviceInfo(i);
        if (!info || info->maxInputChannels <= 0) continue;

        if (any_input == paNoDevice) any_input = i;

        int skip = 0;
        for (int s = 0; skip_names[s]; s++) {
            if (strncmp(info->name, skip_names[s],
                        strlen(skip_names[s])) == 0) {
                skip = 1; break;
            }
        }

        if ((int)info->defaultSampleRate == SAMPLE_RATE) {
            if (fallback_rate == paNoDevice) fallback_rate = i;
            if (!skip && info->maxInputChannels < best_rate_ch) {
                best_rate    = i;
                best_rate_ch = info->maxInputChannels;
            }
        }
    }

    PaDeviceIndex chosen =
        (best_rate     != paNoDevice) ? best_rate  :
        (fallback_rate != paNoDevice) ? fallback_rate :
                                        any_input;

    if (chosen != paNoDevice) {
        const PaDeviceInfo *info = Pa_GetDeviceInfo(chosen);
        fprintf(stderr,
                "Using input device [%d]: %s  (%.0f Hz, %d ch)\n",
                chosen,
                info ? info->name             : "unknown",
                info ? info->defaultSampleRate : 0.0,
                info ? info->maxInputChannels  : 0);
    } else {
        fprintf(stderr, "No input device found.\n");
    }
    return chosen;
}

/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /* ── argument parsing ─────────────────────────────────────────── */
    int            M            = 4;   /* default: QPSK */
    int            fec_enabled  = 0;
    int            fec_depth    = PSK_DEFAULT_ILVE_DEPTH;
    int            carrier_hz   = DEFAULT_CARRIER_CYCLES * SAMPLE_RATE / 256;
    int            buffers_per_symbol = 64;
    int            num_carriers = DEFAULT_NUM_CARRIERS;
    const char    *output_file  = NULL; /* NULL → no file output        */
    PaDeviceIndex  forced_device = paNoDevice;
    const char    *name_pattern  = NULL;

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-e") == 0) {
            fec_enabled = 1;

        } else if (strcmp(argv[a], "-o") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -o requires a filename.\n");
                return 1;
            }
            output_file = argv[++a];

        } else if (strcmp(argv[a], "-d") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -d requires a depth argument.\n");
                return 1;
            }
            fec_depth = atoi(argv[++a]);
            if (fec_depth < 1) {
                fprintf(stderr, "Error: depth must be >= 1.\n");
                return 1;
            }

        } else if (strcmp(argv[a], "--list-devices") == 0) {
            PaError ie = Pa_Initialize();
            if (ie != paNoError) {
                fprintf(stderr, "Pa_Initialize: %s\n", Pa_GetErrorText(ie));
                return 1;
            }
            PaDeviceIndex n = Pa_GetDeviceCount();
            printf("%-4s  %-4s  %-8s  %s\n", "Idx","InCh","Rate(Hz)","Name");
            printf("----  ----  --------  ----\n");
            for (PaDeviceIndex i = 0; i < n; i++) {
                const PaDeviceInfo *di = Pa_GetDeviceInfo(i);
                if (!di || di->maxInputChannels <= 0) continue;
                printf("%-4d  %-4d  %-8.0f  %s\n",
                       i, di->maxInputChannels,
                       di->defaultSampleRate, di->name);
            }
            Pa_Terminate();
            return 0;

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
                        "Error: invalid -s value '%s' (must be integer >= %d and divisible by 16).\n",
                        argv[a], 32);
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

        } else if (strcmp(argv[a], "--device") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: --device requires a device index.\n");
                return 1;
            }
            char *ep;
            long idx = strtol(argv[++a], &ep, 10);
            if (*ep != '\0' || idx < 0) {
                fprintf(stderr, "Error: invalid device index '%s'.\n", argv[a]);
                return 1;
            }
            forced_device = (PaDeviceIndex)idx;

        } else if (strcmp(argv[a], "-n") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "Error: -n requires a name substring.\n");
                return 1;
            }
            name_pattern = argv[++a];

        } else {
            fprintf(stderr, "Unknown argument: %s\n"
                    "Usage: %s [-m M] [-c FREQ_HZ] [-s BUFS] [-k CARRIERS] [-e] [-d DEPTH] [-o OUTPUT_FILE]\n"
                    "          [--device <idx>] [-n <substr>] [--list-devices]\n"
                    "  -c HZ     carrier frequency in Hz (integer, < %d)\n"
                    "  -s N      TX buffers/symbol (>= %d, divisible by 16)\n"
                    "  -k N      number of parallel carriers (1..%d, spacing %d Hz)\n"
                    "  -o FILE   write decoded bytes to FILE when carrier drops\n",
                    argv[a], argv[0], MAX_CARRIER_HZ, MIN_BUFFERS_PER_SYMBOL,
                    MAX_NUM_CARRIERS, CARRIER_SPACING_HZ);
            return 1;
        }
    }

    if (((long)carrier_hz * 256) % SAMPLE_RATE != 0) {
        int nearest = nearest_compatible_hz(carrier_hz);
        fprintf(stderr,
                "Error: carrier %d Hz is incompatible with %d Hz sample rate and 256-sample TX buffers.\n",
                carrier_hz, SAMPLE_RATE);
        fprintf(stderr,
                "       Choose a frequency where (Hz * 256) is divisible by %d.\n",
                SAMPLE_RATE);
        fprintf(stderr,
                "       Nearest compatible frequency: %d Hz.\n",
                nearest);
        fprintf(stderr,
                "       Examples: 750 Hz, 1125 Hz, 1500 Hz.\n");
        return 1;
    }

    int carrier_cycles = carrier_hz * 256 / SAMPLE_RATE;
    int spacing_cycles = CARRIER_SPACING_HZ * 256 / SAMPLE_RATE;
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
        if (((long)hz_c * 256) % SAMPLE_RATE != 0) {
            fprintf(stderr,
                    "Error: derived carrier %d Hz is incompatible with sample timing.\n",
                    hz_c);
            return 1;
        }
    }

    int sym_windows_cfg = (buffers_per_symbol * 256) / FRAMES_PER_BUFFER;
    int decode_window_idx = sym_windows_cfg / 2;
    if (decode_window_idx < 1) decode_window_idx = 1;
    g_num_carriers = num_carriers;
    for (int c = 0; c < num_carriers; c++) {
        int off = carrier_offset_index(c);
        int cyc = carrier_cycles + off * spacing_cycles;
        g_carrier_bins[c] = cyc * CARRIER_BIN_PER_CYCLE;
        g_carrier_hzs[c]  = (float)cyc * SAMPLE_RATE / 256.0f;
    }
    g_carrier_bin = g_carrier_bins[0];
    g_carrier_hz  = g_carrier_hzs[0];

    /* ── precompute carrier phasor table ──────────────────────────── */
    for (int c = 0; c < num_carriers; c++) {
        for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
            exptable[c][i] = cexpf(-I * 2.0f * (float)M_PI
                                    * (float)g_carrier_bins[c] / (float)FRAMES_PER_BUFFER
                                    * (float)i);
        }
    }

    fprintf(stderr,
            "%d-PSK receiver — %d carrier%s, primary bin %d (~%.0f Hz), %d bit%s/symbol each, %d win/sym\n",
            M, num_carriers, num_carriers == 1 ? "" : "s",
            g_carrier_bin, g_carrier_hz,
            ilog2(M), ilog2(M) == 1 ? "" : "s", sym_windows_cfg);

    /* ── SDL2 + SDL_ttf ──────────────────────────────────────────── */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    if (TTF_Init() < 0) {
        fprintf(stderr, "TTF_Init: %s\n", TTF_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow(
        "M-PSK Receiver — FFT + IQ Constellation",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        TTF_Quit(); SDL_Quit();
        return 1;
    }

    SDL_Renderer *ren = SDL_CreateRenderer(win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) {
        fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(win); TTF_Quit(); SDL_Quit();
        return 1;
    }

    /* ── monospace font ──────────────────────────────────────────── */
    TTF_Font *font = NULL;
    static const char *const font_paths[] = {
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        NULL
    };
    for (int fp = 0; font_paths[fp] && !font; fp++)
        font = TTF_OpenFont(font_paths[fp], 21);
    if (!font)
        fprintf(stderr,
                "Warning: no font found — text labels will be absent.\n");

    /* ── PortAudio ───────────────────────────────────────────────── */
    {
        PaError ie = Pa_Initialize();
        if (ie != paNoError) {
            fprintf(stderr, "Pa_Initialize: %s\n", Pa_GetErrorText(ie));
            if (font) TTF_CloseFont(font);
            TTF_Quit(); SDL_DestroyRenderer(ren);
            SDL_DestroyWindow(win); SDL_Quit();
            return 1;
        }
    }

    /* ── device selection ────────────────────────────────────────── */
    PaDeviceIndex input_device = paNoDevice;

    if (forced_device != paNoDevice) {
        const PaDeviceInfo *di = Pa_GetDeviceInfo(forced_device);
        if (!di || di->maxInputChannels <= 0) {
            fprintf(stderr, "Error: device %d is invalid or has no inputs.\n",
                    forced_device);
            goto cleanup_pa;
        }
        fprintf(stderr, "Using forced device [%d]: %s  (%.0f Hz, %d ch)\n",
                forced_device, di->name,
                di->defaultSampleRate, di->maxInputChannels);
        input_device = forced_device;

    } else if (name_pattern != NULL) {
        PaDeviceIndex n = Pa_GetDeviceCount();
        for (PaDeviceIndex i = 0; i < n && input_device == paNoDevice; i++) {
            const PaDeviceInfo *di = Pa_GetDeviceInfo(i);
            if (di && di->maxInputChannels > 0 &&
                strstr(di->name, name_pattern))
                input_device = i;
        }
        if (input_device == paNoDevice) {
            fprintf(stderr, "Error: no input device matching '%s'.\n",
                    name_pattern);
            goto cleanup_pa;
        }
        const PaDeviceInfo *di = Pa_GetDeviceInfo(input_device);
        fprintf(stderr, "Using named device [%d]: %s  (%.0f Hz, %d ch)\n",
                input_device, di->name,
                di->defaultSampleRate, di->maxInputChannels);

    } else {
        input_device = find_input_device();
    }

    if (input_device == paNoDevice) {
        fprintf(stderr, "No suitable input device found.\n");
        goto cleanup_pa;
    }

    /* ── open and start stream ───────────────────────────────────── */
    {
        PaStreamParameters ip;
        ip.device                    = input_device;
        ip.channelCount              = 1;
        ip.sampleFormat              = paFloat32;
        ip.suggestedLatency          =
            Pa_GetDeviceInfo(input_device)->defaultHighInputLatency;
        ip.hostApiSpecificStreamInfo = NULL;

        callback_data_t ud = {0, SIZE_MAX};

        PaStream *stream;
        PaError   pe;

        pe = Pa_OpenStream(&stream, &ip, NULL,
                           SAMPLE_RATE, FRAMES_PER_BUFFER,
                           paClipOff, audio_callback, &ud);
        if (pe != paNoError) {
            fprintf(stderr, "Pa_OpenStream: %s\n", Pa_GetErrorText(pe));
            goto cleanup_pa;
        }

        pe = Pa_StartStream(stream);
        if (pe != paNoError) {
            fprintf(stderr, "Pa_StartStream: %s\n", Pa_GetErrorText(pe));
            Pa_CloseStream(stream);
            goto cleanup_pa;
        }

        /* ── main event / render loop ─────────────────────────────── */
        float local_samples[FRAMES_PER_BUFFER];
        float analysis_magnitudes[BAR_WIDTH];
        float display_magnitudes[BAR_WIDTH];
        float peaks[BAR_WIDTH];

        memset(analysis_magnitudes, 0, sizeof(analysis_magnitudes));
        memset(display_magnitudes, 0, sizeof(display_magnitudes));
        memset(peaks,      0, sizeof(peaks));

        /* Channel quality state */
        float snr_db      = -60.0f;
        int   interference = 0;
        int   intf_bin     = 0;

        /* FEC decode accumulator — collects raw symbol bytes, then
         * deinterleaves + Hamming-decodes when the carrier drops.  */
        uint8_t *fec_sym_buf     = NULL;   /* grows as symbols arrive  */
        size_t   fec_sym_cap     = 0;
        size_t   fec_sym_bits    = 0;      /* exact bit count (NOT fec_sym_len*8)
                                            * fec_sym_len rounds UP to bytes so
                                            * using it as the next-bit position
                                            * would skip bits 2-7 of every byte */
        int      prev_timing_locked = 0;   /* detect the first timing-lock edge */

        /* skip_after_lock: how many symbol boundaries to ignore before
         * the accumulator starts saving data bits.
         *
         * Set to 1 when timing first locks so that the alignment marker
         * (last preamble symbol, phase 180°) is discarded and the
         * accumulator begins exactly at the first TRUE data symbol.
         *
         * Decremented each time a new symbol boundary is detected while
         * timing is already locked, giving the marker one full symbol
         * period of grace regardless of when the boundary falls within
         * the receiver window.
         */
        int      skip_after_lock = 0;

        /* ── Decoded-output live preview ───────────────────────────
         *
         * preview_buf holds the printable representation of every byte
         * decoded so far in the current transmission.  It is rebuilt
         * from fec_sym_buf on each frame that carries new data.
         *
         * Non-printable bytes → middle dot  (·, U+00B7, 0xC2 0xB7).
         * '\n' and '\t' are preserved literally so formatted text looks
         * natural in the panel.
         *
         * For FEC mode psk_fec_decode() is called on the full
         * accumulated buffer; only complete interleave blocks are
         * returned, so the display shows partial-but-correct output as
         * the transmission progresses.
         */
#define PREVIEW_MAX 8192
        char   preview_buf[PREVIEW_MAX + 4];  /* +4: cursor glyph headroom */
        int    preview_chars    = 0;
        int    preview_bytes    = 0;
        int    preview_corr     = 0;
        size_t preview_src_len  = 0;   /* fec_sym_len at last rebuild  */
        int    prev_rx_state    = 0;
        memset(preview_buf, 0, sizeof(preview_buf));
        size_t   fec_sym_len     = 0;      /* bytes used               */
        int      fec_corrections = 0;      /* running correction count */

        /* IQ history (calibrated carrier phasors) */
        IQ_Point iq_hist[MAX_NUM_CARRIERS][IQ_HISTORY];
        int      iq_head  = 0;
        int      iq_count = 0;
        memset(iq_hist, 0, sizeof(iq_hist));

        /* M-PSK decode state */
        int   rx_state     = 0;       /* 0=WAITING, 1=CALIBRATING, 2=DECODING */
        float cal_cos      = 0.0f;    /* phase-average accumulators            */
        float cal_sin      = 0.0f;
        int   cal_count    = 0;
        float cal_mag      = 0.0f;
        float phase_offset = 0.0f;    /* calibrated phase reference (radians)  */
        float ref_mag      = 1.0f;    /* calibrated reference magnitude        */
        int   current_sym  = 0;
        int   current_bits = 0;
        int   current_sym_c[MAX_NUM_CARRIERS] = {0};

        /* ── Timing recovery state ──────────────────────────────────
         *
         * sym_window   — position within the 4-window symbol cycle.
         *                0 = boundary window (straddles two symbols)
         *                1, 2, 3 = safe windows (within one symbol)
         *                -1 = not yet acquired (no boundary seen yet)
         *
         * is_boundary  — set when the current window's phase jump
         *                exceeds π/M, indicating a symbol transition.
         *
         * ft_prev_*    — complex carrier value from the previous window,
         *                used to compute the inter-window phase jump.
         *
         * timing_locked — true once the first boundary is detected and
         *                 the sym_window counter is synchronised.
         */
        int   sym_window    = -1;     /* -1 = not yet acquired              */
        int   is_boundary   =  0;
        int   timing_locked =  0;
        float ft_prev_re    = 0.0f;
        float ft_prev_im    = 0.0f;

        /* ── Symbol-aligned window extraction ────────────────────────
         *
         * last_boundary_sample — absolute sample position when the most
         *                        recent boundary was detected.
         * use_aligned_windows  — enable aligned extraction after first
         *                        boundary is detected.
         *
         * When timing is locked, we extract windows at symbol-aligned
         * positions: last_boundary_sample + (sym_window × 4096)
         * This ensures we decode from the true middle of each symbol.
         */
        size_t last_boundary_sample = 0;
        int    use_aligned_windows   = 0;

        /* ── Carrier PLL state ──────────────────────────────────────
         *
         * pll_error_deg — most recent phase error (degrees) used to
         *                 update phase_offset via the first-order loop.
         *                 Displayed on the PLL error bar.
         */
        float pll_error_deg = 0.0f;

        /* Local copies of shared data */
        float local_ft_re[MAX_NUM_CARRIERS] = {0};
        float local_ft_im[MAX_NUM_CARRIERS] = {0};
        float carrier_mag_c[MAX_NUM_CARRIERS] = {0};
        float carrier_mag = 0.0f;
        float live_rate_bps = 0.0f;
        uint32_t rate_start_ticks = 0;
        size_t   rate_start_bits  = 0;
        int current_bits_c[MAX_NUM_CARRIERS] = {0};

        /* Initial blank frame */
        render_frame(ren, font, display_magnitudes, peaks,
                     0.0f, 0.0f, 0, 0, rx_state, M, 0.0f,
                     sym_windows_cfg, decode_window_idx,
                     sym_window, is_boundary, timing_locked, pll_error_deg,
                     snr_db, interference, intf_bin,
                     fec_enabled, fec_corrections, fec_depth);
        render_iq_panels(ren, font, iq_hist, iq_head, iq_count,
                         M, current_sym_c, rx_state,
                         timing_locked, sym_window, num_carriers);
        render_decoded_output(ren, font,
                               preview_buf, preview_chars,
                               preview_bytes, preview_corr,
                               fec_enabled, rx_state);
        SDL_RenderPresent(ren);

        int running = 1;
        while (running && Pa_IsStreamActive(stream)) {

            /* ── SDL events ─────────────────────────────────────── */
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT) running = 0;
                if (ev.type == SDL_KEYDOWN) {
                    SDL_Keycode k = ev.key.keysym.sym;
                    if (k == SDLK_q || k == SDLK_ESCAPE) running = 0;
                }
            }

            /* ── poll shared buffer ─────────────────────────────── */
            int got = 0;
            if (pthread_mutex_trylock(&s_mutex) == 0) {
                if (s_new_data) {
                    memcpy(local_samples, s_samples, sizeof(local_samples));
                    for (int c = 0; c < num_carriers; c++) {
                        local_ft_re[c] = s_ft_re[c];
                        local_ft_im[c] = s_ft_im[c];
                        carrier_mag_c[c] = s_mag[c];
                    }
                    carrier_mag = carrier_mag_c[0];
                    s_new_data  = 0;
                    got         = 1;
                }
                pthread_mutex_unlock(&s_mutex);
            }

            if (!got) {
                SDL_Delay(5);
                continue;
            }

            /* ── compute FFT data for analysis + display ─────────── */
            compute_dft(local_samples, FRAMES_PER_BUFFER, analysis_magnitudes);
            compute_display_spectrum(local_samples, FRAMES_PER_BUFFER,
                                     display_magnitudes);



            /* ── channel quality estimates ──────────────────────── */
            if (g_carrier_bin >= 0 && g_carrier_bin < BAR_WIDTH) {
                snr_db = psk_snr_estimate_db(analysis_magnitudes, BAR_WIDTH,
                                             g_carrier_bin);
                interference = psk_detect_interference(analysis_magnitudes, BAR_WIDTH,
                                                       g_carrier_bin, &intf_bin);
            } else {
                snr_db = -60.0f;
                interference = 0;
                intf_bin = 0;
            }

            /* ── derived carrier quantities ─────────────────────── */
            float received_phase    = atan2f(local_ft_im[0], local_ft_re[0]);
            float carrier_phase_deg = received_phase * 180.0f / (float)M_PI;

            /* ── Symbol-aligned window extraction ───────────────────
             *
             * If timing is locked and aligned extraction is enabled,
             * extract a window from the circular buffer at the proper
             * symbol-aligned position and recompute the carrier DFT.
             *
             * This solves the fundamental issue: PortAudio windows have
             * arbitrary timing with no relation to symbol boundaries.
             */
            if (use_aligned_windows && timing_locked && sym_window == decode_window_idx) {
                /* Calculate target sample position - always extract window 2
                 * (the 3rd window after boundary, which is in the middle of the symbol) */
                size_t target_offset = (size_t)decode_window_idx * FRAMES_PER_BUFFER;
                size_t target_sample = last_boundary_sample + target_offset;

                pthread_mutex_lock(&slide_mutex);

                /* Check if buffer contains enough data */
                if (slide_sample_count >= target_sample + FRAMES_PER_BUFFER) {
                    /* Extract window from circular buffer */
                    size_t samples_back = slide_sample_count - target_sample;
                    size_t read_start = (slide_write_pos + SLIDE_BUF_SIZE - samples_back) % SLIDE_BUF_SIZE;

                    float aligned_window[FRAMES_PER_BUFFER];
                    for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
                        aligned_window[i] = slide_buf[(read_start + i) % SLIDE_BUF_SIZE];
                    }

                    /* Recompute DFT on aligned window */
                    for (int c = 0; c < num_carriers; c++) {
                        float complex ft_aligned = 0.0f;
                        for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
                            ft_aligned += aligned_window[i] * exptable[c][i];
                        }
                        local_ft_re[c] = crealf(ft_aligned);
                        local_ft_im[c] = cimagf(ft_aligned);
                        carrier_mag_c[c] = cabsf(ft_aligned);
                    }
                    carrier_mag = carrier_mag_c[0];
                    received_phase = atan2f(local_ft_im[0], local_ft_re[0]);
                    carrier_phase_deg = received_phase * 180.0f / (float)M_PI;
                }

                pthread_mutex_unlock(&slide_mutex);
            }

            /* ═══════════════════════════════════════════════════════
             * TIMING RECOVERY  —  symbol boundary detection
             * ═══════════════════════════════════════════════════════
             *
             * Each receiver window is 4096 samples.  The transmitter
             * symbol period is 16 384 samples = exactly 4 windows.
             * One of those four windows will straddle the boundary
             * between two consecutive symbols; its DFT value is a
             * weighted mix of two phases and must not be decoded.
             *
             * Detection strategy — phase-jump comparator:
             *   delta       = ft_curr × conj(ft_prev)
             *   delta_phase = ∠(delta)   ∈ (−π, π]
             *
             * Because the carrier completes exactly CARRIER_BIN full
             * cycles per window (64 cycles in 4096 samples), two
             * consecutive same-symbol windows give delta_phase ≈ 0.
             * A symbol transition injects a phase step of k×(2π/M),
             * so |delta_phase| exceeds the threshold π/M.
             *
             * On detection the sym_window counter resets to 0
             * (boundary slot).  It then counts up: 1, 2, 3 (safe
             * slots).  Decoding and PLL updates are gated to safe
             * slots only.
             * ═══════════════════════════════════════════════════════ */
            {
                float complex ft_curr = local_ft_re[0] + I * local_ft_im[0];
                float complex ft_prev = ft_prev_re  + I * ft_prev_im;

                /* Run the comparator whenever the current window has a
                 * detected carrier — using the same SNR-inclusive gate as
                 * the calibration / decoding state machine.  The previous
                 * window only needs to exceed the low absolute floor
                 * (PSK_SIG_THRESHOLD) to be a valid phase reference; we
                 * cannot check its SNR retrospectively. */

                if ((snr_db >= CARRIER_SNR_MIN_DB || carrier_mag > SIG_THRESHOLD) &&
                    cabsf(ft_prev) > 0.5f) {

                    float complex delta = ft_curr * conjf(ft_prev);
                    float dp = atan2f(cimagf(delta), crealf(delta));
                    float thresh = (float)M_PI / (float)M;

                    if (fabsf(dp) > thresh) {
                        /* Symbol boundary detected in this window */
                        is_boundary   = 1;
                        sym_window    = 0;

                        /* Update boundary position for aligned window extraction
                         * This happens on EVERY boundary so we track the current symbol */
                        pthread_mutex_lock(&slide_mutex);
                        last_boundary_sample = slide_sample_count - FRAMES_PER_BUFFER;
                        pthread_mutex_unlock(&slide_mutex);

                        if (!timing_locked) {
                            use_aligned_windows = 1;
                        }
                        timing_locked = 1;
                    } else {
                        is_boundary = 0;
                        if (timing_locked)
                            sym_window = (sym_window + 1) % sym_windows_cfg;
                    }
                } else {
                    is_boundary = 0;
                    /* Don't advance counter without a valid reference */
                }

                /* Save current complex value for next frame */
                ft_prev_re = local_ft_re[0];
                ft_prev_im = local_ft_im[0];
            }

            /* ── Reset accumulator on first timing lock ───────────────
             *
             * Before timing_locked becomes 1 the accumulator samples
             * every post-calibration window, including the remaining
             * preamble windows.  Those preamble bits land at a
             * non-byte-aligned position (9 windows × 2 bits = 18 bits
             * for QPSK) which corrupts every output byte.
             *
             * When the first symbol boundary is detected (preamble →
             * first data symbol), we discard those preamble bits by
             * zeroing the accumulator.  Subsequent data bits then start
             * at position 0, cleanly byte-aligned.
             */
            if (timing_locked && !prev_timing_locked) {
                /* First timing lock: preamble → alignment marker boundary.
                 * Reset accumulator so preamble bits are discarded, and
                 * arm the skip counter to ignore the marker symbol itself. */
                fec_sym_len     = 0;
                fec_sym_bits    = 0;
                skip_after_lock = 1;
                if (fec_sym_buf) memset(fec_sym_buf, 0, fec_sym_cap);

                /* Reset live preview so preamble zeros don't show */
                preview_chars   = 0;
                preview_bytes   = 0;
                preview_corr    = 0;
                preview_src_len = 0;
                memset(preview_buf, 0, sizeof(preview_buf));
            }

            /* Decrement skip counter on each subsequent symbol boundary
             * (i.e. when timing was already locked in the previous frame).
             * This gives the alignment marker exactly one symbol period
             * before the accumulator opens for data. */
            if (is_boundary && timing_locked && prev_timing_locked
                && skip_after_lock > 0) {
                skip_after_lock--;
            }

            prev_timing_locked = timing_locked;

            /* ═══════════════════════════════════════════════════════
             * CARRIER STATE MACHINE  (WAITING → CALIBRATING → DECODING)
             * ═══════════════════════════════════════════════════════
             *
             * Phase offset (φ_offset)
             * ────────────────────────
             * The received phase equals:
             *   φ_rx = φ_tx + φ_offset
             *
             * φ_offset is CONSTANT for the session (the carrier
             * completes exactly CARRIER_BIN cycles per window, so
             * the DFT phase advances by an exact multiple of 2π
             * each callback and never accumulates drift).
             *
             * The preamble transmits φ_tx = 0°, so the circular
             * mean of φ_rx over CAL_WINDOWS frames equals φ_offset.
             *
             * Decision-directed carrier PLL
             * ──────────────────────────────
             * After the one-shot preamble calibration, slow drift
             * (thermal, quantisation) is corrected by the PLL.
             * For each clean (non-boundary, DECODING) frame:
             *   1. Make the nearest-symbol decision.
             *   2. Compute residual error:
             *        e = corrected_phase − (sym × 2π/M)
             *        wrapped to (−π, π].
             *   3. Integrate into the phase estimate:
             *        φ_offset += PLL_ALPHA × e
             * This is a first-order loop (proportional gain only).
             * It converges to the true offset with a time constant
             * of roughly  1 / PLL_ALPHA  frames ≈ 40 frames ≈ 3 s.
             * ═══════════════════════════════════════════════════════ */
            /* Carrier present when SNR is clearly above noise floor
             * (primary OTA criterion) OR the signal exceeds the old
             * absolute floor (preserves software-loopback behaviour). */

           if (snr_db >= CARRIER_SNR_MIN_DB || carrier_mag > SIG_THRESHOLD) {

                /* Transition from WAITING: begin calibration */
                if (rx_state == 0) {
                    rx_state  = 1;
                    cal_cos   = 0.0f;
                    cal_sin   = 0.0f;
                    cal_mag   = 0.0f;
                    cal_count = 0;

                    /* Reset timing recovery phase history to prevent stale
                     * noise data from before transmission from causing false
                     * boundary detections on the first preamble window. */
                    ft_prev_re = 0.0f;
                    ft_prev_im = 0.0f;

                    /* Reset circular buffer to clear stale noise data from
                     * before this transmission started. This ensures the
                     * sliding buffer only contains samples from the current
                     * transmission, preventing corruption from old noise. */
                    pthread_mutex_lock(&slide_mutex);
                    slide_sample_count = 0;
                    slide_write_pos    = 0;
                    memset(slide_buf, 0, sizeof(slide_buf));
                    pthread_mutex_unlock(&slide_mutex);

                    /* Reset timing recovery state to prevent noise-induced
                     * false locks from before transmission from persisting.
                     * Without this, if timing locked onto noise while in
                     * WAITING state, those stale values would corrupt the
                     * new transmission's symbol alignment. */
                    timing_locked = 0;
                    use_aligned_windows = 0;
                    sym_window = -1;
                    last_boundary_sample = 0;
                    is_boundary = 0;
                }

                if (rx_state == 1) {
                    /* Accumulate received phases via circular mean.
                     * We do NOT skip boundary windows here: the
                     * preamble phase is constant (0°) so even a
                     * mixed boundary window averages correctly. */
                    cal_cos += cosf(received_phase);
                    cal_sin += sinf(received_phase);
                    cal_mag += carrier_mag;
                    if (++cal_count >= CAL_WINDOWS) {
                        phase_offset = atan2f(cal_sin, cal_cos);
                        ref_mag      = cal_mag / (float)CAL_WINDOWS;
                        rx_state     = 2;
                        fprintf(stderr,
                                "Calibrated.  Phase offset = %.1f°, ref_mag = %.1f\n",
                                phase_offset * 180.0f / (float)M_PI, ref_mag);
                    }
                }

                if (rx_state == 2) {
                    /* Correct for phase offset and map to [0, 2π) */
                    float corrected = received_phase - phase_offset;
                    corrected = fmodf(corrected + 4.0f * (float)M_PI,
                                      2.0f * (float)M_PI);

                    /* Map to nearest M-PSK symbol index */
                    int sym = (int)roundf(
                        (float)M * corrected / (2.0f * (float)M_PI));
                    int candidate_sym  = ((sym % M) + M) % M;
                    int candidate_bits = gray_decode(candidate_sym);

                    if (!is_boundary) {
                        /* Safe window: accept this decision */
                        current_sym  = candidate_sym;
                        current_bits = candidate_bits;
                        current_sym_c[0] = candidate_sym;
                        current_bits_c[0] = candidate_bits;

                        for (int c = 1; c < num_carriers; c++) {
                            float corrected_c = atan2f(local_ft_im[c], local_ft_re[c]) - phase_offset;
                            corrected_c = fmodf(corrected_c + 4.0f * (float)M_PI,
                                                2.0f * (float)M_PI);
                            int sym_c = (int)roundf(
                                (float)M * corrected_c / (2.0f * (float)M_PI));
                            int candidate_sym_c = ((sym_c % M) + M) % M;
                            current_sym_c[c] = candidate_sym_c;
                            current_bits_c[c] = gray_decode(candidate_sym_c);
                        }

                        /* ── PLL update ────────────────────────────
                         * Phase error = distance from the ideal angle
                         * for the decided symbol.  Wrapping with
                         * atan2(sin, cos) gives the shortest signed
                         * arc, avoiding the ±π discontinuity.        */
                        float ideal = (float)current_sym
                                    * 2.0f * (float)M_PI / (float)M;
                        float err   = corrected - ideal;
                        err = atan2f(sinf(err), cosf(err));
                        phase_offset += PLL_ALPHA * err;
                        pll_error_deg = err * 180.0f / (float)M_PI;
                    }
                    /* Boundary windows: keep previous symbol values;
                     * do not update the PLL (noisy mixed phase). */
                }

                /* Store phasor in IQ history.
                 * Both boundary and safe points are stored so the
                 * inter-symbol smear is visible in the constellation;
                 * boundary points are flagged for different colouring. */
                {
                    for (int c = 0; c < num_carriers; c++) {
                        float phase_c = atan2f(local_ft_im[c], local_ft_re[c]);
                        float store_phase = (rx_state == 2)
                                          ? (phase_c - phase_offset)
                                          : phase_c;

                        float amp = (rx_state == 2 && ref_mag > 0.0f)
                                  ? carrier_mag_c[c] / ref_mag
                                  : 1.0f;
                        IQ_Point p = {amp * cosf(store_phase), amp * sinf(store_phase),
                                      is_boundary};
                        iq_hist[c][iq_head] = p;
                    }
                    iq_head = (iq_head + 1) % IQ_HISTORY;
                    if (iq_count < IQ_HISTORY) iq_count++;
                }

            } else if (snr_db < CARRIER_SNR_MIN_DB && carrier_mag <= SIG_THRESHOLD) {
                /* Carrier absent (both SNR and absolute magnitude below
                 * their respective floors): abort calibration so the
                 * preamble is recaptured cleanly on the next transmission.
                 * DECODING is preserved across brief signal gaps. */
                if (rx_state == 1) {
                    rx_state = 0;
                    /* Reset aligned extraction flag to prevent stale state from
                     * noise-induced false locks from affecting next transmission */
                    use_aligned_windows = 0;
                    timing_locked = 0;
                    sym_window = -1;
                }
                is_boundary = 0;
            }

            /* ── Symbol accumulator ───────────────────────────────────
             *
             * Decoded symbol bits are packed into fec_sym_buf as a
             * running bitstream used for both the live preview and the
             * final file/stderr output.
             *
             * Guard conditions (all must be true):
             *   rx_state == 2          — in DECODING state
             *   !is_boundary           — not a symbol-boundary window
             *   carrier_mag > threshold— carrier is present (prevents
             *                           stale current_bits from being
             *                           accumulated after the carrier
             *                           drops, which caused the repeated
             *                           "Decoded 1 byte" spam)
             *   !timing_locked ||      — before timing locks, take all
             *   sym_window == decode   — after timing locks, take only
             *                           the configured decode window.
             *                           Without
             *                           this gate every symbol was
             *                           sampled 3 times, tripling the
             *                           bit count and corrupting the
             *                           recovered bytes.
             */
            if (rx_state == 2 && !is_boundary
                && (snr_db >= CARRIER_SNR_MIN_DB || carrier_mag > SIG_THRESHOLD)
                && (!timing_locked || sym_window == decode_window_idx)
                && skip_after_lock == 0) {
                int    bps      = ilog2(M);
                size_t bits_now = fec_sym_bits;   /* exact bit position */
                int    total_bps = bps * num_carriers;

                /* Grow buffer on demand */
                if (bits_now + (size_t)total_bps > fec_sym_cap * 8) {
                    size_t new_cap = fec_sym_cap + 256;
                    uint8_t *nb    = realloc(fec_sym_buf, new_cap);
                    if (nb) {
                        memset(nb + fec_sym_cap, 0, new_cap - fec_sym_cap);
                        fec_sym_buf = nb;
                        fec_sym_cap = new_cap;
                    }
                }
                if (fec_sym_buf) {
                    /* Append bps bits per carrier MSB-first.
                     * IMPORTANT: bits_now starts from fec_sym_bits, NOT
                     * fec_sym_len*8.  fec_sym_len is ceil(bits/8) so
                     * restarting from it would skip bits 2-7 of each
                     * partial byte, producing garbled output. */
                    for (int c = 0; c < num_carriers; c++) {
                        for (int b = bps - 1; b >= 0; b--) {
                            psk_setbit(fec_sym_buf, bits_now,
                                       (current_bits_c[c] >> b) & 1);
                            bits_now++;
                        }
                    }
                    fec_sym_bits = bits_now;
                    fec_sym_len  = (bits_now + 7) / 8;
                }
            }

            /* When carrier just dropped: decode accumulated buffer and
             * write output to the -o file (or print preview to stderr). */
            if (snr_db < CARRIER_SNR_MIN_DB && carrier_mag <= SIG_THRESHOLD
                && fec_sym_len > 0) {
                uint8_t *dec  = NULL;
                size_t   out_n = 0;
                int      corr  = 0;

                if (fec_enabled) {
                    dec = psk_fec_decode(fec_sym_buf, fec_sym_len,
                                         fec_depth, &out_n, &corr);
                    if (corr > 0) fec_corrections += corr;
                } else {
                    /* Raw mode: the bitstream is already decoded data.
                     * Duplicate the buffer so the free() path is uniform. */
                    out_n = fec_sym_len;
                    dec   = malloc(out_n);
                    if (dec) memcpy(dec, fec_sym_buf, out_n);
                }

                if (dec) {
                    /* Write to output file if -o was given */
                    if (output_file) {
                        if (psk_write_file(output_file, dec, out_n) == 0)
                            fprintf(stderr,
                                    "Output: wrote %zu byte(s) to '%s'.\n",
                                    out_n, output_file);
                    }

                    /* Always print a short stderr preview */
                    if (fec_enabled)
                        fprintf(stderr,
                                "Decoded %zu byte(s), %d correction(s): \"",
                                out_n, corr);
                    else
                        fprintf(stderr, "Decoded %zu byte(s): \"", out_n);
                    for (size_t i = 0; i < out_n && i < 64; i++) {
                        unsigned char ch = dec[i];
                        if (ch >= 32 && ch < 127) fputc((int)ch, stderr);
                        else fprintf(stderr, "\\x%02x", ch);
                    }
                    if (out_n > 64) fputs("…", stderr);
                    fputs("\"\n", stderr);

                    free(dec);
                }

                fec_sym_len        = 0;
                fec_sym_bits       = 0;   /* reset for next transmission */
                prev_timing_locked = 0;   /* allow re-lock on next preamble */
                timing_locked      = 0;   /* reset timing lock state */
                sym_window         = -1;  /* reset window tracking */
                skip_after_lock    = 0;
                use_aligned_windows = 0;  /* disable aligned extraction */

                /* Reset circular buffer state to prevent stale noise data
                 * and arithmetic overflow from persisting across transmissions.
                 * Without this reset, slide_sample_count grows unbounded and
                 * can cause incorrect buffer position calculations when a new
                 * transmission starts after the receiver has been idle. */
                pthread_mutex_lock(&slide_mutex);
                slide_sample_count = 0;
                slide_write_pos    = 0;
                memset(slide_buf, 0, sizeof(slide_buf));
                pthread_mutex_unlock(&slide_mutex);
            }

            /* ── live preview rebuild ──────────────────────────────
             *
             * Rebuild the decoded-output preview whenever the symbol
             * accumulator has grown since the last frame.  For non-FEC
             * mode fec_sym_buf already contains raw decoded bytes, so
             * we read them directly.  For FEC mode we run psk_fec_decode
             * on the full buffer; only complete interleave blocks are
             * returned, which avoids displaying garbled partial output.
             *
             * The preview is cleared when rx_state transitions from
             * WAITING (0) to CALIBRATING (1), i.e. a new carrier appears
             * and a fresh transmission is starting.
             */
            if (prev_rx_state == 0 && rx_state == 1) {
                /* New transmission: reset preview */
                preview_chars   = 0;
                preview_bytes   = 0;
                preview_corr    = 0;
                preview_src_len = 0;
                memset(preview_buf, 0, sizeof(preview_buf));

                rate_start_ticks = SDL_GetTicks();
                rate_start_bits  = fec_sym_bits;
                live_rate_bps    = 0.0f;
            }
            prev_rx_state = rx_state;

            if (rate_start_ticks != 0) {
                uint32_t now_ms = SDL_GetTicks();
                float elapsed_s = (float)(now_ms - rate_start_ticks) / 1000.0f;
                if (elapsed_s > 0.05f && fec_sym_bits >= rate_start_bits)
                    live_rate_bps =
                        (float)(fec_sym_bits - rate_start_bits) / elapsed_s;
            }

            if (rx_state == 0 && fec_sym_len == 0)
                rate_start_ticks = 0;

            if (fec_sym_len != preview_src_len && fec_sym_len > 0) {
                preview_src_len = fec_sym_len;

                /* Obtain the source bytes */
                const uint8_t *src     = NULL;
                size_t         src_len = 0;
                uint8_t       *fec_dec = NULL;

                if (fec_enabled) {
                    int    corr    = 0;
                    size_t dec_len = 0;
                    fec_dec = psk_fec_decode(fec_sym_buf, fec_sym_len,
                                             fec_depth, &dec_len, &corr);
                    if (fec_dec) {
                        src     = fec_dec;
                        src_len = dec_len;
                        preview_corr = corr;
                    }
                } else {
                    src     = fec_sym_buf;
                    src_len = fec_sym_len;
                }

                /* Convert bytes to display characters */
                if (src && src_len > 0) {
                    preview_chars = 0;
                    preview_bytes = (int)src_len;
                    memset(preview_buf, 0, sizeof(preview_buf));

                    for (size_t bi = 0;
                         bi < src_len && preview_chars < PREVIEW_MAX - 2;
                         bi++) {
                        unsigned char b = src[bi];
                        if (b >= 0x20 && b < 0x7F) {
                            /* Plain printable ASCII */
                            preview_buf[preview_chars++] = (char)b;
                        } else if (b == '\n' || b == '\r') {
                            preview_buf[preview_chars++] = '\n';
                        } else if (b == '\t') {
                            /* Expand tab to four spaces */
                            for (int sp = 0;
                                 sp < 4 && preview_chars < PREVIEW_MAX - 2;
                                 sp++)
                                preview_buf[preview_chars++] = ' ';
                        } else {
                            /* Non-printable → middle dot (U+00B7) */
                            preview_buf[preview_chars++] = '\xc2';
                            preview_buf[preview_chars++] = '\xb7';
                        }
                    }
                    preview_buf[preview_chars] = '\0';
                }

                if (fec_dec) free(fec_dec);
            }

            /* ── render both panels, then present ───────────────── */
            render_frame(ren, font, display_magnitudes, peaks,
                         carrier_mag, carrier_phase_deg,
                         current_sym, current_bits, rx_state, M, live_rate_bps,
                         sym_windows_cfg, decode_window_idx,
                         sym_window, is_boundary, timing_locked,
                         pll_error_deg,
                         snr_db, interference, intf_bin,
                         fec_enabled, fec_corrections, fec_depth);

            render_iq_panels(ren, font, iq_hist, iq_head, iq_count,
                             M, current_sym_c, rx_state,
                             timing_locked, sym_window, num_carriers);

            render_decoded_output(ren, font,
                                   preview_buf, preview_chars,
                                   preview_bytes, preview_corr,
                                   fec_enabled, rx_state);

            SDL_RenderPresent(ren);
        }

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        free(fec_sym_buf);
    }

cleanup_pa:
    Pa_Terminate();

    if (font) TTF_CloseFont(font);
    TTF_Quit();
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
