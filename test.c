/*
 * test.c — Comprehensive M-PSK test bench
 *
 * Tests every function in psk_common.h and exercises the complete
 * transmit → channel → receive pipeline in software (no PortAudio or
 * SDL2 required).
 *
 * Test categories
 * ───────────────
 *  1. Utility        — psk_is_power_of_2, psk_ilog2
 *  2. Gray coding    — encode, decode, roundtrip, adjacency property
 *  3. Bit packing    — bytes→symbols, symbols→bytes, roundtrip
 *  4. TX buffer      — phase, amplitude, inter-buffer continuity
 *  5. DFT            — magnitude, phase identity, off-freq rejection,
 *                      window-consistency across consecutive windows
 *  6. Phase decode   — ideal phases, noise margin for all M values
 *  7. Timing recovery— boundary detection, window counter, thresholds
 *  8. PLL            — convergence from a known phase error
 *  9. Loopback       — full software channel, zero BER (aligned),
 *                      bounded BER (timing-offset, real-world sim)
 *
 * Software channel model
 * ──────────────────────
 * The loopback test generates the complete audio stream that the
 * transmitter would play and feeds it to the receiver pipeline
 * window-by-window, entirely in memory.
 *
 *   psk_bytes_to_symbols          — bit packing  (transmitter side)
 *   psk_generate_tx_buffer ×64    — carrier audio (one symbol)
 *   psk_rx_process_window          — DFT + timing + PLL + decode
 *   psk_symbols_to_bytes          — bit unpacking (verification)
 *
 * A non-zero timing_offset shifts the receiver's window start into
 * the audio stream, causing some symbol boundaries to fall mid-window.
 * The receiver's boundary detector skips those windows; all other
 * windows decode correctly, giving bounded BER ≤ 1/PSK_SYM_WINDOWS.
 *
 * Build
 * ─────
 *   make test          # builds ./test
 *   make run-tests     # builds and runs ./test
 *   ./test             # returns 0 on full pass, 1 if any test fails
 */

#include "psk_common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─────────────────────────────────────────────────────────────────
 * Terminal colour codes (disabled automatically on non-Unix)
 * ───────────────────────────────────────────────────────────────── */
#ifdef __unix__
#  define C_GREEN  "\033[0;32m"
#  define C_RED    "\033[0;31m"
#  define C_YELLOW "\033[0;33m"
#  define C_CYAN   "\033[0;36m"
#  define C_RESET  "\033[0m"
#else
#  define C_GREEN  ""
#  define C_RED    ""
#  define C_YELLOW ""
#  define C_CYAN   ""
#  define C_RESET  ""
#endif

/* ─────────────────────────────────────────────────────────────────
 * Minimalist test framework
 * ───────────────────────────────────────────────────────────────── */
static int g_pass  = 0;
static int g_fail  = 0;
static int g_skip  = 0;   /* observational tests: counted but not failed */

static void t_assert(int cond, const char *msg, const char *file, int line)
{
    if (cond) {
        printf("  " C_GREEN "PASS" C_RESET "  %s\n", msg);
        g_pass++;
    } else {
        printf("  " C_RED "FAIL" C_RESET "  %s  [%s:%d]\n",
               msg, file, line);
        g_fail++;
    }
}

static void t_observe(const char *msg)
{
    /* Print an informational line (not pass/fail) */
    printf("  " C_CYAN "INFO" C_RESET "  %s\n", msg);
    g_skip++;
}

#define ASSERT(cond, msg) \
    t_assert(!!(cond), (msg), __FILE__, __LINE__)

#define ASSERT_EQ(a, b, msg) \
    t_assert((a) == (b), (msg), __FILE__, __LINE__)

#define ASSERT_NEAR(a, b, tol, msg) \
    t_assert(fabsf((float)(a) - (float)(b)) <= (float)(tol), \
             (msg), __FILE__, __LINE__)

static void section(const char *name)
{
    printf("\n" C_YELLOW "━━━  %s  ━━━" C_RESET "\n", name);
}

/* ═════════════════════════════════════════════════════════════════
 * 1. Utility functions
 * ═════════════════════════════════════════════════════════════════ */
static void test_utility(void)
{
    section("Utility: psk_is_power_of_2 / psk_ilog2");

    /* is_power_of_2 */
    ASSERT(!psk_is_power_of_2(-4),  "is_power_of_2(-4) = false");
    ASSERT(!psk_is_power_of_2(0),   "is_power_of_2(0)  = false");
    ASSERT(!psk_is_power_of_2(1),   "is_power_of_2(1)  = false  (< 2)");
    ASSERT( psk_is_power_of_2(2),   "is_power_of_2(2)  = true");
    ASSERT(!psk_is_power_of_2(3),   "is_power_of_2(3)  = false");
    ASSERT( psk_is_power_of_2(4),   "is_power_of_2(4)  = true");
    ASSERT(!psk_is_power_of_2(5),   "is_power_of_2(5)  = false");
    ASSERT(!psk_is_power_of_2(6),   "is_power_of_2(6)  = false");
    ASSERT( psk_is_power_of_2(8),   "is_power_of_2(8)  = true");
    ASSERT(!psk_is_power_of_2(12),  "is_power_of_2(12) = false");
    ASSERT( psk_is_power_of_2(16),  "is_power_of_2(16) = true");
    ASSERT( psk_is_power_of_2(32),  "is_power_of_2(32) = true");
    ASSERT(!psk_is_power_of_2(33),  "is_power_of_2(33) = false");
    ASSERT( psk_is_power_of_2(64),  "is_power_of_2(64) = true");
    ASSERT( psk_is_power_of_2(1024),"is_power_of_2(1024)= true");

    /* ilog2 */
    ASSERT_EQ(psk_ilog2(2),    1, "ilog2(2)  = 1");
    ASSERT_EQ(psk_ilog2(4),    2, "ilog2(4)  = 2");
    ASSERT_EQ(psk_ilog2(8),    3, "ilog2(8)  = 3");
    ASSERT_EQ(psk_ilog2(16),   4, "ilog2(16) = 4");
    ASSERT_EQ(psk_ilog2(32),   5, "ilog2(32) = 5");
    ASSERT_EQ(psk_ilog2(64),   6, "ilog2(64) = 6");
    ASSERT_EQ(psk_ilog2(4096), 12,"ilog2(4096)=12");
}

/* ═════════════════════════════════════════════════════════════════
 * 2. Gray coding
 * ═════════════════════════════════════════════════════════════════ */

/* Count the number of set bits in a non-negative integer. */
static int popcount(int n)
{
    int c = 0;
    while (n) { c += n & 1; n >>= 1; }
    return c;
}

static void test_gray_coding(void)
{
    section("Gray coding: encode / decode / adjacency");

    /* Known values for BPSK (M=2) and QPSK (M=4) */
    ASSERT_EQ(psk_gray_encode(0), 0, "gray_encode(0) = 0b000 = 0");
    ASSERT_EQ(psk_gray_encode(1), 1, "gray_encode(1) = 0b001 = 1");
    ASSERT_EQ(psk_gray_encode(2), 3, "gray_encode(2) = 0b011 = 3");
    ASSERT_EQ(psk_gray_encode(3), 2, "gray_encode(3) = 0b010 = 2");
    ASSERT_EQ(psk_gray_encode(4), 6, "gray_encode(4) = 0b110 = 6");
    ASSERT_EQ(psk_gray_encode(5), 7, "gray_encode(5) = 0b111 = 7");
    ASSERT_EQ(psk_gray_encode(6), 5, "gray_encode(6) = 0b101 = 5");
    ASSERT_EQ(psk_gray_encode(7), 4, "gray_encode(7) = 0b100 = 4");

    /* Roundtrip: decode(encode(n)) == n for n = 0..63 */
    {
        int ok = 1;
        for (int n = 0; n < 64; n++)
            if (psk_gray_decode(psk_gray_encode(n)) != n) { ok = 0; break; }
        ASSERT(ok, "gray_decode(gray_encode(n)) == n  for n in 0..63");
    }

    /* Roundtrip for larger range: n = 0..255 */
    {
        int ok = 1;
        for (int n = 0; n < 256; n++)
            if (psk_gray_decode(psk_gray_encode(n)) != n) { ok = 0; break; }
        ASSERT(ok, "gray_decode(gray_encode(n)) == n  for n in 0..255");
    }

    /* Adjacency: consecutive Gray codes differ by exactly 1 bit */
    {
        int ok = 1;
        for (int n = 0; n < 63; n++)
            if (popcount(psk_gray_encode(n) ^ psk_gray_encode(n + 1)) != 1)
                { ok = 0; break; }
        ASSERT(ok, "adjacent Gray codes differ by exactly 1 bit (n=0..63)");
    }

    /* Wrap adjacency: gray(2^k − 1) and gray(0) also differ by 1 bit
     * for all k (the code is cyclic). */
    for (int k = 1; k <= 5; k++) {
        int M    = 1 << k;
        int diff = popcount(psk_gray_encode(M - 1) ^ psk_gray_encode(0));
        char lbl[64];
        snprintf(lbl, sizeof(lbl),
                 "Gray(%d-1) XOR Gray(0): popcount = 1  (M=%d wrap)", M, M);
        ASSERT_EQ(diff, 1, lbl);
    }
}

/* ═════════════════════════════════════════════════════════════════
 * 3. Bit packing / symbol conversion
 * ═════════════════════════════════════════════════════════════════ */
static void test_bit_packing(void)
{
    section("Bit packing: bytes_to_symbols / symbols_to_bytes");

    /* ── BPSK: 1 bit per symbol, 1 byte → 8 symbols ─────────── */
    {
        uint8_t data[1] = {0xB4};  /* 1011 0100 */
        size_t  n;
        uint8_t *s = psk_bytes_to_symbols(data, 1, 1, &n);
        ASSERT_EQ((int)n, 8, "BPSK: 1 byte → 8 symbols");
        /* Bit 7 = 1 → raw 1 → gray(1)=1; Bit 6 = 0 → gray(0)=0 … */
        ASSERT_EQ(s[0], psk_gray_encode(1), "BPSK sym[0]: bit7=1");
        ASSERT_EQ(s[1], psk_gray_encode(0), "BPSK sym[1]: bit6=0");
        ASSERT_EQ(s[2], psk_gray_encode(1), "BPSK sym[2]: bit5=1");
        ASSERT_EQ(s[3], psk_gray_encode(1), "BPSK sym[3]: bit4=1");
        ASSERT_EQ(s[4], psk_gray_encode(0), "BPSK sym[4]: bit3=0");
        ASSERT_EQ(s[5], psk_gray_encode(1), "BPSK sym[5]: bit2=1");
        ASSERT_EQ(s[6], psk_gray_encode(0), "BPSK sym[6]: bit1=0");
        ASSERT_EQ(s[7], psk_gray_encode(0), "BPSK sym[7]: bit0=0");
        free(s);
    }

    /* ── QPSK: 2 bits per symbol, 1 byte → 4 symbols ──────────── */
    {
        uint8_t data[1] = {0xCA};  /* 1100 1010 */
        size_t  n;
        uint8_t *s = psk_bytes_to_symbols(data, 1, 2, &n);
        ASSERT_EQ((int)n, 4, "QPSK: 1 byte → 4 symbols");
        /* 11→3, 00→0, 10→2, 10→2 */
        ASSERT_EQ(s[0], psk_gray_encode(3), "QPSK sym[0]: 0b11 = 3");
        ASSERT_EQ(s[1], psk_gray_encode(0), "QPSK sym[1]: 0b00 = 0");
        ASSERT_EQ(s[2], psk_gray_encode(2), "QPSK sym[2]: 0b10 = 2");
        ASSERT_EQ(s[3], psk_gray_encode(2), "QPSK sym[3]: 0b10 = 2");
        free(s);
    }

    /* ── 8-PSK: 3 bits per symbol, 3 bytes → 8 symbols ─────────── */
    {
        uint8_t data[3] = {0xFF, 0x00, 0xAA};
        size_t  n;
        uint8_t *s = psk_bytes_to_symbols(data, 3, 3, &n);
        ASSERT_EQ((int)n, 8, "8-PSK: 3 bytes → 8 symbols");
        /* 0xFF 0x00 = 1111 1111 0000 0000 → groups of 3:
         * 111 111 110 000 000 0 → then 0xAA = 10 101 010 → 10 + (rest pad) */
        ASSERT_EQ(s[0], psk_gray_encode(7), "8-PSK sym[0] = gray(7)");
        ASSERT_EQ(s[1], psk_gray_encode(7), "8-PSK sym[1] = gray(7)");
        ASSERT_EQ(s[2], psk_gray_encode(6), "8-PSK sym[2] = gray(6)");
        ASSERT_EQ(s[3], psk_gray_encode(0), "8-PSK sym[3] = gray(0)");
        free(s);
    }

    /* ── Zero-length input ─────────────────────────────────────── */
    {
        uint8_t dummy = 0;
        size_t  n     = 99;
        uint8_t *s = psk_bytes_to_symbols(&dummy, 0, 2, &n);
        ASSERT_EQ((int)n, 0, "zero-length input → 0 symbols");
        free(s);
    }

    /* ── Roundtrip: bytes → symbols → bytes, all bps 1..5 ──────── */
    {
        const char *msg = "Hello, PSK World!";
        size_t      msg_len = strlen(msg);

        for (int bps = 1; bps <= 5; bps++) {
            size_t   n_syms  = 0;
            uint8_t *syms = psk_bytes_to_symbols((const uint8_t *)msg,
                                                   msg_len, bps, &n_syms);
            size_t   n_bytes = 0;
            uint8_t *recovered = psk_symbols_to_bytes(syms, n_syms,
                                                       bps, &n_bytes);
            int ok = (n_bytes >= msg_len) &&
                     (memcmp(msg, recovered, msg_len) == 0);
            char lbl[80];
            snprintf(lbl, sizeof(lbl),
                     "Roundtrip bps=%d (M=%d): '%s'", bps, 1 << bps, msg);
            ASSERT(ok, lbl);
            free(syms);
            free(recovered);
        }
    }

    /* ── Symbol values are always in range [0, M-1] ────────────── */
    {
        uint8_t data[32];
        for (int i = 0; i < 32; i++) data[i] = (uint8_t)(i * 7 + 13);

        for (int M = 2; M <= 32; M *= 2) {
            int  bps = psk_ilog2(M);
            size_t n = 0;
            uint8_t *s = psk_bytes_to_symbols(data, 32, bps, &n);
            int ok = 1;
            for (size_t i = 0; i < n; i++)
                if ((int)s[i] >= M) { ok = 0; break; }
            char lbl[64];
            snprintf(lbl, sizeof(lbl),
                     "M=%d: all symbols in [0, M-1]", M);
            ASSERT(ok, lbl);
            free(s);
        }
    }
}

/* ═════════════════════════════════════════════════════════════════
 * 4. Transmitter buffer generation
 * ═════════════════════════════════════════════════════════════════ */
static void test_tx_buffer(void)
{
    section("TX buffer: phase, amplitude, inter-buffer continuity");

    float buf[PSK_FRAMES_PER_BUF_TX];

    /* ── First sample equals sin(symbol × 2π/M) ─────────────── */
    for (int M = 2; M <= 32; M *= 2) {
        int ok = 1;
        for (int sym = 0; sym < M; sym++) {
            psk_generate_tx_buffer(buf, sym, M);
            float expected = sinf((float)sym * 2.0f * (float)M_PI / (float)M);
            if (fabsf(buf[0] - expected) > 1e-5f) { ok = 0; break; }
        }
        char lbl[64];
        snprintf(lbl, sizeof(lbl),
                 "M=%d: buf[0] = sin(symbol × 2π/M) for all symbols", M);
        ASSERT(ok, lbl);
    }

    /* ── Maximum amplitude = 1.0 (no clipping) ──────────────── */
    {
        psk_generate_tx_buffer(buf, 0, 4);
        float max = 0.0f;
        for (int t = 0; t < PSK_FRAMES_PER_BUF_TX; t++)
            if (fabsf(buf[t]) > max) max = fabsf(buf[t]);
        ASSERT_NEAR(max, 1.0f, 1e-5f,
                    "QPSK sym=0: max amplitude = 1.0 (full-scale sine)");
    }

    /* ── Buffer is periodic with period N/CARRIER_CYCLES samples ─ */
    /* The carrier completes exactly PSK_CARRIER_CYCLES full cycles
     * in PSK_FRAMES_PER_BUF_TX samples, so the pattern repeats.   */
    {
        psk_generate_tx_buffer(buf, 1, 4);
        int period = PSK_FRAMES_PER_BUF_TX / PSK_CARRIER_CYCLES;  /* = 64 */
        float max_diff = 0.0f;
        for (int t = 0; t < PSK_FRAMES_PER_BUF_TX - period; t++) {
            float d = fabsf(buf[t] - buf[t + period]);
            if (d > max_diff) max_diff = d;
        }
        ASSERT_NEAR(max_diff, 0.0f, 1e-5f,
                    "QPSK sym=1: buffer repeats with period N/CARRIER_CYCLES");
    }

    /* ── Same-symbol consecutive buffers are bit-for-bit identical ─
     * (carrier resets cleanly at every buffer boundary because it
     *  completes exactly CARRIER_CYCLES integer cycles per buffer)  */
    {
        float buf_a[PSK_FRAMES_PER_BUF_TX];
        float buf_b[PSK_FRAMES_PER_BUF_TX];
        psk_generate_tx_buffer(buf_a, 2, 4);
        psk_generate_tx_buffer(buf_b, 2, 4);
        ASSERT(memcmp(buf_a, buf_b, sizeof(buf_a)) == 0,
               "same-symbol tx buffers are identical (clean phase reset)");
    }

    /* ── Different symbols produce different buffers ─────────── */
    {
        float buf_0[PSK_FRAMES_PER_BUF_TX];
        float buf_1[PSK_FRAMES_PER_BUF_TX];
        psk_generate_tx_buffer(buf_0, 0, 4);
        psk_generate_tx_buffer(buf_1, 1, 4);
        ASSERT(memcmp(buf_0, buf_1, sizeof(buf_0)) != 0,
               "QPSK sym=0 and sym=1 produce distinct buffers");
    }

    /* ── BPSK: sym=0 and sym=1 are phase-inverted ────────────── */
    {
        float buf_0[PSK_FRAMES_PER_BUF_TX];
        float buf_1[PSK_FRAMES_PER_BUF_TX];
        psk_generate_tx_buffer(buf_0, 0, 2);
        psk_generate_tx_buffer(buf_1, 1, 2);
        float max_diff = 0.0f;
        for (int t = 0; t < PSK_FRAMES_PER_BUF_TX; t++) {
            float d = fabsf(buf_0[t] + buf_1[t]);   /* should sum to ~0 */
            if (d > max_diff) max_diff = d;
        }
        ASSERT_NEAR(max_diff, 0.0f, 1e-5f,
                    "BPSK: sym=0 and sym=1 are exact phase inversions");
    }
}

/* ═════════════════════════════════════════════════════════════════
 * 5. DFT at carrier bin
 * ═════════════════════════════════════════════════════════════════ */

/* Generate a PSK_FRAMES_PER_BUF_RX-sample window of
 *   sin(2π × CARRIER_BIN / N × t + phi)                           */
static void gen_rx_window(float *out, float phi)
{
    const int   N  = PSK_FRAMES_PER_BUF_RX;
    const float tw = 2.0f * (float)M_PI * (float)PSK_CARRIER_BIN / (float)N;
    for (int t = 0; t < N; t++)
        out[t] = sinf(tw * (float)t + phi);
}

static void test_dft(void)
{
    section("DFT carrier bin: magnitude, phase, rejection, consistency");

    float win[PSK_FRAMES_PER_BUF_RX];
    const float half_N = (float)PSK_FRAMES_PER_BUF_RX / 2.0f;   /* = 2048 */

    /* ── |DFT[CARRIER_BIN]| ≈ N/2 for all M and all symbols ──── */
    for (int M = 2; M <= 16; M *= 2) {
        int ok = 1;
        for (int sym = 0; sym < M; sym++) {
            float phi = (float)sym * 2.0f * (float)M_PI / (float)M;
            gen_rx_window(win, phi);
            if (fabsf(cabsf(psk_dft_carrier(win)) - half_N) > 1.0f)
                { ok = 0; break; }
        }
        char lbl[80];
        snprintf(lbl, sizeof(lbl),
                 "M=%d: |DFT[%d]| ≈ N/2 = %.0f for all symbols",
                 M, PSK_CARRIER_BIN, half_N);
        ASSERT(ok, lbl);
    }

    /* ── angle(DFT[CARRIER_BIN]) = phi − π/2 ─────────────────── */
    /* For sin(ω₀t + φ): X = −i(N/2)e^{iφ}  ⟹  angle = φ − π/2  */
    {
        int ok = 1;
        for (int M = 2; M <= 8; M *= 2) {
            for (int sym = 0; sym < M; sym++) {
                float phi = (float)sym * 2.0f * (float)M_PI / (float)M;
                gen_rx_window(win, phi);
                float rx_phase = atan2f(cimagf(psk_dft_carrier(win)),
                                         crealf(psk_dft_carrier(win)));
                float expected  = phi - (float)M_PI / 2.0f;
                float diff = atan2f(sinf(rx_phase - expected),
                                     cosf(rx_phase - expected));
                if (fabsf(diff) > 0.001f) { ok = 0; }
            }
        }
        ASSERT(ok, "angle(DFT[carrier]) = transmitted_phase − π/2");
    }

    /* ── Off-frequency tone gives small magnitude ──────────────── */
    {
        /* Bin 64 + 7 = bin 71 */
        const int   N  = PSK_FRAMES_PER_BUF_RX;
        const float tw = 2.0f * (float)M_PI
                       * (float)(PSK_CARRIER_BIN + 7) / (float)N;
        for (int t = 0; t < N; t++)
            win[t] = sinf(tw * (float)t);
        ASSERT(cabsf(psk_dft_carrier(win)) < 2.0f,
               "off-by-7-bins tone: |DFT[carrier]| < 2 (high rejection)");
    }

    /* ── DC signal gives zero at the carrier bin ──────────────── */
    {
        for (int t = 0; t < PSK_FRAMES_PER_BUF_RX; t++) win[t] = 1.0f;
        ASSERT(cabsf(psk_dft_carrier(win)) < 1e-3f,
               "DC signal: |DFT[carrier]| ≈ 0");
    }

    /* ── Silence gives zero ──────────────────────────────────── */
    {
        memset(win, 0, sizeof(float) * PSK_FRAMES_PER_BUF_RX);
        ASSERT(cabsf(psk_dft_carrier(win)) < 1e-6f,
               "silence: |DFT[carrier]| = 0");
    }

    /* ── Consecutive same-symbol windows give identical DFT phase ─
     * (core invariant: carrier completes integer cycles per window) */
    {
        const int   N  = PSK_FRAMES_PER_BUF_RX;
        const float phi = 1.2f;
        const float tw  = 2.0f * (float)M_PI * PSK_CARRIER_BIN / (float)N;

        /* Window starting at sample 0 */
        for (int t = 0; t < N; t++)
            win[t] = sinf(tw * (float)t + phi);
        float phase0 = atan2f(cimagf(psk_dft_carrier(win)),
                               crealf(psk_dft_carrier(win)));

        /* Window starting at sample N (next window) */
        for (int t = 0; t < N; t++)
            win[t] = sinf(tw * (float)(t + N) + phi);
        float phase1 = atan2f(cimagf(psk_dft_carrier(win)),
                               crealf(psk_dft_carrier(win)));

        float diff = atan2f(sinf(phase1 - phase0), cosf(phase1 - phase0));
        ASSERT_NEAR(diff, 0.0f, 0.001f,
            "consecutive same-symbol windows: DFT phase identical");
    }

    /* ── Linearity: sum of two tones gives sum of DFTs ─────────── */
    {
        float win_a[PSK_FRAMES_PER_BUF_RX];
        float win_b[PSK_FRAMES_PER_BUF_RX];
        float win_sum[PSK_FRAMES_PER_BUF_RX];
        float phi_a = 0.3f;
        float phi_b = 1.1f;
        gen_rx_window(win_a, phi_a);
        gen_rx_window(win_b, phi_b);
        for (int t = 0; t < PSK_FRAMES_PER_BUF_RX; t++)
            win_sum[t] = win_a[t] + win_b[t];
        float complex X_a   = psk_dft_carrier(win_a);
        float complex X_b   = psk_dft_carrier(win_b);
        float complex X_sum = psk_dft_carrier(win_sum);
        float err = cabsf(X_sum - (X_a + X_b));
        ASSERT(err < 1e-2f, "DFT linearity: DFT(a+b) = DFT(a) + DFT(b)");
    }
}

/* ═════════════════════════════════════════════════════════════════
 * 6. Phase-to-symbol decoding
 * ═════════════════════════════════════════════════════════════════ */
static void test_phase_decode(void)
{
    section("Phase decode: ideal phases and noise margin");

    /* ── Ideal phase → correct symbol for all M and all symbols ── */
    for (int M = 2; M <= 32; M *= 2) {
        int ok = 1;
        for (int sym = 0; sym < M; sym++) {
            float phi  = (float)sym * 2.0f * (float)M_PI / (float)M;
            int decoded = psk_phase_to_symbol(phi, M);
            if (decoded != sym) { ok = 0; break; }
        }
        char lbl[64];
        snprintf(lbl, sizeof(lbl),
                 "M=%d: ideal phases decode to correct symbols", M);
        ASSERT(ok, lbl);
    }

    /* ── Explicit QPSK check at 0°, 90°, 180°, 270° ──────────── */
    {
        float two_pi = 2.0f * (float)M_PI;
        ASSERT_EQ(psk_phase_to_symbol(0.0f,           4), 0, "QPSK 0°   → sym 0");
        ASSERT_EQ(psk_phase_to_symbol(two_pi / 4.0f,  4), 1, "QPSK 90°  → sym 1");
        ASSERT_EQ(psk_phase_to_symbol(two_pi / 2.0f,  4), 2, "QPSK 180° → sym 2");
        ASSERT_EQ(psk_phase_to_symbol(3.0f*two_pi/4.0f,4),3, "QPSK 270° → sym 3");
    }

    /* ── ±25% noise stays within the correct decision region ──── */
    for (int M = 2; M <= 16; M *= 2) {
        float spacing = 2.0f * (float)M_PI / (float)M;
        float noise   = spacing / 4.0f;  /* 25% of decision boundary */
        int ok = 1;
        for (int sym = 0; sym < M; sym++) {
            float ideal   = (float)sym * spacing;
            float noisy_p = fmodf(ideal + noise + 4.0f*(float)M_PI,
                                   2.0f * (float)M_PI);
            float noisy_n = fmodf(ideal - noise + 4.0f*(float)M_PI,
                                   2.0f * (float)M_PI);
            if (psk_phase_to_symbol(noisy_p, M) != sym ||
                psk_phase_to_symbol(noisy_n, M) != sym)
                { ok = 0; break; }
        }
        char lbl[80];
        snprintf(lbl, sizeof(lbl),
                 "M=%-2d: ±25%% noise (±%.1f°) stays in correct region",
                 M, noise * 180.0f / (float)M_PI);
        ASSERT(ok, lbl);
    }

    /* ── Phase wrap: near-360° correctly maps to symbol 0 ──────── */
    {
        float near_360 = 2.0f * (float)M_PI - 0.001f;
        ASSERT_EQ(psk_phase_to_symbol(near_360, 4), 0,
                  "QPSK: phase near 360° wraps to symbol 0");
    }
}

/* ═════════════════════════════════════════════════════════════════
 * 7. Timing recovery
 * ═════════════════════════════════════════════════════════════════ */
static void test_timing(void)
{
    section("Timing recovery: boundary detection and window counter");

    float win[PSK_FRAMES_PER_BUF_RX];

#define GEN(phi) gen_rx_window(win, (phi))

    /* ── No boundary on first window (no previous reference) ──── */
    {
        PskRxState s; psk_rx_state_init(&s);
        GEN(0.0f);
        psk_rx_process_window(&s, win, 4);
        ASSERT(!s.is_boundary, "first window: is_boundary = 0 (no prev)");
    }

    /* ── Same-phase consecutive windows: no boundary ──────────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        GEN(0.5f); psk_rx_process_window(&s, win, 4);
        GEN(0.5f); psk_rx_process_window(&s, win, 4);
        ASSERT(!s.is_boundary, "same-phase windows: is_boundary = 0");
        ASSERT(!s.timing_locked,"same-phase windows: timing_locked = 0");
    }

    /* ── QPSK 90° jump triggers boundary ──────────────────────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        GEN(0.0f);                        psk_rx_process_window(&s, win, 4);
        GEN((float)M_PI / 2.0f);          psk_rx_process_window(&s, win, 4);
        ASSERT( s.is_boundary,   "QPSK 90° jump: is_boundary = 1");
        ASSERT( s.timing_locked, "QPSK 90° jump: timing_locked = 1");
        ASSERT_EQ(s.sym_window, 0, "QPSK 90° jump: sym_window = 0");
    }

    /* ── BPSK 180° jump triggers boundary ─────────────────────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        GEN(0.0f);                        psk_rx_process_window(&s, win, 4);
        GEN((float)M_PI);                  psk_rx_process_window(&s, win, 4);
        ASSERT(s.is_boundary, "BPSK 180° jump: is_boundary = 1");
    }

    /* ── Window counter increments correctly after boundary ────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        GEN(0.0f); psk_rx_process_window(&s, win, 4);          /* seed   */
        GEN(1.0f); psk_rx_process_window(&s, win, 4);          /* jump   */
        ASSERT_EQ(s.sym_window, 0, "after boundary: sym_window = 0");

        GEN(1.0f); psk_rx_process_window(&s, win, 4);
        ASSERT_EQ(s.sym_window, 1, "next same-phase: sym_window = 1");

        GEN(1.0f); psk_rx_process_window(&s, win, 4);
        ASSERT_EQ(s.sym_window, 2, "next same-phase: sym_window = 2");

        GEN(1.0f); psk_rx_process_window(&s, win, 4);
        ASSERT_EQ(s.sym_window, 3, "next same-phase: sym_window = 3");

        GEN(1.0f); psk_rx_process_window(&s, win, 4);
        ASSERT_EQ(s.sym_window, 0, "next same-phase: sym_window wraps to 0");
    }

    /* ── 5° phase change: below threshold for all M ≥ 4 ────────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        float small = 5.0f * (float)M_PI / 180.0f;
        GEN(0.0f);    psk_rx_process_window(&s, win, 4);
        GEN(small);   psk_rx_process_window(&s, win, 4);
        ASSERT(!s.is_boundary,
               "5° phase change: below QPSK threshold (π/4 = 45°)");
    }

    /* ── 8-PSK threshold is narrower: 45° triggers boundary ────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        GEN(0.0f);
        psk_rx_process_window(&s, win, 8);
        GEN((float)M_PI / 4.0f);   /* 45° = 1 symbol spacing in 8-PSK  */
        psk_rx_process_window(&s, win, 8);
        ASSERT(s.is_boundary,
               "8-PSK: 45° jump > π/8 threshold → boundary");
    }

    /* ── Threshold scales correctly with M ─────────────────────── */
    /* For M-PSK the boundary threshold is π/M.  A phase step of
     * π/M + 5° should always trigger and π/M − 5° should not.     */
    for (int M = 4; M <= 16; M *= 2) {
        float thresh = (float)M_PI / (float)M;
        float above  = thresh + 5.0f * (float)M_PI / 180.0f;
        float below  = thresh - 5.0f * (float)M_PI / 180.0f;
        char lbl[80];

        {
            PskRxState s; psk_rx_state_init(&s);
            GEN(0.0f); psk_rx_process_window(&s, win, M);
            GEN(above); psk_rx_process_window(&s, win, M);
            snprintf(lbl, sizeof(lbl),
                     "M=%d: (π/M+5°) = %.1f° triggers boundary",
                     M, above * 180.0f / (float)M_PI);
            ASSERT(s.is_boundary, lbl);
        }
        {
            PskRxState s; psk_rx_state_init(&s);
            GEN(0.0f); psk_rx_process_window(&s, win, M);
            GEN(below); psk_rx_process_window(&s, win, M);
            snprintf(lbl, sizeof(lbl),
                     "M=%d: (π/M-5°) = %.1f° does not trigger boundary",
                     M, below * 180.0f / (float)M_PI);
            ASSERT(!s.is_boundary, lbl);
        }
    }

    /* ── No-carrier window does not trigger boundary ───────────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        GEN(0.5f); psk_rx_process_window(&s, win, 4);

        /* Generate a near-silent window */
        for (int t = 0; t < PSK_FRAMES_PER_BUF_RX; t++)
            win[t] = 0.0f;
        psk_rx_process_window(&s, win, 4);
        ASSERT(!s.is_boundary,
               "silent window after carrier: is_boundary = 0");
    }

#undef GEN
}

/* ═════════════════════════════════════════════════════════════════
 * 8. PLL convergence
 * ═════════════════════════════════════════════════════════════════ */
static void test_pll(void)
{
    section("PLL: drives phase error to zero");

    float win[PSK_FRAMES_PER_BUF_RX];
    const int M = 4;   /* QPSK */

    /* The preamble carrier has transmitted phase φ = 0°.
     * DFT phase of sin(ω₀t + 0°) = 0° − π/2 = −π/2.
     * After PSK_CAL_WINDOWS windows, phase_offset ≈ −π/2.           */
    const float preamble_dft_phase = -(float)M_PI / 2.0f;

    /* ── Verify calibration completes ──────────────────────────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        for (int i = 0; i < PSK_CAL_WINDOWS; i++) {
            gen_rx_window(win, 0.0f);   /* preamble: tx phase = 0° */
            psk_rx_process_window(&s, win, M);
        }
        ASSERT_EQ(s.rx_state, 2,
                  "PLL test: calibration completes after CAL_WINDOWS frames");
        ASSERT_NEAR(s.phase_offset, preamble_dft_phase, 0.01f,
                    "PLL test: phase_offset ≈ −π/2 after preamble calibration");
    }

    /* ── PLL reduces a known phase error over time ──────────────
     *
     * We introduce a deliberate +20° error after calibration and
     * run 100 preamble windows through the state machine.
     *
     * Time constant τ = 1 / PLL_ALPHA ≈ 40 frames.
     * After 100 frames: residual ≈ 20° × (1 − 0.025)^100 ≈ 1.6°.
     */
    {
        const float error_deg = 20.0f;
        PskRxState s; psk_rx_state_init(&s);

        /* Calibrate */
        for (int i = 0; i < PSK_CAL_WINDOWS; i++) {
            gen_rx_window(win, 0.0f);
            psk_rx_process_window(&s, win, M);
        }

        /* Introduce deliberate phase error */
        s.phase_offset -= error_deg * (float)M_PI / 180.0f;

        /* Run PLL convergence */
        for (int i = 0; i < 100; i++) {
            gen_rx_window(win, 0.0f);
            psk_rx_process_window(&s, win, M);
        }

        ASSERT_NEAR(s.pll_error_deg, 0.0f, 5.0f,
                    "PLL: error < 5° after 100 frames from +20° initial error");
    }

    /* ── PLL also handles negative error ───────────────────────── */
    {
        const float error_deg = -15.0f;
        PskRxState s; psk_rx_state_init(&s);
        for (int i = 0; i < PSK_CAL_WINDOWS; i++) {
            gen_rx_window(win, 0.0f);
            psk_rx_process_window(&s, win, M);
        }
        s.phase_offset -= error_deg * (float)M_PI / 180.0f;
        for (int i = 0; i < 100; i++) {
            gen_rx_window(win, 0.0f);
            psk_rx_process_window(&s, win, M);
        }
        ASSERT_NEAR(s.pll_error_deg, 0.0f, 5.0f,
                    "PLL: error < 5° after 100 frames from −15° initial error");
    }

    /* ── PLL does not blow up on a large initial error ──────────── */
    {
        PskRxState s; psk_rx_state_init(&s);
        for (int i = 0; i < PSK_CAL_WINDOWS; i++) {
            gen_rx_window(win, 0.0f);
            psk_rx_process_window(&s, win, M);
        }
        s.phase_offset -= 40.0f * (float)M_PI / 180.0f;  /* 40° error */
        for (int i = 0; i < 200; i++) {
            gen_rx_window(win, 0.0f);
            psk_rx_process_window(&s, win, M);
        }
        ASSERT_NEAR(s.pll_error_deg, 0.0f, 8.0f,
                    "PLL: converges from 40° error after 200 frames");
    }
}

/* ═════════════════════════════════════════════════════════════════
 * 9. Software-channel loopback
 *
 * Generates the complete transmitter audio stream in memory and feeds
 * it to the receiver pipeline window-by-window.  No PortAudio or
 * SDL2 is required.
 *
 * timing_offset > 0 shifts the receiver's window start into the
 * audio stream, simulating the fact that in a real deployment the
 * transmitter and receiver do not share a sample clock.
 * ═════════════════════════════════════════════════════════════════ */

typedef struct {
    int    bit_errors;
    int    total_bits;
    int    sym_errors;
    int    total_syms;
    int    boundary_windows;
    int    total_windows;
} LoopbackResult;

/*
 * run_loopback — full transmit→receive pipeline in software.
 *
 *   data / data_len  : payload bytes
 *   M                : PSK order
 *   timing_offset    : samples to skip at the start of the audio
 *                      stream before the receiver begins reading.
 *                      0 = perfectly aligned (symbol boundaries fall
 *                          exactly at window edges — best case).
 *                      >0 = some boundaries will fall mid-window.
 */
static LoopbackResult run_loopback(const uint8_t *data, size_t data_len,
                                    int M, int timing_offset)
{
    LoopbackResult R = {0, 0, 0, 0, 0, 0};
    int bps = psk_ilog2(M);

    /* ── Build symbol array: preamble (symbol 0) + data ──────── */
    size_t   n_data;
    uint8_t *data_syms = psk_bytes_to_symbols(data, data_len, bps, &n_data);
    if (!data_syms) return R;

    size_t   n_total  = (size_t)PSK_PREAMBLE_SYMBOLS + n_data;
    uint8_t *all_syms = calloc(n_total, 1);   /* preamble = 0 */
    if (!all_syms) { free(data_syms); return R; }
    memcpy(all_syms + PSK_PREAMBLE_SYMBOLS, data_syms, n_data);
    free(data_syms);

    /* ── Generate full audio stream in memory ─────────────────── */
    size_t audio_len = n_total * (size_t)PSK_SAMPLES_PER_SYMBOL;
    float *audio     = malloc(audio_len * sizeof(float));
    if (!audio) { free(all_syms); return R; }

    for (size_t si = 0; si < n_total; si++) {
        float *dst = audio + si * (size_t)PSK_SAMPLES_PER_SYMBOL;
        for (int bi = 0; bi < PSK_BUFFERS_PER_SYMBOL; bi++)
            psk_generate_tx_buffer(dst + bi * PSK_FRAMES_PER_BUF_TX,
                                   all_syms[si], M);
    }

    /* ── Feed to receiver in 4096-sample windows ──────────────── */
    PskRxState s;
    psk_rx_state_init(&s);

    size_t audio_pos = (size_t)timing_offset;
    size_t win_num   = 0;

    while (audio_pos + (size_t)PSK_FRAMES_PER_BUF_RX <= audio_len) {
        R.total_windows++;

        int result = psk_rx_process_window(&s, audio + audio_pos, M);

        if (s.is_boundary) R.boundary_windows++;

        if (result >= 0) {
            /* Which transmitted symbol does the centre of this window
             * belong to?  Use the midpoint for robustness around
             * symbol boundary crossings.                             */
            size_t mid_sample = audio_pos + (size_t)(PSK_FRAMES_PER_BUF_RX / 2);
            size_t tx_sym_idx = mid_sample / (size_t)PSK_SAMPLES_PER_SYMBOL;

            /* Only count windows that fall within a data symbol
             * (not the preamble).                                   */
            if (tx_sym_idx >= (size_t)PSK_PREAMBLE_SYMBOLS &&
                tx_sym_idx < n_total) {

                int expected_sym  = all_syms[tx_sym_idx];
                int expected_bits = psk_gray_decode(expected_sym);
                int decoded_bits  = psk_gray_decode(result);
                int xor_bits      = expected_bits ^ decoded_bits;

                for (int b = 0; b < bps; b++) {
                    R.total_bits++;
                    if ((xor_bits >> b) & 1) R.bit_errors++;
                }
                R.total_syms++;
                if (expected_sym != result) R.sym_errors++;
            }
        }

        audio_pos += (size_t)PSK_FRAMES_PER_BUF_RX;
        win_num++;
    }

    free(audio);
    free(all_syms);
    return R;
}

/* Assert zero BER for an aligned loopback test. */
static void assert_loopback_zero_ber(const char *label,
                                      int M,
                                      const uint8_t *data,
                                      size_t         data_len,
                                      int            timing_offset)
{
    LoopbackResult R = run_loopback(data, data_len, M, timing_offset);

    float ber = (R.total_bits > 0)
              ? (float)R.bit_errors / (float)R.total_bits * 100.0f
              : -1.0f;

    char lbl[256];
    snprintf(lbl, sizeof(lbl),
             "%-24s M=%-2d off=%-4d  %d/%d bits err  BER=%.1f%%",
             label, M, timing_offset,
             R.bit_errors, R.total_bits,
             (ber < 0) ? 0.0f : ber);

    ASSERT(R.bit_errors == 0 && R.total_bits > 0, lbl);
}

/* Print an observational loopback result (non-asserting). */
static void observe_loopback(const char *label,
                              int M,
                              const uint8_t *data,
                              size_t         data_len,
                              int            timing_offset)
{
    LoopbackResult R = run_loopback(data, data_len, M, timing_offset);

    float ber = (R.total_bits > 0)
              ? (float)R.bit_errors / (float)R.total_bits * 100.0f
              : -1.0f;
    float bnd_pct = (R.total_windows > 0)
              ? (float)R.boundary_windows / (float)R.total_windows * 100.0f
              : 0.0f;

    char msg[256];
    snprintf(msg, sizeof(msg),
             "%-22s M=%-2d off=%-5d  BER=%.1f%% (%d/%d bits)  "
             "boundaries=%.0f%%",
             label, M, timing_offset, (ber < 0) ? 0.0f : ber,
             R.bit_errors, R.total_bits, bnd_pct);
    t_observe(msg);
}

static void test_loopback(void)
{
    section("Software-channel loopback");

    /* Common test payloads */
    const char *hello = "Hello, World!";
    const char *engn  = "ENGN1580 Final Project";

    uint8_t all_zeros[32];  memset(all_zeros, 0x00, 32);
    uint8_t all_ones[32];   memset(all_ones,  0xFF, 32);

    uint8_t alternating[32];
    for (int i = 0; i < 32; i++) alternating[i] = (uint8_t)((i & 1) ? 0xAA : 0x55);

    uint8_t ramp[32];
    for (int i = 0; i < 32; i++) ramp[i] = (uint8_t)(i * 8);

    uint8_t counter[64];
    for (int i = 0; i < 64; i++) counter[i] = (uint8_t)i;

    /* ── Aligned loopback (offset = 0): expect exactly 0 BER ──── */
    printf("\n  [Aligned — symbol boundaries fall at window edges]\n");

    for (int M = 2; M <= 16; M *= 2) {
        assert_loopback_zero_ber("hello",       M, (const uint8_t *)hello, strlen(hello), 0);
        assert_loopback_zero_ber("all-zeros",   M, all_zeros,   32, 0);
        assert_loopback_zero_ber("all-ones",    M, all_ones,    32, 0);
        assert_loopback_zero_ber("alternating", M, alternating, 32, 0);
        assert_loopback_zero_ber("ramp",        M, ramp,        32, 0);
    }
    assert_loopback_zero_ber("long-text", 4, (const uint8_t *)engn,
                              strlen(engn), 0);
    assert_loopback_zero_ber("counter-64B", 8, counter, 64, 0);

    /* ── End-to-end bytes roundtrip (QPSK) ─────────────────────── */
    /* Encode → transmit → receive → decode → compare to original   */
    printf("\n  [End-to-end bytes roundtrip via run_loopback]\n");
    {
        const uint8_t *msg     = (const uint8_t *)hello;
        size_t         msg_len = strlen(hello);
        int            M       = 4;
        int            bps     = psk_ilog2(M);

        size_t   n_syms   = 0;
        uint8_t *tx_syms  = psk_bytes_to_symbols(msg, msg_len, bps, &n_syms);

        /* Run loopback; collect decoded symbols */
        size_t   n_total  = (size_t)PSK_PREAMBLE_SYMBOLS + n_syms;
        uint8_t *all_syms = calloc(n_total, 1);
        memcpy(all_syms + PSK_PREAMBLE_SYMBOLS, tx_syms, n_syms);
        free(tx_syms);

        /* Generate audio */
        size_t  audio_len = n_total * (size_t)PSK_SAMPLES_PER_SYMBOL;
        float  *audio     = malloc(audio_len * sizeof(float));
        for (size_t si = 0; si < n_total; si++) {
            float *dst = audio + si * (size_t)PSK_SAMPLES_PER_SYMBOL;
            for (int bi = 0; bi < PSK_BUFFERS_PER_SYMBOL; bi++)
                psk_generate_tx_buffer(dst + bi * PSK_FRAMES_PER_BUF_TX,
                                       all_syms[si], M);
        }

        /* Decode with receiver, collecting one sample per data symbol
         * (take the first non-boundary decode for each symbol slot). */
        uint8_t *rx_syms = calloc(n_syms, 1);
        int      got[256] = {0};   /* 1 once we have a decode for sym i */

        PskRxState s; psk_rx_state_init(&s);
        size_t audio_pos = 0;

        while (audio_pos + (size_t)PSK_FRAMES_PER_BUF_RX <= audio_len) {
            int result = psk_rx_process_window(&s, audio + audio_pos, M);
            if (result >= 0) {
                size_t mid   = audio_pos + (size_t)(PSK_FRAMES_PER_BUF_RX / 2);
                size_t tx_si = mid / (size_t)PSK_SAMPLES_PER_SYMBOL;
                if (tx_si >= (size_t)PSK_PREAMBLE_SYMBOLS &&
                    tx_si < n_total) {
                    size_t data_si = tx_si - (size_t)PSK_PREAMBLE_SYMBOLS;
                    if (!got[data_si]) {
                        rx_syms[data_si] = (uint8_t)result;
                        got[data_si] = 1;
                    }
                }
            }
            audio_pos += (size_t)PSK_FRAMES_PER_BUF_RX;
        }

        /* Decode symbols back to bytes */
        size_t   n_bytes  = 0;
        uint8_t *recovered = psk_symbols_to_bytes(rx_syms, n_syms, bps, &n_bytes);

        int ok = (n_bytes >= msg_len) &&
                 (memcmp(msg, recovered, msg_len) == 0);
        ASSERT(ok, "QPSK end-to-end: 'Hello, World!' recovered byte-for-byte");

        free(audio); free(all_syms); free(rx_syms); free(recovered);
    }

    /* ── Timing-offset loopback: simulation of real-world misalign ─
     *
     * A non-zero offset causes some symbol boundaries to fall inside
     * a receiver window instead of at a window edge.  That window
     * sees a phase mixture and is detected as a boundary (skipped).
     * All other windows still decode correctly → BER bounded by the
     * fraction of boundary windows ≤ 1/PSK_SYM_WINDOWS = 25%.
     *
     * These are observational: printed for inspection, not asserted.
     */
    printf("\n  [Offset — simulated real-world timing misalignment]\n");
    printf("  (INFO lines are observational; BER ≤ 25%% is expected)\n");

    int offsets[] = {512, 1024, 2048, 3072};
    for (int oi = 0; oi < 4; oi++) {
        int off = offsets[oi];
        observe_loopback("hello-QPSK",   4, (const uint8_t *)hello,
                          strlen(hello), off);
        observe_loopback("ramp-8PSK",    8, ramp, 32, off);
        observe_loopback("counter-BPSK", 2, counter, 32, off);
    }

    /* ── Specific: small offset (< quarter window) should give 0 BER
     * because the contaminated window decodes to the dominant symbol. */
    printf("\n  [Small offset (< window/4): expect 0 BER]\n");
    for (int M = 2; M <= 8; M *= 2) {
        assert_loopback_zero_ber("hello-small-off", M,
                                  (const uint8_t *)hello, strlen(hello),
                                  PSK_FRAMES_PER_BUF_RX / 8);
    }

    /* ── Multiple M values with aligned loopback ────────────────── */
    printf("\n  [16-PSK and 32-PSK aligned loopback]\n");
    assert_loopback_zero_ber("hello-16PSK", 16,
                              (const uint8_t *)hello, strlen(hello), 0);
    assert_loopback_zero_ber("hello-32PSK", 32,
                              (const uint8_t *)hello, strlen(hello), 0);
    assert_loopback_zero_ber("ramp-32PSK",  32, ramp, 32, 0);
}

/* ═════════════════════════════════════════════════════════════════
 * main
 * ═════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║      M-PSK Implementation Test Bench         ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    test_utility();
    test_gray_coding();
    test_bit_packing();
    test_tx_buffer();
    test_dft();
    test_phase_decode();
    test_timing();
    test_pll();
    test_loopback();

    printf("\n");
    printf("══════════════════════════════════════════════\n");
    printf("  " C_GREEN "Passed" C_RESET ": %3d\n", g_pass);
    if (g_fail > 0)
        printf("  " C_RED "Failed" C_RESET ": %3d\n", g_fail);
    else
        printf("  Failed: %3d\n", g_fail);
    printf("  Info  : %3d  (observational — not counted)\n", g_skip);
    printf("══════════════════════════════════════════════\n");

    return g_fail > 0 ? 1 : 0;
}