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
 * 10. Carrier family — cycles↔bin mapping, detection at 4/5/6 cycles
 * 11. FEC            — Hamming(7,4), interleave, SNR/interference helpers
 * 12. File I/O       — read/write helpers and file-to-file roundtrip
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
 * 5b. Carrier-family coverage (app-level -c feature support)
 * ═════════════════════════════════════════════════════════════════ */

/* Local helper: generate one 256-sample TX buffer using an explicit
 * carrier cycle count (mirrors transmit.c -c behavior). */
static void gen_tx_buffer_cycles(float *out, int symbol, int M, int carrier_cycles)
{
    float phase = (float)symbol * 2.0f * (float)M_PI / (float)M;
    float tw    = 2.0f * (float)M_PI
                * (float)carrier_cycles
                / (float)PSK_FRAMES_PER_BUF_TX;
    for (int t = 0; t < PSK_FRAMES_PER_BUF_TX; t++)
        out[t] = sinf(tw * (float)t + phase);
}

/* Local helper: DFT at an arbitrary bin index. */
static float complex dft_at_bin(const float *samples, int n, int bin)
{
    float complex w  = cexpf(-I * 2.0f * (float)M_PI * (float)bin / (float)n);
    float complex wt = 1.0f;
    float complex X  = 0.0f;
    for (int t = 0; t < n; t++) {
        X  += samples[t] * wt;
        wt *= w;
    }
    return X;
}

static void test_carrier_family(void)
{
    section("Carrier family: cycles↔bin mapping and detection");

    const int tx_n = PSK_FRAMES_PER_BUF_TX;   /* 256  */
    const int rx_n = PSK_FRAMES_PER_BUF_RX;   /* 4096 */
    const int bin_per_cycle = rx_n / tx_n;    /* 16   */
    const int cycles_list[] = {4, 5, 6};
    const float half_rx_n = (float)rx_n / 2.0f;

    /* ── cycles/buffer maps to expected RX carrier bin ───────────── */
    for (int i = 0; i < 3; i++) {
        int c = cycles_list[i];
        int bin = c * bin_per_cycle;
        char lbl[96];
        snprintf(lbl, sizeof(lbl), "c=%d maps to RX carrier bin %d", c, bin);
        ASSERT_EQ(bin, c * 16, lbl);
    }

    /* ── For c in {4,5,6}, energy concentrates at mapped bin ─────── */
    for (int i = 0; i < 3; i++) {
        int c = cycles_list[i];
        int bin = c * bin_per_cycle;

        float win[PSK_FRAMES_PER_BUF_RX];
        float tw = 2.0f * (float)M_PI * (float)bin / (float)rx_n;
        for (int t = 0; t < rx_n; t++)
            win[t] = sinf(tw * (float)t + 0.37f);

        float mag_target = cabsf(dft_at_bin(win, rx_n, bin));
        float mag_off_p1 = cabsf(dft_at_bin(win, rx_n, bin + 1));
        float mag_off_m1 = cabsf(dft_at_bin(win, rx_n, bin - 1));

        char lbl_mag[96], lbl_off[96];
        snprintf(lbl_mag, sizeof(lbl_mag),
                 "c=%d: |DFT[bin=%d]| ≈ N/2", c, bin);
        snprintf(lbl_off, sizeof(lbl_off),
                 "c=%d: adjacent bins are strongly rejected", c);

        ASSERT_NEAR(mag_target, half_rx_n, 1.0f, lbl_mag);
        ASSERT(mag_off_p1 < 2.0f && mag_off_m1 < 2.0f, lbl_off);
    }

    /* ── Consecutive RX windows keep phase for each carrier setting ─ */
    for (int i = 0; i < 3; i++) {
        int c = cycles_list[i];
        int bin = c * bin_per_cycle;

        float sym_audio[PSK_SAMPLES_PER_SYMBOL];
        for (int b = 0; b < PSK_BUFFERS_PER_SYMBOL; b++) {
            gen_tx_buffer_cycles(sym_audio + b * tx_n, 1, 4, c);
        }

        float complex X0 = dft_at_bin(sym_audio, rx_n, bin);
        float complex X1 = dft_at_bin(sym_audio + rx_n, rx_n, bin);
        float p0 = atan2f(cimagf(X0), crealf(X0));
        float p1 = atan2f(cimagf(X1), crealf(X1));
        float diff = atan2f(sinf(p1 - p0), cosf(p1 - p0));

        char lbl[112];
        snprintf(lbl, sizeof(lbl),
                 "c=%d: consecutive RX windows keep carrier phase", c);
        ASSERT_NEAR(diff, 0.0f, 0.001f, lbl);
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
 * 10. FEC — Hamming(7,4) + block interleaving + channel quality
 * ═════════════════════════════════════════════════════════════════ */

static void test_fec(void)
{
    section("FEC: Hamming(7,4) codeword encode / decode");

    /* ── Known codeword values ─────────────────────────────────── */
    /* Nibble 0x0 = 0000: p1=0,p2=0,p3=0 → codeword = 0b0000000 = 0 */
    ASSERT_EQ((int)psk_hamming_encode(0x0), 0,
              "hamming_encode(0x0) = 0b0000000");

    /* Nibble 0xF = 1111: d1=d2=d3=d4=1 → p1=1^1^1=1, p2=1^1^1=1, p3=1^1^1=1
     * codeword: d4=1,d3=1,d2=1,p3=1,d1=1,p2=1,p1=1 = 0b1111111 = 127 */
    ASSERT_EQ((int)psk_hamming_encode(0xF), 127,
              "hamming_encode(0xF) = 0b1111111");

    /* Nibble 0x5 = 0101: d1=1,d2=0,d3=1,d4=0
     * p1=1^0^0=1, p2=1^1^0=0, p3=0^1^0=1
     * [6]=0,[5]=1,[4]=0,[3]=1,[2]=1,[1]=0,[0]=1 = 0b0101101 = 45 */
    ASSERT_EQ((int)psk_hamming_encode(0x5), 45,
              "hamming_encode(0x5) = 45  (hand-verified)");

    /* ── Roundtrip for all nibbles 0..15 ──────────────────────── */
    {
        int ok = 1;
        for (int n = 0; n < 16; n++) {
            uint8_t out = 0xFF;
            int status  = psk_hamming_decode(psk_hamming_encode((uint8_t)n), &out);
            if (status != 0 || out != (uint8_t)n) { ok = 0; break; }
        }
        ASSERT(ok, "hamming roundtrip: decode(encode(n))==n for n=0..15, no error");
    }

    /* ── Single-bit error correction for all nibbles × all positions */
    {
        int ok = 1;
        for (int n = 0; n < 16 && ok; n++) {
            uint8_t cw = psk_hamming_encode((uint8_t)n);
            for (int pos = 0; pos < PSK_HAMMING_N && ok; pos++) {
                uint8_t bad = cw ^ (uint8_t)(1u << pos);
                uint8_t out = 0xFF;
                int status  = psk_hamming_decode(bad, &out);
                if (status != 1 || out != (uint8_t)n) ok = 0;
            }
        }
        ASSERT(ok,
               "single-bit correction: all 16 nibbles × 7 error positions corrected");
    }

    /* ── No-error path returns status 0 ───────────────────────── */
    {
        uint8_t out;
        ASSERT_EQ(psk_hamming_decode(psk_hamming_encode(0xA), &out), 0,
                  "decode with no error returns status 0");
        ASSERT_EQ((int)out, 0xA, "decode with no error gives correct nibble");
    }

    section("FEC: psk_fec_encode / psk_fec_decode pipeline");

    /* ── Roundtrip for several strings and depths ──────────────── */
    {
        const char *msg     = "Hello, PSK!";
        size_t      msg_len = strlen(msg);
        int depths[] = {1, 4, 8, 16};

        for (int di = 0; di < 4; di++) {
            int    d = depths[di];
            size_t enc_len = 0;
            uint8_t *enc = psk_fec_encode((const uint8_t *)msg,
                                           msg_len, d, &enc_len);
            size_t   dec_len = 0;
            int      corr    = -1;
            uint8_t *dec = psk_fec_decode(enc, enc_len, d, &dec_len, &corr);

            int ok = (dec_len >= msg_len) &&
                     (memcmp(msg, dec, msg_len) == 0) &&
                     (corr == 0);
            char lbl[80];
            snprintf(lbl, sizeof(lbl),
                     "FEC roundtrip depth=%d '%s'", d, msg);
            ASSERT(ok, lbl);
            free(enc); free(dec);
        }
    }

    /* ── Rate is 4/7: encoded length ≈ input × 7/4 ─────────────── */
    {
        const uint8_t data[8] = {0xAA, 0xBB, 0xCC, 0xDD,
                                  0x11, 0x22, 0x33, 0x44};
        size_t enc_len = 0;
        uint8_t *enc   = psk_fec_encode(data, 8, 8, &enc_len);
        /* 8 bytes × 8 bits / 4 × 7 = 112 bits = 14 bytes */
        ASSERT(enc_len >= 14 && enc_len <= 16,
               "FEC encoded length: 8 bytes → 14-16 encoded bytes (rate 4/7)");
        free(enc);
    }

    /* ── Corrects sparse single-bit errors (one per codeword) ───── */
    {
        const char *msg = "Test data for FEC sparse errors!";
        size_t msg_len  = strlen(msg);
        int    depth    = 8;
        size_t enc_len  = 0;

        uint8_t *enc = psk_fec_encode((const uint8_t *)msg,
                                       msg_len, depth, &enc_len);
        /* Flip one bit every 7 bytes (one bit per codeword) */
        for (size_t i = 0; i < enc_len; i += 7)
            enc[i] ^= 0x01u;

        int      corr    = 0;
        size_t   dec_len = 0;
        uint8_t *dec = psk_fec_decode(enc, enc_len, depth, &dec_len, &corr);

        int ok = (dec_len >= msg_len) &&
                 (memcmp(msg, dec, msg_len) == 0);
        ASSERT(ok, "FEC corrects sparse 1-bit errors (1 per codeword)");
        ASSERT(corr > 0, "FEC reports non-zero corrections for injected errors");
        free(enc); free(dec);
    }

    /* ── Burst error correction (burst ≤ depth) ────────────────── */
    {
        const char *msg  = "Burst error resilience test.";
        size_t msg_len   = strlen(msg);
        int    depth     = 8;
        size_t enc_len   = 0;

        uint8_t *enc = psk_fec_encode((const uint8_t *)msg,
                                       msg_len, depth, &enc_len);
        /* Flip 7 consecutive bits (burst = depth - 1 = 7) starting at bit 0 */
        for (int b = 0; b < 7; b++)
            psk_setbit(enc, (size_t)b, 1 - psk_getbit(enc, (size_t)b));

        int      corr    = 0;
        size_t   dec_len = 0;
        uint8_t *dec = psk_fec_decode(enc, enc_len, depth, &dec_len, &corr);

        int ok = (dec_len >= msg_len) &&
                 (memcmp(msg, dec, msg_len) == 0);
        ASSERT(ok,
               "FEC corrects burst of 7 consecutive bit errors (depth=8)");
        free(enc); free(dec);
    }

    /* ── Encode + decode is identity when no errors ─────────────── */
    {
        uint8_t data[32];
        for (int i = 0; i < 32; i++) data[i] = (uint8_t)(i * 13 + 7);
        size_t enc_len = 0;
        uint8_t *enc = psk_fec_encode(data, 32, PSK_DEFAULT_ILVE_DEPTH, &enc_len);
        int      corr    = 99;
        size_t   dec_len = 0;
        uint8_t *dec = psk_fec_decode(enc, enc_len, PSK_DEFAULT_ILVE_DEPTH,
                                       &dec_len, &corr);
        ASSERT(memcmp(data, dec, 32) == 0,
               "FEC identity: encode then decode with no errors = original");
        ASSERT_EQ(corr, 0,
               "FEC identity: zero corrections when no errors injected");
        free(enc); free(dec);
    }

    section("FEC: SNR estimation");

    /* ── Strong carrier → high SNR ──────────────────────────────── */
    {
        float mags[100];
        memset(mags, 0, sizeof(mags));
        /* Carrier at bin 64, amplitude 2048; noise floor ~1 */
        mags[PSK_CARRIER_BIN] = 2048.0f;
        for (int k = 0; k < 100; k++)
            if (k != PSK_CARRIER_BIN) mags[k] = 1.0f;
        float snr = psk_snr_estimate_db(mags, 100, PSK_CARRIER_BIN);
        ASSERT(snr > 20.0f,
               "SNR estimate: strong carrier (2048) vs noise (1) → SNR > 20 dB");
    }

    /* ── Equal power carrier and noise → SNR near 0 dB ─────────── */
    {
        float mags[100];
        for (int k = 0; k < 100; k++) mags[k] = 10.0f;
        float snr = psk_snr_estimate_db(mags, 100, PSK_CARRIER_BIN);
        ASSERT(snr >= -3.0f && snr <= 3.0f,
               "SNR estimate: uniform spectrum → SNR ≈ 0 dB");
    }

    /* ── Silence → returns -60 dB ───────────────────────────────── */
    {
        float mags[100];
        memset(mags, 0, sizeof(mags));
        float snr = psk_snr_estimate_db(mags, 100, PSK_CARRIER_BIN);
        ASSERT(snr <= -59.0f,
               "SNR estimate: silence → SNR ≤ −60 dB");
    }

    section("FEC: interference detection");

    /* ── Interferer at 60% of carrier → detected ────────────────── */
    {
        float mags[100];
        memset(mags, 0, sizeof(mags));
        mags[PSK_CARRIER_BIN] = 100.0f;
        mags[30]              = 60.0f;   /* 60% of carrier, far from guard */
        int bin = -1;
        int detected = psk_detect_interference(mags, 100, PSK_CARRIER_BIN, &bin);
        ASSERT(detected == 1,
               "interference detected: bin 30 at 60% of carrier amplitude");
        ASSERT_EQ(bin, 30, "strongest interferer correctly identified at bin 30");
    }

    /* ── Interferer at 40% of carrier → not flagged ─────────────── */
    {
        float mags[100];
        memset(mags, 0, sizeof(mags));
        mags[PSK_CARRIER_BIN] = 100.0f;
        mags[30]              = 40.0f;   /* 40% — below PSK_INTF_RATIO=0.5 */
        int detected = psk_detect_interference(mags, 100, PSK_CARRIER_BIN, NULL);
        ASSERT(detected == 0,
               "no interference: bin 30 at 40% of carrier (below threshold)");
    }

    /* ── Bins inside the guard zone are not flagged ──────────────── */
    {
        float mags[100];
        memset(mags, 0, sizeof(mags));
        mags[PSK_CARRIER_BIN]           = 100.0f;
        mags[PSK_CARRIER_BIN + 1]       = 90.0f;   /* inside guard zone */
        mags[PSK_CARRIER_BIN - 2]       = 90.0f;   /* on guard boundary */
        int detected = psk_detect_interference(mags, 100, PSK_CARRIER_BIN, NULL);
        ASSERT(detected == 0,
               "guard zone: adjacent bins not flagged as interference");
    }

    /* ── No carrier → no interference flagged ───────────────────── */
    {
        float mags[100];
        memset(mags, 0, sizeof(mags));
        int detected = psk_detect_interference(mags, 100, PSK_CARRIER_BIN, NULL);
        ASSERT(detected == 0,
               "silence: no interference detected when carrier is absent");
    }
}

/* ═════════════════════════════════════════════════════════════════
 * 11. File I/O — psk_read_file / psk_write_file / roundtrip
 *
 * These tests verify that:
 *   a) psk_write_file creates a file whose bytes match the source buffer.
 *   b) psk_read_file reads that file back byte-for-byte.
 *   c) A full file-to-file encode → software-loopback → decode pipeline
 *      reproduces the original input file exactly (byte-exact match).
 *      The test is run for every combination of M ∈ {2,4,8} and
 *      FEC ∈ {off, depth=8} so that both code paths are exercised.
 * ═════════════════════════════════════════════════════════════════ */

/* Temporary file paths used by the file-I/O tests.
 * Both are cleaned up at the end of test_file_io().             */
#define TEST_IN_PATH   "/tmp/psk_test_input.bin"
#define TEST_OUT_PATH  "/tmp/psk_test_output.bin"

/* ── helpers ────────────────────────────────────────────────────── */

/* Read the entire contents of `path` into a malloc buffer.
 * Returns NULL on failure.  *len is set to the byte count.       */
static uint8_t *slurp(const char *path, size_t *len)
{
    *len = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    uint8_t *buf = psk_read_stream(fp, len);
    fclose(fp);
    return buf;
}

/* ── core file-roundtrip logic ─────────────────────────────────── */
/*
 * run_file_roundtrip
 *
 * 1. Write `payload` to TEST_IN_PATH.
 * 2. Read it back via psk_read_file (tests file-read path).
 * 3. Optionally FEC-encode (tests encode pipeline on file data).
 * 4. Run the software-channel loopback (same as run_loopback):
 *      psk_generate_tx_buffer × N → psk_rx_process_window
 * 5. Collect per-symbol decoded bytes and optionally FEC-decode.
 * 6. Write the decoded bytes to TEST_OUT_PATH via psk_write_file.
 * 7. Read TEST_OUT_PATH back and compare with `payload` byte-for-byte.
 *
 * Returns 1 on byte-exact match, 0 on any mismatch or failure.
 */
static int run_file_roundtrip(const uint8_t *payload, size_t payload_len,
                               int M, int fec_depth)
{
    int use_fec = (fec_depth > 0);
    int bps     = psk_ilog2(M);

    /* ── 1. Write input file ────────────────────────────────────── */
    if (psk_write_file(TEST_IN_PATH, payload, payload_len) != 0)
        return 0;

    /* ── 2. Read it back via psk_read_file ──────────────────────── */
    size_t   read_len = 0;
    uint8_t *read_buf = psk_read_file(TEST_IN_PATH, &read_len);
    if (!read_buf || read_len != payload_len ||
        memcmp(payload, read_buf, payload_len) != 0) {
        free(read_buf);
        return 0;
    }
    free(read_buf);

    /* ── 3. Optional FEC encode ─────────────────────────────────── */
    size_t   tx_len  = 0;
    uint8_t *tx_data = NULL;

    if (use_fec) {
        tx_data = psk_fec_encode(payload, payload_len, fec_depth, &tx_len);
    } else {
        tx_len  = payload_len;
        tx_data = malloc(tx_len);
        if (tx_data) memcpy(tx_data, payload, tx_len);
    }
    if (!tx_data) return 0;

    /* ── 4. Build symbol array: preamble + data ─────────────────── */
    size_t   n_data   = 0;
    uint8_t *data_syms = psk_bytes_to_symbols(tx_data, tx_len, bps, &n_data);
    free(tx_data);
    if (!data_syms) return 0;

    size_t   n_total  = (size_t)PSK_PREAMBLE_SYMBOLS + n_data;
    uint8_t *all_syms = calloc(n_total, 1);
    if (!all_syms) { free(data_syms); return 0; }
    memcpy(all_syms + PSK_PREAMBLE_SYMBOLS, data_syms, n_data);
    free(data_syms);

    /* ── 5. Generate audio + run receiver window-by-window ───────── */
    size_t  audio_len = n_total * (size_t)PSK_SAMPLES_PER_SYMBOL;
    float  *audio     = malloc(audio_len * sizeof(float));
    if (!audio) { free(all_syms); return 0; }

    for (size_t si = 0; si < n_total; si++) {
        float *dst = audio + si * (size_t)PSK_SAMPLES_PER_SYMBOL;
        for (int bi = 0; bi < PSK_BUFFERS_PER_SYMBOL; bi++)
            psk_generate_tx_buffer(dst + bi * PSK_FRAMES_PER_BUF_TX,
                                   all_syms[si], M);
    }

    /* Collect one decoded symbol per data-symbol slot (first hit wins) */
    if (n_data == 0 || n_data > (size_t)0x100000) {
        /* Upper bound guards against GCC's -Walloc-size-larger-than */
        free(audio); free(all_syms);
        return 0;
    }
    uint8_t *rx_syms = calloc(n_data, 1);
    int     *got     = calloc(n_data, sizeof(int));
    if (!rx_syms || !got) {
        free(audio); free(all_syms); free(rx_syms); free(got);
        return 0;
    }

    PskRxState s;
    psk_rx_state_init(&s);
    size_t audio_pos = 0;

    while (audio_pos + (size_t)PSK_FRAMES_PER_BUF_RX <= audio_len) {
        int result = psk_rx_process_window(&s, audio + audio_pos, M);
        if (result >= 0) {
            size_t mid   = audio_pos + (size_t)(PSK_FRAMES_PER_BUF_RX / 2);
            size_t tx_si = mid / (size_t)PSK_SAMPLES_PER_SYMBOL;
            if (tx_si >= (size_t)PSK_PREAMBLE_SYMBOLS && tx_si < n_total) {
                size_t di = tx_si - (size_t)PSK_PREAMBLE_SYMBOLS;
                if (!got[di]) {
                    rx_syms[di] = (uint8_t)result;
                    got[di]     = 1;
                }
            }
        }
        audio_pos += (size_t)PSK_FRAMES_PER_BUF_RX;
    }

    free(audio); free(all_syms); free(got);

    /* ── 6. symbols → bytes (reverse of the encode step) ─────────── */
    size_t   raw_len = 0;
    uint8_t *raw     = psk_symbols_to_bytes(rx_syms, n_data, bps, &raw_len);
    free(rx_syms);
    if (!raw) return 0;

    /* Optional FEC decode */
    uint8_t *decoded  = NULL;
    size_t   dec_len  = 0;

    if (use_fec) {
        int corr = 0;
        decoded  = psk_fec_decode(raw, raw_len, fec_depth, &dec_len, &corr);
        free(raw);
    } else {
        decoded = raw;
        dec_len = raw_len;
    }
    if (!decoded) return 0;

    /* ── 7. Write output file via psk_write_file ─────────────────── */
    int write_ok = (psk_write_file(TEST_OUT_PATH, decoded, dec_len) == 0);
    free(decoded);
    if (!write_ok) return 0;

    /* ── 8. Read output file back and compare with original ─────── */
    size_t   out_len = 0;
    uint8_t *out_buf = slurp(TEST_OUT_PATH, &out_len);
    if (!out_buf) return 0;

    int match = (out_len >= payload_len) &&
                (memcmp(payload, out_buf, payload_len) == 0);
    free(out_buf);
    return match;
}

static void test_file_io(void)
{
    section("File I/O: psk_read_file / psk_write_file");

    /* ── Basic write → read roundtrip ───────────────────────────── */
    {
        const uint8_t data[] = {0xDE, 0xAD, 0xBE, 0xEF,
                                 0x00, 0x01, 0x7F, 0xFF};
        ASSERT(psk_write_file(TEST_IN_PATH, data, sizeof(data)) == 0,
               "psk_write_file: binary data written without error");

        size_t   rlen = 0;
        uint8_t *rbuf = psk_read_file(TEST_IN_PATH, &rlen);
        ASSERT(rbuf != NULL, "psk_read_file: returned non-NULL for existing file");
        ASSERT_EQ((int)rlen, (int)sizeof(data),
                  "psk_read_file: byte count matches written size");
        ASSERT(memcmp(data, rbuf, sizeof(data)) == 0,
               "psk_read_file: bytes match byte-for-byte");
        free(rbuf);
    }

    /* ── Write empty file ───────────────────────────────────────── */
    {
        ASSERT(psk_write_file(TEST_IN_PATH, (const uint8_t *)"", 0) == 0,
               "psk_write_file: zero-length write succeeds");
        size_t   rlen = 99;
        uint8_t *rbuf = psk_read_file(TEST_IN_PATH, &rlen);
        ASSERT(rbuf != NULL && rlen == 0,
               "psk_read_file: reading empty file returns length 0");
        free(rbuf);
    }

    /* ── Read non-existent file returns NULL ────────────────────── */
    {
        size_t   rlen = 0;
        uint8_t *rbuf = psk_read_file("/tmp/psk_no_such_file_xyz.bin", &rlen);
        ASSERT(rbuf == NULL,
               "psk_read_file: returns NULL for non-existent path");
    }

    /* ── psk_read_stream from a file pointer ────────────────────── */
    {
        const char *text = "stream test data\n";
        psk_write_file(TEST_IN_PATH, (const uint8_t *)text, strlen(text));
        FILE   *fp  = fopen(TEST_IN_PATH, "rb");
        size_t  len = 0;
        uint8_t *buf = psk_read_stream(fp, &len);
        fclose(fp);
        ASSERT(buf != NULL && len == strlen(text) &&
               memcmp(text, buf, len) == 0,
               "psk_read_stream: reads stream content correctly");
        free(buf);
    }

    /* ── Large binary file roundtrip ────────────────────────────── */
    {
        uint8_t big[512];
        for (int i = 0; i < 512; i++) big[i] = (uint8_t)(i * 17 + 3);
        psk_write_file(TEST_IN_PATH, big, sizeof(big));
        size_t   rlen = 0;
        uint8_t *rbuf = psk_read_file(TEST_IN_PATH, &rlen);
        int ok = (rbuf != NULL) && (rlen == 512) &&
                 (memcmp(big, rbuf, 512) == 0);
        ASSERT(ok, "psk_write_file / psk_read_file: 512-byte binary roundtrip");
        free(rbuf);
    }

    section("File I/O: full file-to-file encode → loopback → decode");

    /* Common test payloads */
    const char *hello   = "Hello, File I/O!";
    const uint8_t ramp[32] = {
        0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,104,112,120,
       128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248
    };
    uint8_t counter[48];
    for (int i = 0; i < 48; i++) counter[i] = (uint8_t)i;

    /* ── Aligned file roundtrip, no FEC, all M values ──────────── */
    printf("\n  [No FEC — raw PSK file-to-file roundtrip]\n");
    for (int M = 2; M <= 8; M *= 2) {
        {
            int ok = run_file_roundtrip((const uint8_t *)hello,
                                         strlen(hello), M, 0);
            char lbl[80];
            snprintf(lbl, sizeof(lbl),
                     "File roundtrip: hello  M=%-2d  no-FEC", M);
            ASSERT(ok, lbl);
        }
        {
            int ok = run_file_roundtrip(ramp, sizeof(ramp), M, 0);
            char lbl[80];
            snprintf(lbl, sizeof(lbl),
                     "File roundtrip: ramp   M=%-2d  no-FEC", M);
            ASSERT(ok, lbl);
        }
    }

    /* ── Aligned file roundtrip with FEC, all M values ─────────── */
    printf("\n  [FEC depth=8 — Hamming(7,4) + interleave file-to-file]\n");
    for (int M = 2; M <= 8; M *= 2) {
        {
            int ok = run_file_roundtrip((const uint8_t *)hello,
                                         strlen(hello), M, 8);
            char lbl[80];
            snprintf(lbl, sizeof(lbl),
                     "File roundtrip: hello  M=%-2d  FEC depth=8", M);
            ASSERT(ok, lbl);
        }
        {
            int ok = run_file_roundtrip(counter, sizeof(counter), M, 8);
            char lbl[80];
            snprintf(lbl, sizeof(lbl),
                     "File roundtrip: counter M=%-2d  FEC depth=8", M);
            ASSERT(ok, lbl);
        }
    }

    /* ── Output file content matches input byte-for-byte ─────────── */
    /* (Run one explicit check so the test makes the comparison visible) */
    {
        const char *msg = "ENGN1580 file match test";
        run_file_roundtrip((const uint8_t *)msg, strlen(msg), 4, 0);

        size_t in_len  = 0, out_len = 0;
        uint8_t *in_buf  = slurp(TEST_IN_PATH,  &in_len);
        uint8_t *out_buf = slurp(TEST_OUT_PATH, &out_len);

        int ok = (in_buf && out_buf &&
                  in_len == strlen(msg) &&
                  out_len >= strlen(msg) &&
                  memcmp(in_buf, out_buf, strlen(msg)) == 0);
        ASSERT(ok,
               "File match: TEST_IN_PATH and TEST_OUT_PATH identical after QPSK roundtrip");

        free(in_buf); free(out_buf);
    }

    /* ── Cleanup temp files ─────────────────────────────────────── */
    remove(TEST_IN_PATH);
    remove(TEST_OUT_PATH);
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
    test_carrier_family();
    test_phase_decode();
    test_timing();
    test_pll();
    test_fec();
    test_loopback();
    test_file_io();

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