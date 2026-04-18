/*
 * psk_common.h — Shared M-PSK signal processing primitives
 *
 * This header is the single source of truth for all signal-processing
 * mathematics used by transmit.c, receive.c, and test.c.  It has no
 * dependencies on PortAudio or SDL2 and can be included in any plain
 * C translation unit.
 *
 * All public symbols are prefixed with psk_ to avoid name collisions
 * with the static helpers that transmit.c and receive.c define
 * internally.  Every function is declared static inline so the header
 * can be included in multiple translation units without linker errors.
 *
 * Contents (in order)
 * ───────────────────
 *   1. System constants — sample rate, buffer sizes, carrier geometry
 *   2. Recovery parameters — signal threshold, calibration, PLL gain
 *   3. Utility — power-of-two test, integer log₂
 *   4. Gray coding — encode, decode, adjacency guaranteed
 *   5. Bit packing — bytes → Gray-coded symbols, symbols → bytes
 *   6. Transmitter — generate one 256-sample carrier buffer
 *   7. Receiver DFT — compute DFT[CARRIER_BIN] for a 4096-sample window
 *   8. Phase decoding — corrected phase → nearest symbol index
 *   9. PskRxState — receiver state struct and process-window function
 *
 * Key frequency relationship
 * ──────────────────────────
 * The carrier frequency is chosen so that it lies at an exact DFT bin
 * in both the transmitter and receiver:
 *
 *   Transmitter: CARRIER_CYCLES / FRAMES_PER_BUF_TX = 4/256   cycles/sample
 *   Receiver:    CARRIER_BIN    / FRAMES_PER_BUF_RX = 64/4096 cycles/sample
 *
 * Both equal 1/64 cycles/sample = 750 Hz at 48 000 Hz.
 *
 * Because the carrier completes exactly CARRIER_BIN full cycles per
 * receiver window, the DFT phase is identical for every window that
 * lies entirely within a single symbol (no drift, no inter-window
 * accumulation).  This is the foundation that makes the calibration
 * and timing-recovery algorithms work.
 *
 * Phase offset identity
 * ─────────────────────
 * For a pure sine wave  sin(ω₀ t + φ)  at the carrier frequency, the
 * DFT at bin CARRIER_BIN gives:
 *
 *   X[k] = −i · (N/2) · e^{iφ}
 *
 * so  angle(X[k]) = φ − π/2.
 *
 * The receiver calibrates this −π/2 offset from the known-phase
 * preamble.  After calibration, corrected_phase = received_phase −
 * phase_offset = φ.
 */

#ifndef PSK_COMMON_H
#define PSK_COMMON_H

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════
 * 1. System constants
 * ═══════════════════════════════════════════════════════════════════ */

#define PSK_SAMPLE_RATE       48000   /* audio sample rate (Hz)              */
#define PSK_FRAMES_PER_BUF_TX   256   /* transmitter PortAudio buffer size   */
#define PSK_FRAMES_PER_BUF_RX  4096   /* receiver analysis window (0x1000)   */
#define PSK_CARRIER_CYCLES       4    /* cycles per TX buffer → 750 Hz       */
#define PSK_CARRIER_BIN         64    /* DFT bin in RX window → 750 Hz       */
#define PSK_BUFFERS_PER_SYMBOL  64    /* TX buffers per symbol = 16 384 samp */
#define PSK_SYM_WINDOWS          4    /* RX windows per symbol (exact)       */
#define PSK_PREAMBLE_SYMBOLS     4    /* phase-0° symbols prepended to data  */

/* Derived: total audio samples for one symbol */
#define PSK_SAMPLES_PER_SYMBOL  \
    (PSK_BUFFERS_PER_SYMBOL * PSK_FRAMES_PER_BUF_TX)   /* = 16 384 */

/* Sanity: RX windows per symbol must equal PSK_SYM_WINDOWS */
/* PSK_SAMPLES_PER_SYMBOL / PSK_FRAMES_PER_BUF_RX = 16384/4096 = 4 ✓ */

/* ═══════════════════════════════════════════════════════════════════
 * 2. Recovery parameters
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * PSK_SIG_THRESHOLD
 *   Minimum |DFT[CARRIER_BIN]| required to consider the carrier present.
 *   A full-scale sine gives |DFT| = N/2 = 2048; threshold = 150 ≈ 7%.
 */
#define PSK_SIG_THRESHOLD   150.0f

/*
 * PSK_CAL_WINDOWS
 *   Number of receiver windows averaged for preamble phase calibration.
 *   = 2 preamble symbols × PSK_SYM_WINDOWS windows/symbol.
 */
#define PSK_CAL_WINDOWS      8

/*
 * PSK_PLL_ALPHA
 *   First-order decision-directed PLL loop gain.
 *   After each clean (non-boundary) symbol decision the loop integrates:
 *     phase_offset += PSK_PLL_ALPHA × phase_error
 *   Time constant ≈ 1/α frames ≈ 40 frames ≈ 3.4 s at 12 Hz.
 */
#define PSK_PLL_ALPHA       0.025f

/*
 * PSK_AGC_ALPHA
 *   Smoothing gain for the exponential AGC amplitude tracker.
 *   agc_mag = α·|ft| + (1−α)·agc_mag
 *   Time constant ≈ 1/α frames ≈ 1.4 s at 12 Hz.
 */
#define PSK_AGC_ALPHA       0.05f

/* ═══════════════════════════════════════════════════════════════════
 * 3. Utility
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * psk_is_power_of_2 — returns non-zero iff n is a power of two and n ≥ 2.
 */
static inline int psk_is_power_of_2(int n)
{
    return (n >= 2) && ((n & (n - 1)) == 0);
}

/*
 * psk_ilog2 — integer base-2 logarithm.
 * Valid only for exact powers of two; behaviour undefined otherwise.
 */
static inline int psk_ilog2(int n)
{
    int k = 0;
    while (n > 1) { n >>= 1; ++k; }
    return k;
}

/* ═══════════════════════════════════════════════════════════════════
 * 4. Gray coding
 *
 * The binary-reflected Gray code maps consecutive integers to code
 * words that differ by exactly one bit.  This ensures that a symbol
 * decision error to an adjacent constellation point causes only a
 * single bit error in the decoded output, minimising bit-error rate
 * near decision boundaries.
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * psk_gray_encode — natural binary n → Gray code word.
 * Property:  psk_gray_encode(n) XOR psk_gray_encode(n±1)  has exactly
 *            one bit set (the single-step Hamming-distance guarantee).
 */
static inline int psk_gray_encode(int n)
{
    return n ^ (n >> 1);
}

/*
 * psk_gray_decode — Gray code word → natural binary.
 * Inverse of psk_gray_encode; satisfies
 *   psk_gray_decode(psk_gray_encode(n)) == n  for all n ≥ 0.
 */
static inline int psk_gray_decode(int g)
{
    int n = g;
    for (int mask = g >> 1; mask; mask >>= 1)
        n ^= mask;
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * 5. Bit packing
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * psk_bytes_to_symbols
 *
 * Pack raw data bytes into an array of Gray-coded M-PSK symbol indices
 * (each in [0, M-1]).
 *
 * Bits are consumed MSB-first from each input byte and grouped into
 * chunks of  bps = log2(M)  bits.  The final chunk is zero-padded on
 * the right when the total bit count is not a multiple of bps.
 *
 * Returns a calloc-allocated array; caller must free().
 * *out_n is set to the number of symbols produced.
 * Returns NULL on allocation failure.
 */
static inline uint8_t *psk_bytes_to_symbols(const uint8_t *data,
                                             size_t         data_len,
                                             int            bps,
                                             size_t        *out_n)
{
    size_t   total_bits = data_len * 8;
    size_t   n_syms     = (total_bits + (size_t)bps - 1) / (size_t)bps;
    uint8_t *syms       = (uint8_t *)calloc(n_syms, 1);
    if (!syms) return NULL;

    for (size_t s = 0; s < n_syms; s++) {
        int raw = 0;
        for (int b = 0; b < bps; b++) {
            size_t bit_idx = s * (size_t)bps + (size_t)b;
            int    bit     = 0;
            if (bit_idx < total_bits) {
                size_t byte_i = bit_idx / 8;
                int    bit_i  = 7 - (int)(bit_idx % 8);   /* MSB first */
                bit = (data[byte_i] >> bit_i) & 1;
            }
            raw = (raw << 1) | bit;
        }
        syms[s] = (uint8_t)psk_gray_encode(raw);
    }

    *out_n = n_syms;
    return syms;
}

/*
 * psk_symbols_to_bytes
 *
 * Inverse of psk_bytes_to_symbols.  Decodes Gray-coded symbol indices
 * back into bytes (MSB-first).  Only complete bytes are returned;
 * any trailing partial byte is discarded.
 *
 * Returns a calloc-allocated array; caller must free().
 * *out_len is set to the number of bytes produced.
 * Returns NULL on allocation failure.
 */
static inline uint8_t *psk_symbols_to_bytes(const uint8_t *syms,
                                             size_t         n_syms,
                                             int            bps,
                                             size_t        *out_len)
{
    size_t   total_bits = n_syms * (size_t)bps;
    size_t   n_bytes    = total_bits / 8;
    uint8_t *bytes      = (uint8_t *)calloc(n_bytes, 1);
    if (!bytes) return NULL;

    for (size_t s = 0; s < n_syms; s++) {
        int raw = psk_gray_decode(syms[s]);
        for (int b = 0; b < bps; b++) {
            size_t bit_idx = s * (size_t)bps + (size_t)b;
            if (bit_idx >= total_bits) break;
            size_t byte_i = bit_idx / 8;
            if (byte_i >= n_bytes) break;
            int bit_i = 7 - (int)(bit_idx % 8);
            if ((raw >> (bps - 1 - b)) & 1)
                bytes[byte_i] |= (uint8_t)(1u << bit_i);
        }
    }

    *out_len = n_bytes;
    return bytes;
}

/* ═══════════════════════════════════════════════════════════════════
 * 6. Transmitter: single 256-sample carrier buffer
 *
 * Fills PSK_FRAMES_PER_BUF_TX samples with:
 *
 *   out[t] = sin(2π · PSK_CARRIER_CYCLES · t / PSK_FRAMES_PER_BUF_TX
 *               + symbol × 2π / M)
 *
 * Because the carrier completes exactly PSK_CARRIER_CYCLES full cycles
 * per buffer, consecutive same-symbol buffers connect phase-continuously
 * with no discontinuity or wrap-around bookkeeping.
 * ═══════════════════════════════════════════════════════════════════ */
static inline void psk_generate_tx_buffer(float *out, int symbol, int M)
{
    float phase = (float)symbol * 2.0f * (float)M_PI / (float)M;
    float tw    = 2.0f * (float)M_PI
                * (float)PSK_CARRIER_CYCLES
                / (float)PSK_FRAMES_PER_BUF_TX;
    for (int t = 0; t < PSK_FRAMES_PER_BUF_TX; t++)
        out[t] = sinf(tw * (float)t + phase);
}

/* ═══════════════════════════════════════════════════════════════════
 * 7. Receiver DFT: complex carrier value for one 4096-sample window
 *
 * Returns:
 *   X[PSK_CARRIER_BIN] = Σ_{t=0}^{N-1} samples[t] · w^t,
 *                         w = e^{−i 2π · CARRIER_BIN / N}
 *
 * Uses the incremental twiddle-factor method: one cexpf call per
 * invocation (not per sample), one complex multiply per sample.
 *
 * For a pure sine  sin(ω₀ t + φ)  at the carrier frequency the result
 * is  X = −i · (N/2) · e^{iφ},  so:
 *   |X|       = N/2 = 2048
 *   angle(X)  = φ − π/2
 * ═══════════════════════════════════════════════════════════════════ */
static inline float complex psk_dft_carrier(const float *samples)
{
    const int N = PSK_FRAMES_PER_BUF_RX;

    float complex w  = cexpf(-I * 2.0f * (float)M_PI
                             * (float)PSK_CARRIER_BIN / (float)N);
    float complex wt = 1.0f;
    float complex X  = 0.0f;

    for (int t = 0; t < N; t++) {
        X  += samples[t] * wt;
        wt *= w;
    }
    return X;
}

/* ═══════════════════════════════════════════════════════════════════
 * 8. Phase decoding: corrected phase → nearest symbol index
 *
 * corrected_phase must already be in [0, 2π) — i.e., the raw received
 * DFT phase with phase_offset subtracted and normalised.
 *
 * Returns the integer symbol index in [0, M-1] whose ideal phase
 *   k × 2π/M
 * is closest to corrected_phase.
 * ═══════════════════════════════════════════════════════════════════ */
static inline int psk_phase_to_symbol(float corrected_phase, int M)
{
    int sym = (int)roundf((float)M * corrected_phase / (2.0f * (float)M_PI));
    return ((sym % M) + M) % M;
}

/* ═══════════════════════════════════════════════════════════════════
 * 9. PskRxState — receiver state machine
 *
 * Holds all mutable state for one receiver session, covering:
 *
 *   Carrier state machine (WAITING → CALIBRATING → DECODING)
 *   ───────────────────────────────────────────────────────────
 *   WAITING     : carrier below threshold
 *   CALIBRATING : carrier appeared; accumulating preamble phases
 *                 via circular mean to find phase_offset
 *   DECODING    : phase_offset known; mapping phase → symbol
 *
 *   Timing recovery (phase-jump boundary detector)
 *   ───────────────────────────────────────────────
 *   Compares consecutive complex carrier values:
 *     delta       = ft_curr × conj(ft_prev)
 *     delta_phase = ∠(delta)  ∈ (−π, π]
 *   If |delta_phase| > π/M a symbol boundary crossed this window.
 *   sym_window counts 0 (boundary) → 1 → 2 → 3 → repeat.
 *   Only windows with sym_window ≠ 0 are decoded.
 *
 *   Decision-directed carrier PLL (first order)
 *   ────────────────────────────────────────────
 *   After each non-boundary decode:
 *     error        = corrected_phase − (symbol × 2π/M)
 *     phase_offset += PSK_PLL_ALPHA × error
 *   Drives residual calibration error to zero over ~40 frames.
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    /* ── Carrier state machine ──────────────────────────────── */
    int   rx_state;        /* 0=WAITING  1=CALIBRATING  2=DECODING */
    float cal_cos;         /* circular-mean phase accumulators      */
    float cal_sin;
    int   cal_count;       /* windows accumulated so far            */
    float phase_offset;    /* estimated phase bias (radians)        */

    /* ── Latest decode result ───────────────────────────────── */
    int   current_sym;     /* most recently decoded symbol index    */
    int   current_bits;    /* Gray-decoded bits for current_sym     */
    float pll_error_deg;   /* PLL phase error in degrees            */
    float carrier_mag;     /* |DFT[CARRIER_BIN]| from last window   */
    float carrier_phase_deg; /* raw received phase in degrees       */
    float agc_mag;         /* AGC: exponentially-smoothed |ft|      */

    /* ── Timing recovery ────────────────────────────────────── */
    int   sym_window;      /* position in 4-window cycle; -1=none  */
    int   is_boundary;     /* 1 when |delta_phase| > π/M           */
    int   timing_locked;   /* 1 after first boundary detection      */
    float ft_prev_re;      /* previous window complex carrier (Re)  */
    float ft_prev_im;      /* previous window complex carrier (Im)  */
} PskRxState;

/*
 * psk_rx_state_init — zero-fill state and set sym_window to −1.
 * Must be called before the first call to psk_rx_process_window.
 */
static inline void psk_rx_state_init(PskRxState *s)
{
    memset(s, 0, sizeof(*s));
    s->sym_window = -1;   /* −1 = timing not yet acquired */
}

/*
 * psk_rx_process_window
 *
 * Run one complete receiver analysis step for a single
 * PSK_FRAMES_PER_BUF_RX-sample window of audio.
 *
 * Steps (executed in order every call):
 *
 *   1. DFT at carrier bin   → magnitude and raw phase
 *   2. Timing recovery      → is_boundary, sym_window, timing_locked
 *   3. Carrier state machine→ WAITING / CALIBRATING / DECODING
 *   4. PLL update           → phase_offset refinement (DECODING only)
 *
 * Returns the decoded symbol index (0 … M−1) when a symbol decision
 * was made (DECODING state, non-boundary window).
 * Returns −1 in all other cases (boundary window, calibrating, no
 * carrier, etc.).
 *
 * All persistent state is updated in *s in place.
 */
static inline int psk_rx_process_window(PskRxState  *s,
                                        const float *window,
                                        int          M)
{
    const float two_pi = 2.0f * (float)M_PI;

    /* ── Step 1: DFT at carrier bin ─────────────────────────── */
    float complex ft  = psk_dft_carrier(window);
    float         mag = cabsf(ft);
    float         phi = atan2f(cimagf(ft), crealf(ft));

    s->carrier_mag       = mag;
    s->carrier_phase_deg = phi * 180.0f / (float)M_PI;

    /* AGC: long-term amplitude envelope (decays slowly during silence) */
    s->agc_mag = PSK_AGC_ALPHA * mag
               + (1.0f - PSK_AGC_ALPHA) * s->agc_mag;

    /* ── Step 2: Timing recovery ─────────────────────────────
     *
     * Boundary detection uses the inter-window phase jump:
     *   delta       = ft_curr × conj(ft_prev)
     *   delta_phase = ∠(delta)
     *
     * Same-symbol windows give delta_phase ≈ 0 because the carrier
     * completes an exact integer number of cycles per window.
     * A symbol transition injects a step of k × (2π/M), detectable
     * when |delta_phase| exceeds the half-symbol threshold π/M.
     */
    s->is_boundary = 0;
    {
        float complex ft_prev = s->ft_prev_re + I * s->ft_prev_im;
        if (mag > PSK_SIG_THRESHOLD &&
            cabsf(ft_prev) > PSK_SIG_THRESHOLD) {

            float complex delta = ft * conjf(ft_prev);
            float         dp    = atan2f(cimagf(delta), crealf(delta));
            float         thr   = (float)M_PI / (float)M;

            if (fabsf(dp) > thr) {
                s->is_boundary   = 1;
                s->sym_window    = 0;
                s->timing_locked = 1;
            } else {
                if (s->timing_locked)
                    s->sym_window =
                        (s->sym_window + 1) % PSK_SYM_WINDOWS;
            }
        }
    }
    s->ft_prev_re = crealf(ft);
    s->ft_prev_im = cimagf(ft);

    /* ── Step 3: Carrier state machine ──────────────────────── */
    if (mag > PSK_SIG_THRESHOLD) {

        if (s->rx_state == 0) {
            s->rx_state  = 1;
            s->cal_cos   = 0.0f;
            s->cal_sin   = 0.0f;
            s->cal_count = 0;
        }

        if (s->rx_state == 1) {
            /* Circular mean: accumulate cos/sin of received phase
             * to avoid the ±π wrap discontinuity in a plain average. */
            s->cal_cos += cosf(phi);
            s->cal_sin += sinf(phi);
            if (++s->cal_count >= PSK_CAL_WINDOWS) {
                s->phase_offset = atan2f(s->cal_sin, s->cal_cos);
                s->rx_state     = 2;
            }
        }

        if (s->rx_state == 2 && !s->is_boundary) {
            /* Correct for offset, normalise to [0, 2π) */
            float corrected = phi - s->phase_offset;
            corrected = fmodf(corrected + 4.0f * (float)M_PI, two_pi);

            int sym = psk_phase_to_symbol(corrected, M);
            s->current_sym  = sym;
            s->current_bits = psk_gray_decode(sym);

            /* ── Step 4: PLL update ──────────────────────────
             *
             * Compute phase error as the signed arc from the ideal
             * constellation angle to the corrected received phase.
             * atan2(sin(e), cos(e)) wraps cleanly to (−π, π] and
             * avoids any discontinuity at the ±π boundary.
             *
             *   error        = corrected − (sym × 2π/M)
             *   phase_offset += α × error
             */
            float ideal = (float)sym * two_pi / (float)M;
            float err   = atan2f(sinf(corrected - ideal),
                                  cosf(corrected - ideal));
            s->phase_offset  += PSK_PLL_ALPHA * err;
            s->pll_error_deg  = err * 180.0f / (float)M_PI;

            return sym;
        }

    } else {
        /* Carrier dropped: abort mid-preamble calibration so the
         * next transmission's preamble is captured cleanly.
         * DECODING is intentionally preserved across brief gaps. */
        if (s->rx_state == 1) {
            s->rx_state  = 0;
            s->cal_count = 0;
        }
        s->is_boundary = 0;
    }

    return -1;   /* no symbol decision this window */
}

/* ═══════════════════════════════════════════════════════════════════
 * 10. Hamming(7,4) Forward Error Correction
 *
 * Encodes 4 data bits (a nibble) into a 7-bit codeword with 3
 * parity bits.  Any single-bit error is correctable via a 3-bit
 * syndrome that encodes the error position (1-indexed) in binary.
 *
 * Bit layout in the uint8_t codeword (bit 6 = MSB, bit 0 = LSB):
 *   [6]=d4  [5]=d3  [4]=d2  [3]=p3  [2]=d1  [1]=p2  [0]=p1
 *
 * Parity equations (positions 1,2,4 are parity; 3,5,6,7 are data):
 *   p1 = d1^d2^d4   (positions 1,3,5,7)
 *   p2 = d1^d3^d4   (positions 2,3,6,7)
 *   p3 = d2^d3^d4   (positions 4,5,6,7)
 *
 * Rate: 4/7 ≈ 57%.  Paired with Gray coding and block interleaving,
 * a burst of ≤ depth consecutive bit errors (one per codeword) is
 * fully correctable.
 * ═══════════════════════════════════════════════════════════════════ */

#define PSK_HAMMING_K           4    /* data bits per codeword              */
#define PSK_HAMMING_N           7    /* total bits per codeword             */
#define PSK_DEFAULT_ILVE_DEPTH  8    /* default interleave depth (codewords)*/

/* ── Bit-stream helpers (MSB-first within each byte) ────────────── */
static inline int psk_getbit(const uint8_t *a, size_t i)
{
    return (a[i / 8] >> (7 - (int)(i % 8))) & 1;
}
static inline void psk_setbit(uint8_t *a, size_t i, int bit)
{
    int shift = 7 - (int)(i % 8);
    if (bit) a[i / 8] |=  (uint8_t)(1u << shift);
    else     a[i / 8] &= (uint8_t)~(1u << shift);
}

/*
 * psk_hamming_encode
 * nibble: bits [3:0] = d4 d3 d2 d1.  Returns 7-bit codeword.
 */
static inline uint8_t psk_hamming_encode(uint8_t nibble)
{
    int d1 = (nibble >> 0) & 1;
    int d2 = (nibble >> 1) & 1;
    int d3 = (nibble >> 2) & 1;
    int d4 = (nibble >> 3) & 1;
    int p1 = d1 ^ d2 ^ d4;
    int p2 = d1 ^ d3 ^ d4;
    int p3 = d2 ^ d3 ^ d4;
    return (uint8_t)((d4<<6)|(d3<<5)|(d2<<4)|(p3<<3)|(d1<<2)|(p2<<1)|p1);
}

/*
 * psk_hamming_decode
 * Corrects up to 1 bit in codeword and writes data nibble to *nibble_out.
 * Returns 0 = no error, 1 = corrected.
 */
static inline int psk_hamming_decode(uint8_t codeword, uint8_t *nibble_out)
{
    int r1 = (codeword >> 0) & 1;
    int r2 = (codeword >> 1) & 1;
    int r3 = (codeword >> 2) & 1;
    int r4 = (codeword >> 3) & 1;
    int r5 = (codeword >> 4) & 1;
    int r6 = (codeword >> 5) & 1;
    int r7 = (codeword >> 6) & 1;

    int s1  = r1 ^ r3 ^ r5 ^ r7;
    int s2  = r2 ^ r3 ^ r6 ^ r7;
    int s3  = r4 ^ r5 ^ r6 ^ r7;
    int syn = s1 | (s2 << 1) | (s3 << 2);

    int status = 0;
    if (syn) {
        codeword ^= (uint8_t)(1u << (syn - 1));
        r3 = (codeword >> 2) & 1;
        r5 = (codeword >> 4) & 1;
        r6 = (codeword >> 5) & 1;
        r7 = (codeword >> 6) & 1;
        status = 1;
    }
    *nibble_out = (uint8_t)((r7<<3)|(r6<<2)|(r5<<1)|r3);
    return status;
}

/* ═══════════════════════════════════════════════════════════════════
 * 11. Full FEC encode / decode (Hamming + block interleaving)
 *
 * Encode pipeline:
 *   bytes → bits → nibbles → Hamming(7,4) codewords
 *        → block-interleave → pack to bytes
 *
 * Decode pipeline (inverse):
 *   bytes → bits → block-deinterleave → 7-bit groups
 *        → Hamming decode → nibbles → bytes
 *
 * Block interleaver (depth D codewords per block):
 *   Writes a D-row × 7-column matrix row-by-row (codeword bits across
 *   columns) and reads it column-by-column for transmission.
 *   A burst of ≤ D consecutive corrupted transmitted bits hits at most
 *   1 bit per codeword, which Hamming corrects independently.
 *
 *   interleave:   output[j] = input[(j % D) × W  +  j / D]  where W = n/D
 *   deinterleave: output[(i % W) × D  +  i / W] = input[i]
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * psk_fec_encode
 *   data / data_len  — raw input bytes
 *   depth            — interleave depth in codewords (≥ 1)
 *   out_len          — set to encoded byte count
 * Returns calloc-allocated buffer; caller must free().
 * Input is zero-padded to a multiple of 4 bits then to a multiple of
 * depth codewords for clean block alignment.
 */
static inline uint8_t *psk_fec_encode(const uint8_t *data, size_t data_len,
                                       int depth, size_t *out_len)
{
    *out_len = 0;
    if (!data || data_len == 0) return (uint8_t *)calloc(1, 1);

    size_t D         = (depth > 0) ? (size_t)depth : 1;
    size_t n_nib     = (data_len * 8 + 3) / 4;
    size_t rem       = n_nib % D;
    if (rem) n_nib  += D - rem;                    /* pad to multiple of D */

    size_t n_enc_bits  = n_nib * (size_t)PSK_HAMMING_N;
    size_t n_out_bytes = (n_enc_bits + 7) / 8;
    uint8_t *enc = (uint8_t *)calloc(n_out_bytes, 1);
    if (!enc) return NULL;

    /* Hamming encode each nibble */
    size_t n_in_bits = data_len * 8;
    for (size_t cw = 0; cw < n_nib; cw++) {
        uint8_t nibble = 0;
        for (int b = 0; b < PSK_HAMMING_K; b++) {
            size_t idx = cw * (size_t)PSK_HAMMING_K + (size_t)b;
            int    bit = (idx < n_in_bits) ? psk_getbit(data, idx) : 0;
            nibble |= (uint8_t)(bit << b);
        }
        uint8_t cw_byte = psk_hamming_encode(nibble);
        for (int b = 0; b < PSK_HAMMING_N; b++)
            psk_setbit(enc, cw * (size_t)PSK_HAMMING_N + (size_t)b,
                       (cw_byte >> b) & 1);
    }

    /* Block interleave: transpose D-row × 7-col blocks */
    if (D > 1) {
        size_t block_bits = D * (size_t)PSK_HAMMING_N;
        size_t n_blocks   = n_nib / D;
        uint8_t *tmp = (uint8_t *)calloc(n_out_bytes, 1);
        if (tmp) {
            for (size_t blk = 0; blk < n_blocks; blk++) {
                size_t base = blk * block_bits;
                for (size_t row = 0; row < D; row++)
                    for (int col = 0; col < PSK_HAMMING_N; col++)
                        psk_setbit(tmp,
                            base + (size_t)col * D + row,
                            psk_getbit(enc,
                                base + row * (size_t)PSK_HAMMING_N
                                     + (size_t)col));
            }
            memcpy(enc, tmp, n_out_bytes);
            free(tmp);
        }
    }

    *out_len = n_out_bytes;
    return enc;
}

/*
 * psk_fec_decode
 *   encoded / enc_len  — FEC-encoded bytes
 *   depth              — must match encoder
 *   out_len            — set to decoded byte count
 *   corrections_out    — if non-NULL, set to number of bits corrected
 * Returns calloc-allocated buffer; caller must free().
 */
static inline uint8_t *psk_fec_decode(const uint8_t *encoded, size_t enc_len,
                                       int depth, size_t *out_len,
                                       int *corrections_out)
{
    *out_len = 0;
    if (corrections_out) *corrections_out = 0;
    if (!encoded || enc_len == 0) return (uint8_t *)calloc(1, 1);

    size_t D           = (depth > 0) ? (size_t)depth : 1;
    size_t n_codewords = (enc_len * 8) / (size_t)PSK_HAMMING_N;
    if (!n_codewords) return (uint8_t *)calloc(1, 1);

    size_t n_enc_bits  = n_codewords * (size_t)PSK_HAMMING_N;
    size_t n_enc_bytes = (n_enc_bits + 7) / 8;
    uint8_t *work = (uint8_t *)calloc(n_enc_bytes, 1);
    if (!work) return NULL;
    memcpy(work, encoded, enc_len < n_enc_bytes ? enc_len : n_enc_bytes);

    /* Block deinterleave (inverse transpose) */
    if (D > 1 && n_codewords >= D) {
        size_t block_bits = D * (size_t)PSK_HAMMING_N;
        size_t n_blocks   = n_codewords / D;
        uint8_t *tmp = (uint8_t *)calloc(n_enc_bytes, 1);
        if (tmp) {
            for (size_t blk = 0; blk < n_blocks; blk++) {
                size_t base = blk * block_bits;
                for (int col = 0; col < PSK_HAMMING_N; col++)
                    for (size_t row = 0; row < D; row++)
                        psk_setbit(tmp,
                            base + row * (size_t)PSK_HAMMING_N + (size_t)col,
                            psk_getbit(work,
                                base + (size_t)col * D + row));
            }
            memcpy(work, tmp, n_enc_bytes);
            free(tmp);
        }
    }

    /* Hamming decode */
    size_t n_out_bits  = n_codewords * (size_t)PSK_HAMMING_K;
    size_t n_out_bytes = (n_out_bits + 7) / 8;
    uint8_t *out = (uint8_t *)calloc(n_out_bytes, 1);
    if (!out) { free(work); return NULL; }

    int corrections = 0;
    for (size_t cw = 0; cw < n_codewords; cw++) {
        uint8_t cw_byte = 0;
        for (int b = 0; b < PSK_HAMMING_N; b++)
            if (psk_getbit(work, cw * (size_t)PSK_HAMMING_N + (size_t)b))
                cw_byte |= (uint8_t)(1u << b);

        uint8_t nibble;
        corrections += psk_hamming_decode(cw_byte, &nibble);

        for (int b = 0; b < PSK_HAMMING_K; b++) {
            size_t oi = cw * (size_t)PSK_HAMMING_K + (size_t)b;
            if (oi < n_out_bits)
                psk_setbit(out, oi, (nibble >> b) & 1);
        }
    }

    free(work);
    if (corrections_out) *corrections_out = corrections;
    *out_len = n_out_bytes;
    return out;
}

/* ═══════════════════════════════════════════════════════════════════
 * 12. Channel-quality estimation: SNR, interference, AGC
 *
 * SNR estimate
 * ────────────
 *   SNR_dB = 10·log10( carrier_power / mean_noise_power )
 *
 *   carrier_power = magnitudes[carrier_bin]²
 *   noise_floor   = mean of magnitudes[k]² for |k−carrier_bin| > GUARD
 *
 * Interference detection
 * ──────────────────────
 *   A non-carrier bin k is flagged when
 *   magnitudes[k] > PSK_INTF_RATIO × magnitudes[carrier_bin].
 *   Signals with a high carrier-to-interference ratio may corrupt the
 *   DFT phase and bypass the PLL's ability to track the true carrier.
 *
 * AGC (Automatic Gain Control)
 * ────────────────────────────
 *   agc_mag = α·|ft| + (1−α)·agc_mag   (updated in psk_rx_process_window)
 *   Provides a slow-moving estimate of the received signal level that
 *   is independent of symbol timing.  The receiver can use this to
 *   normalise the IQ diagram and to detect signal presence without
 *   relying solely on the instantaneous |ft|.
 * ═══════════════════════════════════════════════════════════════════ */

#define PSK_SNR_GUARD    2       /* bins excluded around carrier for noise   */
#define PSK_INTF_RATIO   0.5f   /* interferer threshold: > 50% of carrier   */

/*
 * psk_snr_estimate_db
 * Returns estimated SNR in dB from the BAR_WIDTH-bin magnitude array.
 * Returns -60.0 when the carrier is below the noise floor.
 */
static inline float psk_snr_estimate_db(const float *magnitudes,
                                         int n_bins, int carrier_bin)
{
    float sig2      = magnitudes[carrier_bin] * magnitudes[carrier_bin];
    float noise_sum = 0.0f;
    int   noise_cnt = 0;

    for (int k = 0; k < n_bins; k++) {
        int dist = k - carrier_bin;
        if (dist < 0) dist = -dist;
        if (dist <= PSK_SNR_GUARD) continue;
        noise_sum += magnitudes[k] * magnitudes[k];
        noise_cnt++;
    }

    if (sig2 < 1e-20f) return -60.0f;
    if (noise_cnt == 0 || noise_sum < 1e-20f) return  60.0f;
    float noise_avg = noise_sum / (float)noise_cnt;
    return 10.0f * log10f(sig2 / noise_avg);
}

/*
 * psk_detect_interference
 * Returns 1 if any bin outside the guard zone exceeds
 * PSK_INTF_RATIO × carrier magnitude.
 * Writes the strongest interferer's bin index to *intf_bin_out
 * (left unchanged when no interference found).
 */
static inline int psk_detect_interference(const float *magnitudes,
                                           int n_bins, int carrier_bin,
                                           int *intf_bin_out)
{
    float threshold = PSK_INTF_RATIO * magnitudes[carrier_bin];
    float max_mag   = 0.0f;
    int   max_bin   = -1;

    for (int k = 0; k < n_bins; k++) {
        int dist = k - carrier_bin;
        if (dist < 0) dist = -dist;
        if (dist <= PSK_SNR_GUARD) continue;
        if (magnitudes[k] > threshold && magnitudes[k] > max_mag) {
            max_mag = magnitudes[k];
            max_bin = k;
        }
    }

    if (max_bin >= 0) {
        if (intf_bin_out) *intf_bin_out = max_bin;
        return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * 13. File I/O helpers
 *
 * These three functions provide a uniform interface for reading and
 * writing binary data to/from files or streams, shared by transmit.c,
 * receive.c, and test.c.
 *
 * psk_read_stream  — drain an open FILE* into a heap buffer.
 * psk_read_file    — open a path, drain it, close it.
 * psk_write_file   — open a path for writing, write all bytes, close.
 *
 * All returned buffers must be freed by the caller.
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * psk_read_stream
 *
 * Read all remaining bytes from the open stream `fp` into a
 * calloc-allocated buffer.  Sets *out_len to the number of bytes read.
 * Returns NULL on allocation failure (errno is set by malloc/realloc).
 */
static inline uint8_t *psk_read_stream(FILE *fp, size_t *out_len)
{
    size_t   cap = 4096, len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) { *out_len = 0; return NULL; }

    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (len == cap) {
            uint8_t *tmp = (uint8_t *)realloc(buf, cap *= 2);
            if (!tmp) { free(buf); *out_len = 0; return NULL; }
            buf = tmp;
        }
        buf[len++] = (uint8_t)c;
    }
    *out_len = len;
    return buf;
}

/*
 * psk_read_file
 *
 * Open the file at `path` in binary read mode, read its entire
 * contents into a heap buffer, close the file, and return the buffer.
 * Sets *out_len to the number of bytes read.
 * Returns NULL and prints an error to stderr on failure.
 */
static inline uint8_t *psk_read_file(const char *path, size_t *out_len)
{
    *out_len = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' for reading.\n", path);
        return NULL;
    }
    uint8_t *buf = psk_read_stream(fp, out_len);
    fclose(fp);
    if (!buf)
        fprintf(stderr, "Error: allocation failed reading '%s'.\n", path);
    return buf;
}

/*
 * psk_write_file
 *
 * Write `len` bytes from `data` to the file at `path` (binary mode,
 * creates or truncates).  Returns 0 on success, -1 on failure (error
 * printed to stderr).
 */
static inline int psk_write_file(const char *path,
                                  const uint8_t *data, size_t len)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' for writing.\n", path);
        return -1;
    }
    size_t written = fwrite(data, 1, len, fp);
    fclose(fp);
    if (written != len) {
        fprintf(stderr, "Error: wrote %zu of %zu bytes to '%s'.\n",
                written, len, path);
        return -1;
    }
    return 0;
}

#endif /* PSK_COMMON_H */