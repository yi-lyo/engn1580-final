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
    uint8_t *syms       = calloc(n_syms, 1);
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
    uint8_t *bytes      = calloc(n_bytes, 1);
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

#endif /* PSK_COMMON_H */