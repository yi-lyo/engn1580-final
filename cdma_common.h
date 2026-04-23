/*
 * cdma_common.h — CDMA (Code Division Multiple Access) spreading codes
 *
 * This header provides spread-spectrum functionality using Walsh-Hadamard
 * codes to improve noise resilience of the PSK modulation system.
 *
 * CDMA Overview
 * ─────────────
 * Each data symbol is spread across multiple "chips" using an orthogonal
 * spreading code. The spreading factor (SF) provides processing gain:
 *
 *   Processing Gain (dB) = 10·log₁₀(SF)
 *
 * For SF=16: ~12 dB processing gain, meaning the system can tolerate
 * 12 dB more noise than the non-spread version.
 *
 * Walsh-Hadamard Codes
 * ────────────────────
 * These are perfectly orthogonal binary codes (+1/-1 valued). For SF=N,
 * we can generate N orthogonal codes, each of length N.
 *
 * Integration with PSK
 * ────────────────────
 * Without CDMA: 1 symbol → 1 PSK transmission (e.g., 1 QPSK symbol = 2 bits)
 * With CDMA:    1 symbol → SF PSK transmissions (chips)
 *               Each chip is ±1, transmitted as 0° or 180° PSK phase
 *
 * Receiver uses correlation: for each received chip sequence, correlate
 * with all possible codes and pick the one with maximum correlation.
 */

#ifndef CDMA_COMMON_H
#define CDMA_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * CDMA Configuration
 * ═══════════════════════════════════════════════════════════════════ */

/* Spreading factor: number of chips per symbol.
 * Must be a power of 2. Common values: 8, 16, 32
 * Higher = more processing gain but lower data rate */
#define CDMA_SPREADING_FACTOR   16

/* Number of orthogonal codes available (equals spreading factor) */
#define CDMA_NUM_CODES          CDMA_SPREADING_FACTOR

/* Each code symbol can be one of CDMA_NUM_CODES values (0 to NUM_CODES-1) */
#define CDMA_BITS_PER_SYMBOL    4   /* log2(16) = 4 bits per CDMA symbol */

/* Correlation threshold as fraction of perfect correlation.
 * Perfect correlation = SF. We require at least 70% for detection. */
#define CDMA_CORRELATION_THRESHOLD  (0.70f * CDMA_SPREADING_FACTOR)

/* ═══════════════════════════════════════════════════════════════════
 * Walsh-Hadamard Code Generation
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Walsh-Hadamard matrix is generated recursively:
 *
 *   H₁ = [+1]
 *
 *   H₂ = [+1 +1]    H₄ = [+1 +1 +1 +1]
 *        [+1 -1]         [+1 -1 +1 -1]
 *                        [+1 +1 -1 -1]
 *                        [+1 -1 -1 +1]
 *
 *   Hₙ = [H_{n/2}   H_{n/2}]
 *        [H_{n/2}  -H_{n/2}]
 *
 * Each row is an orthogonal code of length N.
 */

/*
 * cdma_generate_walsh_code
 *   Generates the code_index-th Walsh-Hadamard code of length SF.
 *   Output buffer 'code' must be allocated with SF elements.
 *   Codes are represented as +1/-1 (int8_t).
 *
 *   code_index: 0 to SF-1
 *   SF:         spreading factor (must be power of 2)
 */
static inline void cdma_generate_walsh_code(int8_t *code, int code_index, int SF)
{
    /* Initialize with all +1 */
    for (int i = 0; i < SF; i++) {
        code[i] = 1;
    }
    
    /* Hadamard construction: for each bit position in code_index */
    int size = 1;
    while (size < SF) {
        if (code_index & size) {
            /* Flip the appropriate half */
            for (int i = 0; i < SF; i++) {
                if (i & size) {
                    code[i] = -code[i];
                }
            }
        }
        size <<= 1;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Spreading and Despreading
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * cdma_spread_symbol
 *   Spreads a data symbol (0 to NUM_CODES-1) into chips.
 *   The symbol index selects which Walsh code to use.
 *   
 *   symbol:     Input symbol value (0 to CDMA_NUM_CODES-1)
 *   chips_out:  Output buffer for chips (must be SF elements)
 *   SF:         Spreading factor
 *
 *   Output chips are +1/-1 (int8_t)
 */
static inline void cdma_spread_symbol(int symbol, int8_t *chips_out, int SF)
{
    if (symbol < 0 || symbol >= SF) {
        /* Invalid symbol, use all-zeros code */
        for (int i = 0; i < SF; i++) {
            chips_out[i] = 1;
        }
        return;
    }
    
    cdma_generate_walsh_code(chips_out, symbol, SF);
}

/*
 * cdma_correlate
 *   Correlates received chips with a specific Walsh code.
 *   Returns correlation value (sum of chip * code[i]).
 *   Perfect match = +SF, complete mismatch = -SF.
 *
 *   chips:      Received chips (+1/-1, may have errors)
 *   code_index: Which Walsh code to correlate with
 *   SF:         Spreading factor
 */
static inline int cdma_correlate(const int8_t *chips, int code_index, int SF)
{
    int8_t code[64];  /* Max supported SF = 64 */
    if (SF > 64) return 0;
    
    cdma_generate_walsh_code(code, code_index, SF);
    
    int correlation = 0;
    for (int i = 0; i < SF; i++) {
        correlation += chips[i] * code[i];
    }
    
    return correlation;
}

/*
 * cdma_despread_chips
 *   Despreads received chips by correlating with all possible codes.
 *   Returns the code index (symbol) with maximum correlation.
 *   Returns -1 if no correlation exceeds threshold.
 *
 *   chips:         Received chips (+1/-1)
 *   SF:            Spreading factor
 *   correlation_out: Optional pointer to store best correlation value
 */
static inline int cdma_despread_chips(const int8_t *chips, int SF, int *correlation_out)
{
    int best_symbol = -1;
    int best_corr = (int)(CDMA_CORRELATION_THRESHOLD) - 1;
    
    for (int sym = 0; sym < SF; sym++) {
        int corr = cdma_correlate(chips, sym, SF);
        
        if (corr > best_corr) {
            best_corr = corr;
            best_symbol = sym;
        }
    }
    
    if (correlation_out) {
        *correlation_out = best_corr;
    }
    
    return best_symbol;
}

/* ═══════════════════════════════════════════════════════════════════
 * Soft Decision Support
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * cdma_correlate_soft
 *   Soft-decision correlation using floating-point chip values.
 *   Chips can be in range [-1.0, +1.0] representing confidence.
 *   
 *   soft_chips:  Soft chip values (float)
 *   code_index:  Which Walsh code to correlate with
 *   SF:          Spreading factor
 *
 *   Returns correlation value (sum of soft_chip[i] * code[i])
 */
static inline float cdma_correlate_soft(const float *soft_chips, int code_index, int SF)
{
    int8_t code[64];
    if (SF > 64) return 0.0f;
    
    cdma_generate_walsh_code(code, code_index, SF);
    
    float correlation = 0.0f;
    for (int i = 0; i < SF; i++) {
        correlation += soft_chips[i] * (float)code[i];
    }
    
    return correlation;
}

/*
 * cdma_despread_soft
 *   Soft-decision despreading using floating-point chips.
 *   Returns the symbol with maximum correlation.
 *   
 *   soft_chips:      Soft chip values (float, -1.0 to +1.0)
 *   SF:              Spreading factor
 *   correlation_out: Optional pointer to store best correlation value
 */
static inline int cdma_despread_soft(const float *soft_chips, int SF, float *correlation_out)
{
    int best_symbol = -1;
    float best_corr = CDMA_CORRELATION_THRESHOLD - 1.0f;
    
    for (int sym = 0; sym < SF; sym++) {
        float corr = cdma_correlate_soft(soft_chips, sym, SF);
        
        if (corr > best_corr) {
            best_corr = corr;
            best_symbol = sym;
        }
    }
    
    if (correlation_out) {
        *correlation_out = best_corr;
    }
    
    return best_symbol;
}

/* ═══════════════════════════════════════════════════════════════════
 * Data Conversion Utilities
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * cdma_bytes_to_symbols
 *   Converts byte data to CDMA symbol indices.
 *   Each symbol carries log2(SF) bits.
 *   
 *   data:      Input byte array
 *   data_len:  Number of input bytes
 *   n_syms:    Output: number of symbols generated
 *
 *   Returns malloc'd array of symbol indices; caller must free.
 */
static inline uint8_t *cdma_bytes_to_symbols(const uint8_t *data, size_t data_len, size_t *n_syms)
{
    int bits_per_sym = CDMA_BITS_PER_SYMBOL;
    size_t total_bits = data_len * 8;
    size_t n_symbols = (total_bits + bits_per_sym - 1) / bits_per_sym;
    
    uint8_t *symbols = (uint8_t *)calloc(n_symbols, sizeof(uint8_t));
    if (!symbols) return NULL;
    
    for (size_t s = 0; s < n_symbols; s++) {
        int sym_val = 0;
        
        for (int b = 0; b < bits_per_sym; b++) {
            size_t bit_idx = s * bits_per_sym + b;
            if (bit_idx >= total_bits) break;
            
            size_t byte_i = bit_idx / 8;
            int bit_i = 7 - (bit_idx % 8);  /* MSB first */
            int bit = (data[byte_i] >> bit_i) & 1;
            
            sym_val |= (bit << (bits_per_sym - 1 - b));
        }
        
        symbols[s] = (uint8_t)sym_val;
    }
    
    *n_syms = n_symbols;
    return symbols;
}

/*
 * cdma_symbols_to_bytes
 *   Converts CDMA symbols back to bytes.
 *   
 *   symbols:   Array of symbol indices
 *   n_syms:    Number of symbols
 *   n_bytes:   Output: number of bytes generated
 *
 *   Returns malloc'd byte array; caller must free.
 */
static inline uint8_t *cdma_symbols_to_bytes(const uint8_t *symbols, size_t n_syms, size_t *n_bytes)
{
    int bits_per_sym = CDMA_BITS_PER_SYMBOL;
    size_t total_bits = n_syms * bits_per_sym;
    size_t n_out_bytes = (total_bits + 7) / 8;
    
    uint8_t *bytes = (uint8_t *)calloc(n_out_bytes, sizeof(uint8_t));
    if (!bytes) return NULL;
    
    for (size_t s = 0; s < n_syms; s++) {
        int sym_val = symbols[s];
        
        for (int b = 0; b < bits_per_sym; b++) {
            size_t bit_idx = s * bits_per_sym + b;
            if (bit_idx >= total_bits) break;
            
            size_t byte_i = bit_idx / 8;
            int bit_i = 7 - (bit_idx % 8);  /* MSB first */
            int bit = (sym_val >> (bits_per_sym - 1 - b)) & 1;
            
            if (bit) {
                bytes[byte_i] |= (1 << bit_i);
            }
        }
    }
    
    *n_bytes = n_out_bytes;
    return bytes;
}

/* ═══════════════════════════════════════════════════════════════════
 * PSK Integration Helpers
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * cdma_chip_to_psk_symbol
 *   Converts a CDMA chip (+1/-1) to a BPSK symbol index.
 *   +1 → symbol 0 (phase 0°)
 *   -1 → symbol 1 (phase 180°)
 */
static inline int cdma_chip_to_psk_symbol(int8_t chip)
{
    return (chip < 0) ? 1 : 0;
}

/*
 * cdma_psk_symbol_to_chip
 *   Converts a BPSK symbol back to a chip.
 *   symbol 0 (phase ~0°)   → +1
 *   symbol 1 (phase ~180°) → -1
 */
static inline int8_t cdma_psk_symbol_to_chip(int psk_symbol)
{
    return (psk_symbol == 1) ? -1 : 1;
}

/*
 * cdma_psk_phase_to_chip_soft
 *   Converts PSK phase to soft chip value.
 *   Phase near 0° → +1.0
 *   Phase near 180° → -1.0
 *   
 *   corrected_phase: Phase in radians [0, 2π)
 *   
 *   Returns soft chip value in [-1.0, +1.0]
 */
static inline float cdma_psk_phase_to_chip_soft(float corrected_phase)
{
    /* Normalize to [-π, π] */
    float phi = corrected_phase;
    if (phi > M_PI) phi -= 2.0f * (float)M_PI;
    
    /* Map phase to soft chip:
     * 0° → +1.0, ±180° → -1.0
     * Use cosine for smooth transition */
    return cosf(phi);
}

/* ═══════════════════════════════════════════════════════════════════
 * Statistics and Diagnostics
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * cdma_calculate_processing_gain_db
 *   Returns the theoretical processing gain in dB.
 */
static inline float cdma_calculate_processing_gain_db(int SF)
{
    return 10.0f * log10f((float)SF);
}

/*
 * cdma_estimate_chip_error_rate
 *   Estimates chip error rate from correlation value.
 *   Perfect correlation = SF (no errors)
 *   Zero correlation = SF/2 (50% chip errors)
 *   
 *   correlation: Correlation value from despread
 *   SF:          Spreading factor
 *
 *   Returns estimated chip error rate (0.0 to 1.0)
 */
static inline float cdma_estimate_chip_error_rate(int correlation, int SF)
{
    /* correlation = SF - 2*num_errors
     * num_errors = (SF - correlation) / 2 */
    float error_rate = (float)(SF - correlation) / (2.0f * (float)SF);
    
    /* Clamp to [0, 1] */
    if (error_rate < 0.0f) error_rate = 0.0f;
    if (error_rate > 1.0f) error_rate = 1.0f;
    
    return error_rate;
}

#endif /* CDMA_COMMON_H */