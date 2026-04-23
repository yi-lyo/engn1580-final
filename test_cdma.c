/*
 * test_cdma.c — Unit tests for CDMA spreading/despreading
 *
 * Tests Walsh-Hadamard code generation, spreading, despreading,
 * correlation, and noise resilience without requiring audio hardware.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -o test_cdma test_cdma.c -lm
 *
 * Run:
 *   ./test_cdma
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "cdma_common.h"

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("TEST: %s ... ", name); \
        fflush(stdout); \
    } while (0)

#define PASS() \
    do { \
        tests_passed++; \
        printf("PASS\n"); \
    } while (0)

#define FAIL(msg) \
    do { \
        printf("FAIL: %s\n", msg); \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════
 * Test 1: Walsh code orthogonality
 * ═══════════════════════════════════════════════════════════════════ */
void test_walsh_orthogonality(void)
{
    TEST("Walsh code orthogonality");
    
    int SF = 16;
    int8_t codes[16][16];
    
    /* Generate all 16 Walsh codes */
    for (int i = 0; i < SF; i++) {
        cdma_generate_walsh_code(codes[i], i, SF);
    }
    
    /* Test orthogonality: dot product of different codes should be 0 */
    int failed = 0;
    for (int i = 0; i < SF && !failed; i++) {
        for (int j = 0; j < SF && !failed; j++) {
            int dot = 0;
            for (int k = 0; k < SF; k++) {
                dot += codes[i][k] * codes[j][k];
            }
            
            if (i == j) {
                /* Autocorrelation should equal SF */
                if (dot != SF) {
                    FAIL("autocorrelation incorrect");
                    failed = 1;
                }
            } else {
                /* Cross-correlation should be 0 */
                if (dot != 0) {
                    FAIL("cross-correlation non-zero");
                    failed = 1;
                }
            }
        }
    }
    
    if (!failed) {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 2: Spread and despread with no errors
 * ═══════════════════════════════════════════════════════════════════ */
void test_spread_despread_perfect(void)
{
    TEST("Spread/despread (no errors)");
    
    int SF = 16;
    int failed = 0;
    
    /* Test all possible symbols */
    for (int sym = 0; sym < SF && !failed; sym++) {
        int8_t chips[16];
        
        /* Spread the symbol */
        cdma_spread_symbol(sym, chips, SF);
        
        /* Despread it back */
        int recovered = cdma_despread_chips(chips, SF, NULL);
        
        if (recovered != sym) {
            FAIL("symbol mismatch");
            failed = 1;
        }
    }
    
    if (!failed) {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 3: Noise tolerance (single chip error)
 * ═══════════════════════════════════════════════════════════════════ */
void test_single_chip_error(void)
{
    TEST("Single chip error tolerance");
    
    int SF = 16;
    int failed = 0;
    
    /* Test symbol 5 with single chip flip at position 3 */
    int8_t chips[16];
    cdma_spread_symbol(5, chips, SF);
    
    /* Flip one chip */
    chips[3] = -chips[3];
    
    /* Should still recover correct symbol (correlation 14 vs 16) */
    int correlation;
    int recovered = cdma_despread_chips(chips, SF, &correlation);
    
    if (recovered != 5) {
        FAIL("failed to recover with 1 error");
        failed = 1;
    } else if (correlation != 14) {  /* SF - 2*errors = 16 - 2*1 = 14 */
        FAIL("incorrect correlation value");
        failed = 1;
    }
    
    if (!failed) {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 4: Multiple chip errors
 * ═══════════════════════════════════════════════════════════════════ */
void test_multiple_chip_errors(void)
{
    TEST("Multiple chip error handling");
    
    int SF = 16;
    int8_t chips[16];
    cdma_spread_symbol(7, chips, SF);
    
    /* Flip 2 chips (more reliable test) */
    chips[2] = -chips[2];
    chips[7] = -chips[7];
    
    int correlation;
    int recovered = cdma_despread_chips(chips, SF, &correlation);
    
    /* Correlation = 16 - 2*2 = 12 */
    if (recovered != 7 || correlation != 12) {
        FAIL("failed with 2 errors");
    } else {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 5: Soft-decision despreading
 * ═══════════════════════════════════════════════════════════════════ */
void test_soft_decision(void)
{
    TEST("Soft-decision despreading");
    
    int SF = 16;
    float soft_chips[16];
    
    /* Create soft chips for symbol 3 with some uncertainty */
    int8_t hard_chips[16];
    cdma_spread_symbol(3, hard_chips, SF);
    
    for (int i = 0; i < SF; i++) {
        /* Add some noise/uncertainty (reduce magnitude to 0.8) */
        soft_chips[i] = (float)hard_chips[i] * 0.8f;
    }
    
    float correlation;
    int recovered = cdma_despread_soft(soft_chips, SF, &correlation);
    
    if (recovered != 3) {
        FAIL("soft decision failed");
    } else if (correlation < 10.0f) {  /* Should be ~12.8 */
        FAIL("soft correlation too low");
    } else {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 6: Bytes to symbols and back
 * ═══════════════════════════════════════════════════════════════════ */
void test_byte_conversion(void)
{
    TEST("Byte <-> symbol conversion");
    
    uint8_t test_data[] = {0xAB, 0xCD, 0xEF, 0x12};
    size_t data_len = 4;
    
    /* Convert to symbols */
    size_t n_syms;
    uint8_t *symbols = cdma_bytes_to_symbols(test_data, data_len, &n_syms);
    
    if (!symbols) {
        FAIL("conversion to symbols failed");
        return;
    }
    
    /* Convert back to bytes */
    size_t n_bytes;
    uint8_t *recovered = cdma_symbols_to_bytes(symbols, n_syms, &n_bytes);
    
    if (!recovered) {
        FAIL("conversion to bytes failed");
        free(symbols);
        return;
    }
    
    /* Compare (note: may have padding) */
    int failed = 0;
    for (size_t i = 0; i < data_len; i++) {
        if (recovered[i] != test_data[i]) {
            FAIL("byte mismatch");
            failed = 1;
            break;
        }
    }
    
    free(symbols);
    free(recovered);
    
    if (!failed) {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 7: Processing gain calculation
 * ═══════════════════════════════════════════════════════════════════ */
void test_processing_gain(void)
{
    TEST("Processing gain calculation");
    
    float gain_db = cdma_calculate_processing_gain_db(16);
    
    /* 10 * log10(16) ≈ 12.04 dB */
    if (fabs(gain_db - 12.04f) > 0.1f) {
        FAIL("incorrect processing gain");
    } else {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 8: Chip error rate estimation
 * ═══════════════════════════════════════════════════════════════════ */
void test_chip_error_rate(void)
{
    TEST("Chip error rate estimation");
    
    int SF = 16;
    
    /* Perfect correlation → 0% error */
    float cer = cdma_estimate_chip_error_rate(16, SF);
    if (fabs(cer - 0.0f) > 0.01f) {
        FAIL("perfect correlation error rate wrong");
        return;
    }
    
    /* 2 chip errors → correlation 12 → 12.5% error */
    cer = cdma_estimate_chip_error_rate(12, SF);
    if (fabs(cer - 0.125f) > 0.01f) {
        FAIL("2-error rate wrong");
        return;
    }
    
    /* 8 chip errors → correlation 0 → 50% error */
    cer = cdma_estimate_chip_error_rate(0, SF);
    if (fabs(cer - 0.5f) > 0.01f) {
        FAIL("50% error rate wrong");
        return;
    }
    
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 9: PSK integration helpers
 * ═══════════════════════════════════════════════════════════════════ */
void test_psk_integration(void)
{
    TEST("PSK integration helpers");
    
    /* Test chip to PSK symbol conversion */
    if (cdma_chip_to_psk_symbol(+1) != 0) {
        FAIL("+1 chip should be PSK symbol 0");
        return;
    }
    if (cdma_chip_to_psk_symbol(-1) != 1) {
        FAIL("-1 chip should be PSK symbol 1");
        return;
    }
    
    /* Test reverse conversion */
    if (cdma_psk_symbol_to_chip(0) != +1) {
        FAIL("PSK 0 should be +1 chip");
        return;
    }
    if (cdma_psk_symbol_to_chip(1) != -1) {
        FAIL("PSK 1 should be -1 chip");
        return;
    }
    
    /* Test soft phase conversion */
    float soft_chip = cdma_psk_phase_to_chip_soft(0.0f);  /* 0° → +1 */
    if (fabs(soft_chip - 1.0f) > 0.01f) {
        FAIL("0° phase should give +1 soft chip");
        return;
    }
    
    soft_chip = cdma_psk_phase_to_chip_soft((float)M_PI);  /* 180° → -1 */
    if (fabs(soft_chip - (-1.0f)) > 0.01f) {
        FAIL("180° phase should give -1 soft chip");
        return;
    }
    
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * Test 10: End-to-end with simulated transmission
 * ═══════════════════════════════════════════════════════════════════ */
void test_end_to_end(void)
{
    TEST("End-to-end simulation");
    
    const char *message = "CDMA";
    size_t msg_len = strlen(message);
    
    /* Convert to symbols */
    size_t n_syms;
    uint8_t *symbols = cdma_bytes_to_symbols((const uint8_t *)message, 
                                              msg_len, &n_syms);
    if (!symbols) {
        FAIL("symbol conversion failed");
        return;
    }
    
    /* Simulate transmission: spread each symbol and add 1 chip error */
    int SF = 16;
    int8_t *all_chips = (int8_t *)malloc(n_syms * SF);
    if (!all_chips) {
        FAIL("allocation failed");
        free(symbols);
        return;
    }
    
    for (size_t s = 0; s < n_syms; s++) {
        cdma_spread_symbol(symbols[s], &all_chips[s * SF], SF);
        
        /* Add one random chip error per symbol */
        int error_pos = (s * 7) % SF;  /* Deterministic "random" */
        all_chips[s * SF + error_pos] = -all_chips[s * SF + error_pos];
    }
    
    /* Simulate reception: despread each symbol */
    uint8_t *recovered_syms = (uint8_t *)malloc(n_syms);
    if (!recovered_syms) {
        FAIL("allocation failed");
        free(all_chips);
        free(symbols);
        return;
    }
    
    for (size_t s = 0; s < n_syms; s++) {
        recovered_syms[s] = (uint8_t)cdma_despread_chips(&all_chips[s * SF], 
                                                          SF, NULL);
    }
    
    /* Convert back to bytes */
    size_t n_bytes;
    uint8_t *recovered_bytes = cdma_symbols_to_bytes(recovered_syms, n_syms, 
                                                      &n_bytes);
    
    /* Compare */
    int failed = 0;
    for (size_t i = 0; i < msg_len; i++) {
        if (recovered_bytes[i] != (uint8_t)message[i]) {
            FAIL("message corrupted");
            failed = 1;
            break;
        }
    }
    
    free(all_chips);
    free(symbols);
    free(recovered_syms);
    free(recovered_bytes);
    
    if (!failed) {
        PASS();
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Main test runner
 * ═══════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("CDMA Unit Tests\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    test_walsh_orthogonality();
    test_spread_despread_perfect();
    test_single_chip_error();
    test_multiple_chip_errors();
    test_soft_decision();
    test_byte_conversion();
    test_processing_gain();
    test_chip_error_rate();
    test_psk_integration();
    test_end_to_end();
    
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    
    if (tests_passed == tests_run) {
        printf("✓ All tests PASSED\n");
        return 0;
    } else {
        printf("✗ %d tests FAILED\n", tests_run - tests_passed);
        return 1;
    }
}