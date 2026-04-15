#include <math.h>
#include <portaudio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLE_RATE 48000
#define FRAMES_PER_BUFFER 256
#define BUFFERS_PER_SYMBOL 16

typedef struct {
    size_t callback_count;
    size_t symbol_count;
    size_t num_symbols;
} callback_data_t;

float sintable[FRAMES_PER_BUFFER];
float sintable2[FRAMES_PER_BUFFER];
float *audio_bufs[2] = {sintable, sintable2};

uint8_t *symbol_buf;

// This callback is called by PortAudio whenever it needs more audio
static int audio_callback(const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo *timeInfo,
                          PaStreamCallbackFlags statusFlags, void *userData) {
    callback_data_t *data = (callback_data_t *)userData;
    memcpy(outputBuffer, audio_bufs[symbol_buf[data->symbol_count]],
           FRAMES_PER_BUFFER * sizeof(float));
    data->callback_count++;
    if (data->callback_count == BUFFERS_PER_SYMBOL) {
        data->callback_count = 0;
        data->symbol_count++;
    }
    if (data->symbol_count >= data->num_symbols) {
        return paComplete;
    }
    return paContinue;
}

uint8_t *read_stdin(size_t *out_len) {
    size_t capacity = 4096;
    size_t length = 0;
    uint8_t *buf = malloc(capacity);

    if (!buf) {
        return NULL;
    }

    int c;
    while ((c = getchar()) != EOF) {
        if (length == capacity) {
            capacity *= 2;
            uint8_t *tmp = realloc(buf, capacity);
            if (!tmp) {
                free(buf);
                return NULL;
            }
            buf = tmp;
        }
        buf[length++] = (uint8_t)c;
    }

    *out_len = length;
    return buf; // caller must free()
}

uint8_t *unpack_bits(const uint8_t *const source_buf, const size_t length) {
    uint8_t *unpacked = malloc(length << 3);
    if (unpacked == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < length; i++) {
        for (int bit_i = 0; bit_i < 8; bit_i++) {
            unpacked[(i << 3) + bit_i] = (source_buf[i] >> (7 - bit_i)) & 1;
        }
    }
    return unpacked;
}

int main(void) {
    size_t buf_length;
    uint8_t *buf = read_stdin(&buf_length);
    if (buf == NULL) {
        fprintf(stderr, "Failed to read from stdin.\n");
        return 1;
    }
    if (buf_length == 0) {
        fprintf(stderr, "No input from stdin.\n");
        free(buf);
        return 1;
    }
    symbol_buf = unpack_bits(buf, buf_length);
    free(buf);
    if (symbol_buf == NULL) {
        fprintf(stderr, "Failed to unpack bits.\n");
        return 1;
    }
    size_t num_symbols = buf_length << 3;

    for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
        sintable[i] = sinf((i * 8.0f * M_PI) / FRAMES_PER_BUFFER);
        sintable2[i] = sinf((i * 12.0f * M_PI) / FRAMES_PER_BUFFER);
    }

    PaError err;
    PaStream *stream;
    callback_data_t user_data = {0, 0, num_symbols};

    err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "PortAudio error: %s\n", Pa_GetErrorText(err));
        return 1;
    }
    err = Pa_OpenDefaultStream(&stream, 0, 1, // no input, 1 output channel
                               paFloat32,     // 32-bit float samples
                               SAMPLE_RATE, FRAMES_PER_BUFFER, audio_callback,
                               &user_data);
    if (err != paNoError) {
        fprintf(stderr, "PortAudio error: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "PortAudio error: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    // Wait for the stream to finish
    while (Pa_IsStreamActive(stream)) {
        Pa_Sleep(100); // Sleep for 100ms to avoid busy-waiting
    }

    // Clean up
    Pa_CloseStream(stream);
    Pa_Terminate();

    free(symbol_buf);

    printf("Program finished.\n");
    return 0;
}
