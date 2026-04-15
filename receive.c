#include <complex.h>
#include <math.h>
#include <portaudio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLE_RATE 48000
#define FRAMES_PER_BUFFER 0x1000

#define FREQ1_CYCLES_PER_BUFFER 0x40
#define FREQ2_CYCLES_PER_BUFFER 0x60

// Precomputed lookup tables of e^(-i 2 pi f t)
float complex exptable1[FRAMES_PER_BUFFER];
float complex exptable2[FRAMES_PER_BUFFER];

typedef struct {
    size_t callback_count;
    size_t max_callback_count;
} callback_data_t;

static int audio_callback(const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo *timeInfo,
                          PaStreamCallbackFlags statusFlags, void *userData) {
    float complex ft1 = 0, ft2 = 0;
    const float *const micInput = (float *)inputBuffer;
    for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
        ft1 += micInput[i] * exptable1[i];
        ft2 += micInput[i] * exptable2[i];
    }
    float ft1_mag = cabsf(ft1), ft2_mag = cabsf(ft2);
    printf("%f %f\n", ft1_mag, ft2_mag);
    callback_data_t *user_data = (callback_data_t *)userData;
    user_data->callback_count++;
    if (user_data->callback_count >= user_data->max_callback_count) {
        return paComplete;
    }
    return paContinue;
}

int main(void) {
    for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
        exptable1[i] = cexpf(-2.0f * I * M_PI * FREQ1_CYCLES_PER_BUFFER /
                             FRAMES_PER_BUFFER * i);
        exptable2[i] = cexpf(-2.0f * I * M_PI * FREQ2_CYCLES_PER_BUFFER /
                             FRAMES_PER_BUFFER * i);
    }

    PaError err;
    PaStream *stream;
    callback_data_t user_data = {0, 256};

    err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "PortAudio error: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    err = Pa_OpenDefaultStream(&stream, 1, 0, // 1 input channel, no output
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

    while (Pa_IsStreamActive(stream)) {
        Pa_Sleep(100);
    }

    Pa_CloseStream(stream);
    Pa_Terminate();

    return 0;
}
