#include <portaudio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLE_RATE 48000
#define FRAMES_PER_BUFFER 256

typedef struct {
    float phase;
} CallbackData;

float sintable[FRAMES_PER_BUFFER];

// This callback is called by PortAudio whenever it needs more audio
static int audioCallback(
    const void *inputBuffer, void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    // CallbackData *data = (CallbackData *)userData;
    // float *out = (float *)outputBuffer;

    memcpy(outputBuffer, sintable, FRAMES_PER_BUFFER * sizeof(float));
    return paContinue;
}

int main(void) {
    for (size_t i = 0; i < FRAMES_PER_BUFFER; i++) {
        sintable[i] = sinf((i * 8.0f * M_PI) / FRAMES_PER_BUFFER);
    }

    CallbackData data = {0};
    PaStream *stream;

    Pa_Initialize();
    Pa_OpenDefaultStream(&stream,
        0, 1,               // no input, 1 output channel
        paFloat32,          // 32-bit float samples
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        audioCallback, &data);

    Pa_StartStream(stream);
    Pa_Sleep(3000);         // play for 3 seconds
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    return 0;
}
