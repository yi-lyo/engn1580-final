# Digital Receiver Processing Pipeline

## 1. Front End Processing
The front end of the receiver is responsible for ingesting raw audio and extracting baseband parameters (such as I/Q data, magnitude, and phase).

* **Audio Capture & Buffering:** Raw audio samples are continuously captured via PortAudio in block sizes of `FRAMES_PER_BUFFER` (typically 4096 samples). This happens asynchronously within `audio_callback()`.
* **Discrete Fourier Transform (DFT) Projection:** Inside the audio callback, incoming samples are immediately multiplied by complex exponentials corresponding to the expected carrier frequencies (`exptable`). This computes the discrete Fourier transform specifically at the carrier bins.
* **Complex Feature Extraction:** The output of the DFT is stored in shared variables mapping the I (real) and Q (imaginary) components (`s_ft_re` and `s_ft_im`), as well as the magnitude (`s_mag`).
* **Sliding Buffer for History:** Simultaneously, raw audio samples are appended to a circular buffer (`slide_buf`). This preserves the continuous waveform for exact, time-aligned window extraction later on.
* **Spectrum and Channel Analysis:** The main thread performs a wider FFT (`compute_dft` and `compute_display_spectrum`) to render the waterfall/spectrum visualization, estimate the Signal-to-Noise Ratio (SNR), and detect adjacent-channel interference.

## 2. Timing and Carrier Recovery and Synchronization
Because the receiver and transmitter lack a shared clock, the receiver must align itself perfectly with the symbol boundaries and correct for frequency/phase drift.

* **Timing Recovery (Symbol Boundary Detection):**
  * The transmitter sends a symbol whose duration spans a known multiple of receiver windows (e.g., exactly 4 windows per symbol). 
  * Because the carrier completes an exact number of cycles per window, the phase of the carrier is relatively constant *unless* the symbol changes. 
  * The codebase uses a **Phase-Jump Comparator** (`delta = ft_curr * conjf(ft_prev)`). If the phase changes by more than π/M between consecutive windows, the software knows it just processed a window that straddles two symbols.
* **Symbol Alignment & Aligned Extraction:**
  * When a boundary is detected, a counter (`sym_window`) resets. The subsequent windows are deemed "safe" because they fit entirely within one symbol.
  * If `use_aligned_windows` is enabled, the codebase fetches exactly the middle 4096 samples of the symbol out of `slide_buf` and re-runs the DFT to get an impeccably aligned measurement.
* **Carrier Recovery State Machine:** The receiver moves through three states to lock onto the carrier:
  * **WAITING:** Loops until the received signal SNR and magnitude surpass the noise floor threshold (`CARRIER_SNR_MIN_DB` and `SIG_THRESHOLD`).
  * **CALIBRATING:** The preamble transmits a symbol at a constant phase of 0°. The receiver averages the received phase (`cal_cos`, `cal_sin`) over several windows (`CAL_WINDOWS`) to calculate a static `phase_offset`.
  * **DECODING:** Active decoding state. 
* **Decision-Directed Phase-Locked Loop (PLL):** During the `DECODING` state, thermal noise and quantization cause the phase to drift. After making a symbol decision, the system measures the residual error `err` between the received phase and the nearest "ideal" phase. It feeds this error back using a proportional loop filter (`phase_offset += PLL_ALPHA * err`), tracking phase drift in real-time.

## 3. Demodulation and Decision
Once the symbol boundaries are aligned and the reference phase is established, the receiver extracts the logical data from the waveform.

* **Phase Correction:** For PSK modulation, the real-time received phase is corrected by subtracting the calculated `phase_offset` and wrapping the angle to the interval [0, 2π). 
* **M-PSK Symbol Decision:** The corrected phase angle is mapped mathematically to the nearest constellation point to determine the M-PSK symbol: `sym = round(M * corrected / (2 * PI))`. 
* **M-FSK Symbol Decision:** If FSK modulation is used, the system instead correlates the audio window against M different candidate tone frequencies (`fsk_exptable`). It decides on the symbol by picking the frequency that yields the largest magnitude response.
* **Gray Decoding:** The determined integer symbol index (e.g., 0 to 3 for QPSK) is passed through a Gray decoder (`gray_decode(candidate_sym)`). This maps adjacent constellation points to bit sequences that differ by only one bit, reducing the bit-error-rate. 
* **Multi-Carrier Expansion:** If multiple carriers are active (OFDM-like behavior), this process runs in parallel for all frequencies, pulling down multiple sets of bits per symbol.

## 4. Decoding and Output
The final step translates the streams of bits into usable data bytes and outputs it to the user.

* **Bitstream Accumulation:** If the current window is "safe" (not on a boundary) and timing is locked, the extracted bits are appended bit-by-bit to a dynamically resizing buffer `fec_sym_buf`.
* **Preamble Discarding:** To prevent corruption of the byte sequence, the accumulator is explicitly zeroed out upon the *first* timing lock. This discards the raw preamble bits and ensures the data buffer aligns perfectly on byte boundaries. 
* **Forward Error Correction (FEC):** When the carrier drops below the noise floor (transmission ends), the accumulated bitstream is flushed. If FEC is enabled, the bit buffer is fed into `psk_fec_decode()`, which attempts to mathematically fix single or burst bit errors that occurred over the air.
* **Data Output & GUI Preview:** The final, corrected byte array is written out. The code writes the binary output to a file if requested via the `-o` flag (`psk_write_file`), logs a preview excerpt to standard error, and continuously updates the visual SDL GUI widget so the user can watch the stream decode in real time.