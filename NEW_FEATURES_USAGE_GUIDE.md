## New Feature Usage Guide for F5-TTS

This guide outlines how to use the recently added features: Detailed Progress Updates, Enhanced Error Handling, and Intermediate Results Saving.

### 1. Create a Sample Long Text for Testing

To effectively test features like chunk-based progress, error skipping, and intermediate saves, you'll need a long text input.

**Method:**

1.  **Base Paragraph:** Start with a paragraph of text. For example:
    ```
    The quick brown fox jumps over the lazy dog. This classic sentence contains all letters of the English alphabet, making it a good test case. We are repeating this paragraph multiple times to simulate a long text input for the F5-TTS system. This will help us observe chunked processing, error handling, and intermediate save functionalities. F5-TTS aims for fluent and faithful speech, so let's put it through its paces.
    ```
2.  **Repeat:** Copy this paragraph and paste it multiple times into a new plain text file (e.g., `long_text_sample.txt`). Repeating it 10-15 times should generate enough content to be split into several chunks by the TTS system (typically 5-10 chunks, depending on your reference audio length and chunking settings).

3.  **Optional - Text with Potential Error:** To test the `skip_on_error` feature, you can create a variation of this file, say `long_text_with_error_sample.txt`. In one of an_error_inducing_sequence_!@#$%^&*()_+[]{};':"\|,.<>/?` (the exact nature of what causes an error can be model-dependent).
    *Example of a modified sentence:*
    ```
    This classic sentence contains all letters of the English alphabet, making it a good test case. This_is_an_error_inducing_sequence_!@#$%^&*()_+[]{};':"\|,.<>/? We are repeating this paragraph multiple times...
    ```
    *Note: Creating a sequence that reliably causes an error without knowing specific model sensitivities can be challenging. This example is illustrative.*

### 2. Usage Summary and Examples

#### A. Detailed Progress Updates

This enhancement provides more granular feedback during audio generation, showing the current chunk being processed, the total number of chunks, and an Estimated Time of Arrival (ETA).

*   **CLI (Command Line Interface):**
    *   The `tqdm` progress bar in your terminal will automatically display this enhanced information. No special commands are needed.
    *   **Example Display:**
        ```
        Processing chunk 3/10, ETA: 00:45: 30%|███       | 3/10 [00:15<00:45,  6.50s/it]
        ```

*   **Gradio UI:**
    *   The progress bar in the Gradio interface will also show the "Processing chunk X/Y, ETA: MM:SS" message in its description area when you synthesize speech.

#### B. Enhanced Error Handling (`skip_on_error`)

This feature allows the TTS process to skip problematic text chunks that might otherwise cause generation to fail, and continue with the rest ofthe text.

*   **CLI Example:**
    Use the `long_text_with_error_sample.txt` file created earlier (or a similar file where you suspect a chunk might cause an issue).

    1.  **Command without skipping (default behavior):**
        ```bash
        f5-tts_infer-cli --config your_config.toml --gen_file long_text_with_error_sample.txt --output_file error_test_stop.wav
        ```
        *   **Explanation:** If a text chunk causes an error, the generation process may halt. The output file might be incomplete or not created.

    2.  **Command with error skipping:**
        ```bash
        f5-tts_infer-cli --config your_config.toml --gen_file long_text_with_error_sample.txt --output_file error_test_skip.wav --skip_on_error
        ```
        *   **Explanation:** With `--skip_on_error`, problematic chunks are skipped. Errors are logged to the console, and the process continues. `error_test_skip.wav` will contain audio from successfully processed chunks.

*   **Gradio UI Example:**

    1.  **Upload Text:** In the "Basic-TTS" tab (or other relevant tabs), upload your `long_text_with_error_sample.txt`.
    2.  **Enable Skipping:** Expand "Advanced Settings" and check the "Skip on Error" checkbox.
    3.  **Generate:** Click "Synthesize".
    4.  **Outcome:** Generation will attempt to complete. Skipped chunk errors will be logged (usually to the server console where Gradio is running). The final audio will comprise successfully synthesized segments.

#### C. Intermediate Results Saving (`save_intermediate_every_n_chunks`)

This feature saves the generated audio incrementally after a specified number of chunks.

*   **CLI Example:**
    Use the `long_text_sample.txt` file.

    *   **Command:**
        ```bash
        f5-tts_infer-cli --config your_config.toml --gen_file long_text_sample.txt --output_file long_output.wav --save_intermediate_every_n_chunks 3
        ```
    *   **Explanation:**
        *   An intermediate audio file will be saved after every 3 successful chunks.
        *   Intermediate files will be named like `long_output_intermediate_part_1.wav` (chunks 1-3), `long_output_intermediate_part_2.wav` (chunks 1-6), etc., in the same directory as `long_output.wav`.
        *   The final `long_output.wav` will contain all successfully generated chunks with proper cross-fading.
        *   *Note:* Intermediate saves are simple concatenations. The final output applies any configured cross-fading.

*   **Gradio UI Example:**

    1.  **Setup:** Upload reference audio and your `long_text_sample.txt`.
    2.  **Enable Intermediate Saving:** In "Advanced Settings", set "Save Intermediate Audio Every N Chunks" to a value like `2`. (Default `0` disables it).
    3.  **Generate:** Click "Synthesize".
    4.  **Outcome:**
        *   Intermediate audio files (e.g., `f5tts_gradio_intermediate_YYYYMMDD_HHMMSS_part_X.wav`) are saved to your system's temporary directory (e.g., `/tmp/` or `%TEMP%`).
        *   The full paths of these intermediate files are logged in the Gradio UI's notification area or the server console.

### 3. General Testing Notes

*   **Reference Audio:** Use a clean, short (<12s) reference audio file (WAV format recommended) and its accurate transcription (if not using auto-transcribe) for reliable testing.
*   **Verifying Intermediate Saves:** For `save_intermediate_every_n_chunks`, start with a small number (e.g., `1` or `2`) to easily verify file creation.
*   **Testing `skip_on_error`:** As mentioned, creating a reliably erroring text chunk can be difficult. Focus on whether the `--skip_on_error` flag allows the process to continue past *any* encountered errors (even if you have to simulate one by modifying code temporarily for a test, though that's beyond user testing). Check console logs for error messages about skipped chunks.

This guide should assist in understanding and utilizing these new F5-TTS enhancements.
