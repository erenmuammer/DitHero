# DitHero Audio Converter

A web application for high-quality audio format conversion (to WAV with selectable sample rate and bit depth), featuring professional TPDF dithering for integer formats. Developed by Muammer Eren for Parydise Inc.

## Quick Start

1.  **Prerequisites:**
    *   Python 3.7+
    *   `libsndfile` library (`brew install libsndfile` on macOS, `sudo apt-get install libsndfile1` on Debian/Ubuntu)
    *   (Optional, for MP3 etc.) `ffmpeg` (`brew install ffmpeg`, `sudo apt-get install ffmpeg`)

2.  **Setup:**
    ```bash
    git clone https://github.com/erenmuammer/DitHero.git
    cd DitHero
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    # If using ffmpeg formats, ensure pydub is installed: pip install pydub
    ```

3.  **Run:**
    ```bash
    python3 app.py
    ```
    Access at `http://127.0.0.1:5001`

## Features

*   Convert various audio formats to WAV.
*   Selectable target sample rate (44.1kHz, 48kHz, 88.2kHz, 96kHz).
*   Selectable target bit depth (16-bit Int, 24-bit Int, 32-bit Float).
*   TPDF Dithering applied correctly for integer outputs.
*   Web UI with drag & drop and batch conversion.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 