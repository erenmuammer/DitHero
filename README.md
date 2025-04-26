# DitHero Audio Converter

A simple yet powerful web application to convert various audio file formats to high-quality 16-bit, 44.1kHz WAV files using professional TPDF dithering.

## Features

*   **High-Quality Conversion:** Converts audio to 16-bit, 44.1kHz WAV format.
*   **Professional Dithering:** Applies Triangular Probability Density Function (TPDF) dither before quantization to minimize artifacts.
*   **Quality Resampling:** Uses `resampy` library for high-quality sample rate conversion when needed.
*   **Wide Format Support:** Reads various formats like WAV, AIFF, FLAC, Ogg Vorbis, and MP3 (requires `ffmpeg` for MP3 and others handled by `pydub`).
*   **Web Interface:** Easy-to-use web UI built with Flask and Bootstrap.
*   **Batch Processing:** Upload and convert multiple files sequentially.

## Requirements

*   **Python:** 3.7+ recommended.
*   **libsndfile:** Required by the `soundfile` library for reading/writing many audio formats.
    *   macOS: `brew install libsndfile`
    *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install libsndfile1`
    *   Windows: Download from the [official website](https://libsndfile.github.io/libsndfile/) or use a package manager like Chocolatey.
*   **ffmpeg (Optional but Recommended):** Required by `pydub` for handling formats not natively supported by `libsndfile` (like MP3).
    *   macOS: `brew install ffmpeg`
    *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install ffmpeg`
    *   Windows: Download from the [official website](https://ffmpeg.org/download.html).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd DitHero
    ```
2.  **(Recommended) Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you intend to process MP3s or other formats requiring `ffmpeg`, make sure you also install `pydub` (`pip install pydub`) and have `ffmpeg` installed on your system.*

## Running the Application

1.  **Start the Flask server:**
    ```bash
    python3 app.py
    ```
2.  **Access the application:** Open your web browser and navigate to `http://127.0.0.1:5001` (or the address shown in the terminal).

## Usage

1.  Click "Choose Audio File(s)" to select one or more audio files from your computer.
2.  The application will analyze each file and display its details (Sample Rate, Bit Depth, etc.) and status.
3.  For files marked as "Ready", you can:
    *   Click the individual "Convert" button next to a file.
    *   Click the "Convert All Ready" button at the top right of the queue to process all ready files sequentially.
4.  Once a file is converted successfully, its status will change to "Completed", and a "Download" button will appear.

## Author

*   **Muammer Eren**
*   muammer.eren@tedu.edu.tr

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if added). 