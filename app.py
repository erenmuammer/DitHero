import os
import sys
from flask import Flask, request, render_template, jsonify, send_from_directory
import soundfile as sf
import numpy as np
import resampy # Added import
from werkzeug.utils import secure_filename
import uuid # For unique filenames
# Try importing pydub for fallback, handle if not installed
try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. MP3 and some other formats might not be supported.", file=sys.stderr)

# Configuration
UPLOAD_FOLDER = 'uploads'
CONVERTED_FOLDER = 'converted'
# Extended list based on soundfile and pydub (if ffmpeg is installed)
ALLOWED_EXTENSIONS = {'wav', 'aiff', 'aif', 'flac', 'ogg', 'mp3', 'opus', 'm4a', 'aac'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERTED_FOLDER'] = CONVERTED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024 # Increased limit slightly (e.g., 64MB)

# Ensure upload and converted directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Moved from main.py ---
def apply_tpdf_dither(audio_data_float):
    """Applies Triangular Probability Density Function (TPDF) dither."""
    # Generate TPDF noise: sum of two uniform distributions
    # Use float64 for intermediate noise calculations for precision
    noise1 = np.random.uniform(-0.5, 0.5, size=audio_data_float.shape).astype(np.float64)
    noise2 = np.random.uniform(-0.5, 0.5, size=audio_data_float.shape).astype(np.float64)
    tpdf_noise = noise1 + noise2

    # Scale noise to the range of one quantization step for 16-bit audio
    # For float audio in [-1.0, 1.0], the 16-bit step is 2.0 / 65536
    quantization_step = 2.0 / (2**16)
    scaled_noise = tpdf_noise * quantization_step

    # Add dither noise before quantization
    # Ensure result stays within float32 if original was float32
    dithered_audio = (audio_data_float.astype(np.float64) + scaled_noise).astype(audio_data_float.dtype)
    return dithered_audio

# --- Moved and adapted from main.py ---
def convert_audio(input_path, output_path, target_sr=44100, target_bit_depth=16):
    """Converts an audio file to 16-bit, 44.1kHz WAV with TPDF dither."""
    print(f"Conversion process started for {input_path}") # Log start
    try:
        # --- 1. Load Audio File ---
        print(f"Loading audio file: {input_path}")
        audio_data = None
        original_sr = None
        try:
            # Read as float32, ensure it's (samples, channels)
            audio_data, original_sr = sf.read(input_path, dtype='float32', always_2d=True)
            print(f"Read with soundfile. Original SR: {original_sr} Hz, Shape: {audio_data.shape}")
            # Soundfile might return (samples, channels), which is what we want.
            # Let's remove the potentially confusing transpose check unless issues arise.
            # if audio_data.shape[0] < audio_data.shape[1]: # If (channels, samples)
            #     audio_data = audio_data.T # Transpose to (samples, channels)

        except Exception as e_sf:
            print(f"Soundfile failed: {e_sf}. Trying pydub fallback...")
            if not PYDUB_AVAILABLE:
                print("pydub not available for fallback.", file=sys.stderr)
                raise RuntimeError(f"Soundfile error: {e_sf}. Pydub not available.") from e_sf

            try:
                audio_segment = AudioSegment.from_file(input_path)
                original_sr = audio_segment.frame_rate
                # Convert to numpy array (samples, channels) float32 in [-1.0, 1.0]
                samples = np.array(audio_segment.get_array_of_samples())
                if audio_segment.channels == 2:
                    audio_data = samples.reshape((-1, 2)).astype(np.float32)
                else:
                    audio_data = samples.reshape((-1, 1)).astype(np.float32)
                # Scale to [-1.0, 1.0] based on sample width
                max_val = (2**(audio_segment.sample_width * 8 - 1))
                audio_data /= max_val
                print(f"Successfully read with pydub. Original SR: {original_sr} Hz, Shape: {audio_data.shape}")

            except Exception as e_pd:
                print(f"Pydub also failed: {e_pd}", file=sys.stderr)
                raise RuntimeError(f"Failed to read audio with soundfile and pydub: {e_pd}") from e_pd

        if audio_data is None or original_sr is None:
             raise RuntimeError("Audio data could not be loaded.")

        # --- 2. Sample Rate Conversion (if necessary) ---
        if original_sr != target_sr:
            print(f"Resampling from {original_sr} Hz to {target_sr} Hz...")
            # resampy expects (samples, channels) or (samples,) for mono
            # It handles multi-channel correctly if shape is (samples, channels)
            # Use kaiser_best for high quality
            resampled_data = resampy.resample(audio_data, original_sr, target_sr, filter='kaiser_best', axis=0)
            audio_data = resampled_data
            print(f"Resampling complete. New shape: {audio_data.shape}")
        else:
            print("Sample rate matches target. No resampling needed.")

        # --- 3. Apply TPDF Dither ---
        print("Applying TPDF dither...")
        dithered_audio_float = apply_tpdf_dither(audio_data)

        # --- 4. Quantization to Target Bit Depth (16-bit integer) ---
        # Clipping is important after adding noise
        print("Clipping audio data between -1.0 and 1.0...")
        dithered_audio_float = np.clip(dithered_audio_float, -1.0, 1.0)

        print(f"Quantizing to {target_bit_depth}-bit...")
        if target_bit_depth == 16:
            # Scale to 16-bit integer range [-32767, 32767] for writing
            # Using 32767 instead of 32768 to avoid potential clipping issues with max value
            audio_data_int = (dithered_audio_float * 32767.0).astype(np.int16)
            subtype = 'PCM_16'
        else:
            # Should not happen with current setup, but good practice
            raise ValueError(f"Unsupported target bit depth: {target_bit_depth}")

        # --- 5. Save Output File ---
        print(f"Saving converted file to: {output_path}")
        sf.write(output_path, audio_data_int, target_sr, subtype=subtype, format='WAV')
        print(f"Conversion successful: {output_path}")
        return True # Indicate success

    except Exception as e:
        # Log the full error for debugging on the server
        print(f"Error during audio conversion process for {input_path}: {e}", file=sys.stderr)
        # Optionally print traceback:
        # import traceback
        # traceback.print_exc()
        # Clean up potentially partially written output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Cleaned up partial output file: {output_path}")
            except Exception as e_clean:
                 print(f"Error cleaning up partial output {output_path}: {e_clean}", file=sys.stderr)
        return False # Indicate failure

# --- End of moved functions ---


def get_audio_info(file_path):
    """Reads basic audio file information using soundfile and fallback."""
    info = {'samplerate': None, 'channels': None, 'format': None, 'subtype': None, 'duration_seconds': None, 'estimated_bit_depth': None, 'error': None}
    try:
        sf_info = sf.info(file_path)
        info['samplerate'] = sf_info.samplerate
        info['channels'] = sf_info.channels
        info['format'] = sf_info.format_info
        info['subtype'] = sf_info.subtype_info
        info['duration_seconds'] = sf_info.duration

        if sf_info.subtype and 'PCM' in sf_info.subtype:
             if '16' in sf_info.subtype: info['estimated_bit_depth'] = 16
             elif '24' in sf_info.subtype: info['estimated_bit_depth'] = 24
             elif '32' in sf_info.subtype: info['estimated_bit_depth'] = 32
             elif '08' in sf_info.subtype: info['estimated_bit_depth'] = 8
        print(f"Soundfile Info: {sf_info}") # Debugging

    except Exception as e_sf:
        info['error'] = f"Soundfile Error: {e_sf}"
        if PYDUB_AVAILABLE:
            try:
                print(f"Attempting fallback analysis with pydub for {file_path}") # Debugging
                audio_segment = AudioSegment.from_file(file_path)
                info['samplerate'] = audio_segment.frame_rate
                info['channels'] = audio_segment.channels
                info['duration_seconds'] = audio_segment.duration_seconds
                try:
                    media_info = mediainfo(file_path)
                    info['format'] = media_info.get('codec_name', 'N/A')
                    # Handle potential ValueError if bits_per_sample is not an integer string
                    bits_str = media_info.get('bits_per_sample')
                    info['estimated_bit_depth'] = int(bits_str) if bits_str and bits_str.isdigit() else None
                    print(f"Pydub MediaInfo: {media_info}") # Debugging
                except Exception as e_mi:
                    print(f"Could not get detailed media info via pydub: {e_mi}")
                    # Try to guess format from extension if mediainfo fails
                    _, ext = os.path.splitext(file_path)
                    info['format'] = info['format'] or ext.lower().strip('.') or 'N/A (pydub)'
                info['error'] = None # Clear soundfile error if pydub worked

            except Exception as e_pd:
                info['error'] = info['error'] or f"Pydub Error: {e_pd}"
                print(f"Pydub analysis error: {e_pd}") # Debugging
        else:
             info['error'] = info['error'] or "Pydub not installed. Cannot analyze this format."
             print("Pydub not installed for fallback analysis.") # Debugging

    return info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(filepath)
            print(f"File saved to: {filepath}") # Debugging
            audio_info = get_audio_info(filepath)
            print(f"Audio Info: {audio_info}") # Debugging

            if audio_info.get('error') and not audio_info.get('samplerate'): # If error AND couldn't even get basic info
                 # Clean up the unusable file
                 if os.path.exists(filepath):
                    try: os.remove(filepath)
                    except: pass # Ignore cleanup error
                 return jsonify({'error': f"Failed to analyze file: {audio_info['error']}"}), 500
            elif audio_info.get('error'):
                 # Allow conversion attempt even if some info is missing/errored
                 print(f"Warning during analysis: {audio_info['error']}")
                 return jsonify({
                     'message': f'File uploaded, but analysis incomplete: {audio_info["error"]}. Conversion might still work.',
                     'info': audio_info,
                     'filepath': unique_filename,
                     'original_filename': original_filename
                 })
            else:
                 return jsonify({
                     'message': 'File uploaded and analyzed successfully',
                     'info': audio_info,
                     'filepath': unique_filename,
                     'original_filename': original_filename
                 })

        except Exception as e:
            print(f"Error during file upload/save: {e}", file=sys.stderr)
            if os.path.exists(filepath): # Cleanup on save error
                 try: os.remove(filepath)
                 except: pass
            return jsonify({'error': f'Server error during upload: {e}'}), 500
    else:
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400


@app.route('/convert', methods=['POST'])
def convert_file_route():
    data = request.get_json()
    input_unique_filename = data.get('filepath')
    original_filename = data.get('original_filename', 'converted_file')

    if not input_unique_filename:
        return jsonify({'error': 'Missing filepath for conversion'}), 400

    # Validate that the filename looks like one we generated (UUID prefix)
    # This is a basic security measure
    try:
        uuid.UUID(input_unique_filename.split('_')[0])
    except (ValueError, IndexError):
        return jsonify({'error': 'Invalid filepath format.'}), 400


    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_unique_filename)

    if not os.path.exists(input_path):
        # Check if it's already been converted and cleaned up maybe?
        print(f"Error: Input file {input_path} not found for conversion.")
        return jsonify({'error': 'Uploaded file not found on server. It might have been cleaned up or never saved correctly. Please upload again.'}), 404

    # Construct output filename
    base, _ = os.path.splitext(original_filename)
    # Ensure the base name is also secured
    safe_base = secure_filename(base)
    output_filename = f"{safe_base}_16bit_441kHz.wav"
    output_path = os.path.join(app.config['CONVERTED_FOLDER'], output_filename)

    print(f"Starting conversion task: {input_path} -> {output_path}")

    # --- Call the actual conversion logic ---
    try:
        # Use the convert_audio function we moved into this file
        success = convert_audio(input_path, output_path)
    except Exception as e:
        # Catch unexpected errors during the call itself
        print(f"Fatal error during conversion call for {input_path}: {e}", file=sys.stderr)
        success = False


    # --- Cleanup Uploaded File ---
    # Clean up the uploaded file regardless of success/failure now that conversion attempt is done
    # (Unless debugging requires keeping it)
    if os.path.exists(input_path):
       try:
           os.remove(input_path)
           print(f"Cleaned up uploaded file: {input_path}")
       except Exception as e:
           print(f"Warning: Error cleaning up upload {input_path}: {e}", file=sys.stderr)


    if success:
         print(f"Conversion task successful for {original_filename}.")
         return jsonify({
             'message': 'Conversion successful!',
             'download_filename': output_filename # Send filename for download URL
         })
    else:
        # convert_audio function should handle cleanup of partial output,
        # but we return a server error here.
        print(f"Conversion task failed for {original_filename}.")
        return jsonify({'error': 'Conversion failed. See server logs for details.'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    # Basic check against path traversal
    if '..' in filename or filename.startswith('/'):
        from flask import abort
        print(f"Download forbidden for unsafe filename: {filename}")
        abort(400, description="Invalid filename")

    safe_filename = secure_filename(filename)
    print(f"Download request for: {safe_filename}")
    try:
        # Before sending, check if file actually exists to provide better error
        file_path = os.path.join(app.config['CONVERTED_FOLDER'], safe_filename)
        if not os.path.isfile(file_path):
             print(f"Download error: File not found at path - {file_path}")
             from flask import abort
             abort(404, description="Converted file not found. It may have been cleaned up or the conversion failed.")

        return send_from_directory(app.config['CONVERTED_FOLDER'], safe_filename, as_attachment=True)
    except FileNotFoundError: # Should be caught above, but as fallback
        print(f"Download error: File not found - {safe_filename}")
        from flask import abort
        abort(404, description="Resource not found")
    except Exception as e:
        print(f"Error during download attempt for {safe_filename}: {e}", file=sys.stderr)
        from flask import abort
        abort(500, description="Server error during download.")


if __name__ == '__main__':
    print(f"Flask starting. Pydub available: {PYDUB_AVAILABLE}")
    # Consider setting debug=False when deploying
    app.run(debug=True, host='0.0.0.0', port=5001) 