import os
import sys
import subprocess
import json
import zipfile
import tempfile
import time
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
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

# Check if FFprobe is available (typically comes with FFmpeg)
def is_ffprobe_available():
    try:
        subprocess.run(['ffprobe', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

FFPROBE_AVAILABLE = is_ffprobe_available()
print(f"FFprobe available: {FFPROBE_AVAILABLE}")

# Function to get accurate bitrate using FFprobe
def get_mp3_bitrate_with_ffprobe(file_path):
    """Use FFprobe to get accurate MP3 bitrate information"""
    try:
        # Run ffprobe with JSON output format for easy parsing
        command = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_format', 
            '-show_streams', 
            file_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFprobe error: {result.stderr}")
            return None
            
        # Parse the JSON output
        data = json.loads(result.stdout)
        print(f"FFprobe data: {data}")
        
        # Try to find bitrate in the output
        bitrate = None
        
        # First check format section for bitrate
        if 'format' in data and 'bit_rate' in data['format']:
            bitrate = int(data['format']['bit_rate'])
        
        # If not found, check each audio stream
        if not bitrate and 'streams' in data:
            for stream in data['streams']:
                if stream.get('codec_type') == 'audio' and 'bit_rate' in stream:
                    bitrate = int(stream['bit_rate'])
                    break
        
        # Convert from bps to kbps if needed
        if bitrate and bitrate > 1000:
            bitrate = int(bitrate / 1000)
            
        print(f"FFprobe detected bitrate: {bitrate} kbps")
        return bitrate
        
    except Exception as e:
        print(f"Error using FFprobe: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Moved from main.py ---
def apply_tpdf_dither(audio_data_float, target_bit_depth):
    """Applies TPDF dither scaled to the target integer bit depth."""
    if not isinstance(target_bit_depth, int) or target_bit_depth <= 0:
        # Safety check, shouldn't happen with current usage but good practice
        print(f"Warning: Invalid target_bit_depth ({target_bit_depth}) for dither. Skipping dither.")
        return audio_data_float

    print(f"Applying TPDF dither scaled for {target_bit_depth}-bit target...")
    # Generate TPDF noise: sum of two uniform distributions
    # Use float64 for intermediate noise calculations for precision
    noise1 = np.random.uniform(-0.5, 0.5, size=audio_data_float.shape).astype(np.float64)
    noise2 = np.random.uniform(-0.5, 0.5, size=audio_data_float.shape).astype(np.float64)
    tpdf_noise = noise1 + noise2

    # Scale noise to the range of one quantization step for the TARGET bit depth
    quantization_step = 2.0 / (2**target_bit_depth)
    scaled_noise = tpdf_noise * quantization_step

    # Add dither noise before quantization
    # Ensure result stays within float32 if original was float32
    dithered_audio = (audio_data_float.astype(np.float64) + scaled_noise).astype(audio_data_float.dtype)
    return dithered_audio

# --- Moved and adapted from main.py ---
def convert_audio(input_path, output_path, target_sr=44100, target_bit_depth=16):
    """Converts an audio file to target format with optional TPDF dither."""
    print(f"Conversion process started for {input_path} to SR={target_sr}, BitDepth={target_bit_depth}") # Log start
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

        # --- 3. Apply TPDF Dither (Only for Integer Targets) ---
        apply_dither = False
        if isinstance(target_bit_depth, int) and target_bit_depth in [16, 24]: # Check if it's an integer target we support dithering for
            # Pass the target_bit_depth to the dither function
            audio_data = apply_tpdf_dither(audio_data, target_bit_depth)
            apply_dither = True # Keep track if dither was applied (optional)
        elif target_bit_depth == 'float':
            print("Skipping dither for non-integer target (float).")
        else:
            # Should not happen if validation in /convert is correct
             print(f"Warning: Dither not applied for unexpected target_bit_depth: {target_bit_depth}")

        # --- 4. Quantization/Clipping ---
        # Clipping is always important, especially if dither was added or if source was float
        print("Clipping audio data between -1.0 and 1.0...")
        audio_data = np.clip(audio_data, -1.0, 1.0)

        print(f"Preparing for target format: {target_bit_depth}-bit...")
        subtype = None
        if target_bit_depth == 16:
            # Scale to 16-bit integer range
            audio_data_out = (audio_data * 32767.0).astype(np.int16)
            subtype = 'PCM_16'
        elif target_bit_depth == 24:
             # Scale to 24-bit integer range (use float first, then scale to int range for sf.write)
            # sf.write handles scaling for PCM_24 if data is int32, but it's safer to scale manually
            # Scale to approx range [-8388607, 8388607]
            audio_data_out = (audio_data * 8388607.0).astype(np.int32) # Write as int32, sf handles 24bit packing
            subtype = 'PCM_24'
        # Note: Soundfile PCM_32 might not be universally compatible. Often float is preferred.
        # elif target_bit_depth == 32:
        #     audio_data_out = (audio_data * 2147483647.0).astype(np.int32)
        #     subtype = 'PCM_32'
        elif target_bit_depth == 'float': # Check for string 'float' from JS
            audio_data_out = audio_data.astype(np.float32)
            subtype = 'FLOAT'
        else:
            raise ValueError(f"Unsupported target bit depth: {target_bit_depth}")

        # --- 5. Save Output File ---
        print(f"Saving converted file to: {output_path} with subtype: {subtype}")
        sf.write(output_path, audio_data_out, target_sr, subtype=subtype, format='WAV')
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

# --- End of convert_audio function ---


def get_audio_info(file_path):
    """Reads basic audio file information using soundfile and fallback."""
    info = {'samplerate': None, 'channels': None, 'format': None, 'subtype': None, 'duration_seconds': None, 'estimated_bit_depth': None, 'error': None, 'bit_rate': None}
    
    # For MP3 files, try to get accurate bitrate with ffprobe first
    if file_path.lower().endswith('.mp3') and FFPROBE_AVAILABLE:
        print(f"Using FFprobe to get MP3 bitrate for {file_path}")
        bitrate = get_mp3_bitrate_with_ffprobe(file_path)
        if bitrate:
            info['bit_rate'] = bitrate
            print(f"FFprobe detected bitrate: {bitrate} kbps")
    
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
                    # Get detailed mediainfo for MP3 files to extract bitrate
                    print(f"Getting detailed media info via pydub for {file_path}")
                    media_info = mediainfo(file_path)
                    print(f"Full MediaInfo: {media_info}") # Debug: Dump entire media_info
                    
                    # Look for format/codec info
                    info['format'] = media_info.get('codec_name', 'N/A')
                    
                    # Handle potential ValueError if bits_per_sample is not an integer string
                    bits_str = media_info.get('bits_per_sample')
                    info['estimated_bit_depth'] = int(bits_str) if bits_str and bits_str.isdigit() else None
                    
                    # Only check for bitrate if we don't already have it from FFprobe
                    if info['bit_rate'] is None:
                        # Check all possible bitrate keys in mediainfo
                        possible_bitrate_keys = ['bit_rate', 'bitrate', 'nominal_bit_rate']
                        
                        # Try different keys for bit rate
                        bit_rate = None
                        for key in possible_bitrate_keys:
                            if key in media_info and media_info[key]:
                                bit_rate = media_info[key]
                                print(f"Found bitrate in key '{key}': {bit_rate}")
                                break
                        
                        # Also check if the filename contains bitrate info (common for some MP3s)
                        if not bit_rate and os.path.basename(file_path).lower().endswith('.mp3'):
                            # Check if mediainfo has audio stream details
                            if 'streams' in media_info and len(media_info['streams']) > 0:
                                for stream in media_info['streams']:
                                    if stream.get('codec_type') == 'audio' and 'bit_rate' in stream:
                                        bit_rate = stream['bit_rate']
                                        print(f"Found bitrate in stream info: {bit_rate}")
                                        break
                        
                        if bit_rate:
                            try:
                                # FFmpeg sometimes returns bit rate as a string like "320000" or "320k"
                                # Convert to integer and handle different formats
                                if isinstance(bit_rate, str) and 'k' in bit_rate.lower():
                                    # Handle "320k" format
                                    bit_rate_val = int(bit_rate.lower().replace('k', '').strip())
                                    print(f"Parsed '{bit_rate}' as {bit_rate_val} kbps")
                                else:
                                    bit_rate_val = int(bit_rate)
                                    # If larger than 1000, it's likely in bps, convert to kbps for display
                                    if bit_rate_val > 1000:
                                        bit_rate_val = int(bit_rate_val / 1000)
                                        print(f"Converted {bit_rate} bps to {bit_rate_val} kbps")
                                    
                                info['bit_rate'] = bit_rate_val
                                print(f"Final bitrate value: {info['bit_rate']} kbps")
                            except (ValueError, TypeError) as e:
                                # Just store as-is if parsing fails
                                print(f"Could not parse bitrate value '{bit_rate}': {e}")
                                info['bit_rate'] = bit_rate
                        else:
                            # For MP3 files, try to estimate bitrate from file size and duration
                            if os.path.basename(file_path).lower().endswith('.mp3') and info['duration_seconds'] and info['bit_rate'] is None:
                                try:
                                    file_size_bytes = os.path.getsize(file_path)
                                    # Bitrate = file size in bits / duration in seconds
                                    # This is approximate but works reasonably well for MP3
                                    estimated_bitrate = int((file_size_bytes * 8) / info['duration_seconds'] / 1000)
                                    info['bit_rate'] = estimated_bitrate
                                    print(f"Estimated bitrate from file size: {estimated_bitrate} kbps")
                                except Exception as e_size:
                                    print(f"Could not estimate bitrate from file size: {e_size}")
                    
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
    if not data:
        return jsonify({'error': 'Invalid request format (no JSON)'}), 400

    input_unique_filename = data.get('filepath')
    original_filename = data.get('original_filename', 'converted_file')
    # Get target format if specified
    target_format = data.get('target_format', '').lower()
    
    # Get target SR and Bit Depth from request, with defaults
    try:
        target_sr = int(data.get('target_sr', 44100)) # Default to 44100
    except (ValueError, TypeError):
        target_sr = 44100
        print(f"Warning: Invalid target_sr received, defaulting to {target_sr}")

    target_bit_depth_str = str(data.get('target_bit_depth', '16')).lower() # Default to '16'
    # Map string to internal representation (int or 'float')
    if target_bit_depth_str == 'float':
        target_bit_depth = 'float'
    elif target_bit_depth_str.isdigit() and int(target_bit_depth_str) in [16, 24]: # Add 32 if supported later
        target_bit_depth = int(target_bit_depth_str)
    else:
        target_bit_depth = 16 # Default to 16-bit if invalid
        print(f"Warning: Invalid target_bit_depth received ('{target_bit_depth_str}'), defaulting to {target_bit_depth}")

    # Get MP3 bitrate if provided
    mp3_bitrate = None
    try:
        if 'mp3_bitrate' in data and data['mp3_bitrate'] is not None:
            mp3_bitrate = int(data['mp3_bitrate'])
            # Validate that it's a common bitrate value
            if mp3_bitrate not in [128, 192, 256, 320]:
                print(f"Warning: Unusual MP3 bitrate received: {mp3_bitrate}, but will attempt")
    except (ValueError, TypeError):
        print(f"Warning: Invalid mp3_bitrate received, ignoring")
        mp3_bitrate = None


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

    # Determine conversion type
    convert_to_mp3 = target_format == 'mp3' or original_filename.lower().endswith('.mp3')
    
    # Special case: convert from WAV/AIFF/FLAC to MP3
    convert_wav_to_mp3 = False
    if target_format == 'mp3' and not original_filename.lower().endswith('.mp3'):
        convert_wav_to_mp3 = True
        if not mp3_bitrate:
            mp3_bitrate = 320  # Default to high quality if not specified
        print(f"Converting from WAV/AIFF/FLAC to MP3 at {mp3_bitrate}kbps")

    # Construct output filename reflecting the format
    base, ext = os.path.splitext(original_filename)
    safe_base = secure_filename(base)
    
    if convert_wav_to_mp3:
        # For WAV to MP3 conversion
        output_filename = f"{safe_base}_{mp3_bitrate}kbps.mp3"
    elif convert_to_mp3:
        # For MP3 bitrate conversion
        output_filename = f"{safe_base}_{mp3_bitrate}kbps.mp3"
    else:
        # Format bit depth string for filename (for non-MP3 files)
        bit_depth_fname = f"{target_bit_depth}bit" if isinstance(target_bit_depth, int) else "float"
        output_filename = f"{safe_base}_{bit_depth_fname}_{target_sr//1000}kHz.wav"
    
    output_path = os.path.join(app.config['CONVERTED_FOLDER'], output_filename)

    print(f"Starting conversion task: {input_path} -> {output_path}")
    if convert_wav_to_mp3:
        print(f"WAV to MP3 conversion with target bitrate: {mp3_bitrate} kbps")
    elif convert_to_mp3:
        print(f"MP3 conversion with target bitrate: {mp3_bitrate} kbps")
    else:
        print(f"Standard conversion with SR={target_sr}, BD={target_bit_depth}")

    # --- Call the actual conversion logic ---
    try:
        success = False
        if convert_wav_to_mp3:
            # Use WAV to MP3 conversion
            success = convert_to_mp3_format(input_path, output_path, mp3_bitrate)
        elif convert_to_mp3:
            # Use MP3-specific conversion
            success = convert_mp3_bitrate(input_path, output_path, mp3_bitrate)
        else:
            # Pass the target parameters to regular converter
            success = convert_audio(input_path, output_path, target_sr=target_sr, target_bit_depth=target_bit_depth)
    except Exception as e:
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

# Function to convert MP3 bitrate
def convert_mp3_bitrate(input_path, output_path, target_bitrate):
    """
    Converts an MP3 file to a different bitrate using pydub (which uses FFmpeg)
    
    Args:
        input_path: Path to the input MP3 file
        output_path: Path to save the converted MP3 file
        target_bitrate: Target bitrate in kbps (e.g., 128, 192, 256, 320)
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    print(f"Converting MP3 bitrate: {input_path} -> {output_path} @ {target_bitrate}kbps")
    
    if not PYDUB_AVAILABLE:
        print("Error: pydub not available. Cannot convert MP3 bitrate.")
        return False
    
    try:
        # Get current bitrate for logging
        current_bitrate = None
        try:
            info = mediainfo(input_path)
            print(f"Input file mediainfo: {info}")
            
            # Try to find bitrate in the mediainfo
            for key in ['bit_rate', 'bitrate', 'nominal_bit_rate']:
                if key in info and info[key]:
                    current_bitrate = info[key]
                    break
                    
            # Parse the bitrate
            if current_bitrate:
                if isinstance(current_bitrate, str) and 'k' in current_bitrate.lower():
                    current_bitrate = int(current_bitrate.lower().replace('k', '').strip())
                else:
                    current_bitrate = int(current_bitrate)
                    if current_bitrate > 1000:
                        current_bitrate = int(current_bitrate / 1000)
                print(f"Current MP3 bitrate: {current_bitrate}kbps")
            
            # If mediainfo doesn't provide bitrate, estimate from file size
            if not current_bitrate:
                file_size = os.path.getsize(input_path)
                duration = float(info.get('duration', 0))
                if duration > 0:
                    estimated_bitrate = int((file_size * 8) / duration / 1000)
                    current_bitrate = estimated_bitrate
                    print(f"Estimated MP3 bitrate from file size: {current_bitrate}kbps")
                
        except Exception as e:
            print(f"Could not determine current bitrate: {e}")
        
        # Load the MP3 file
        print(f"Loading MP3 file: {input_path}")
        audio = AudioSegment.from_mp3(input_path)
        
        # Get some stats about the audio
        print(f"Audio stats: {len(audio)/1000}s, {audio.channels} channels, {audio.frame_rate}Hz, {audio.sample_width*8} bits")
        
        # Prepare FFmpeg parameters for better quality
        # Different approaches for different target bitrates
        parameters = []
        
        if target_bitrate >= 256:
            # For high bitrates, use strict CBR encoding
            parameters = [
                "-c:a", "libmp3lame",                # MP3 codec
                "-b:a", f"{target_bitrate}k",        # Target bitrate
                "-abr", "0",                         # Disable ABR mode
                "-ac", str(audio.channels),          # Keep same number of channels
                "-ar", str(audio.frame_rate),        # Keep same sample rate
                "-q:a", "0",                         # Highest quality encoding
                "-compression_level", "0",           # Highest compression level
                "-application", "audio",             # Audio application type
                "-cutoff", "20000",                  # Maximum frequency cutoff
            ]
            
            # Add CBR mode enforcement - key fix for 320kbps
            if target_bitrate == 320:
                parameters.extend([
                    "-vbr", "off",                   # Disable VBR
                    "-minrate", f"{target_bitrate}k", # Set minimum bitrate
                    "-maxrate", f"{target_bitrate}k", # Set maximum bitrate
                    "-bufsize", f"{target_bitrate}k", # Set buffer size
                    "-joint_stereo", "0",            # Disable joint stereo for maximum quality
                ])
                print(f"Using strict CBR encoding for {target_bitrate}kbps")
            else:
                parameters.extend([
                    "-joint_stereo", "1",            # Use joint stereo for good quality/size
                ])
                print(f"Using high-quality CBR encoding ({target_bitrate}kbps)")
        else:
            # For lower bitrates, use variable bitrate (VBR) for better quality/size ratio
            # Map bitrate to VBR quality setting (0=best, 9=worst)
            vbr_quality = 0  # Default high quality
            if target_bitrate <= 128:
                vbr_quality = 4  # Medium quality for lower bitrates
            elif target_bitrate <= 192:
                vbr_quality = 2  # Better quality for medium bitrates
                
            parameters = ["-q:a", str(vbr_quality)]
            print(f"Using VBR encoding with quality level {vbr_quality} for {target_bitrate}kbps")
            
        # Export with the new bitrate
        print(f"Exporting MP3 with bitrate: {target_bitrate}kbps")
        print(f"Using FFmpeg parameters: {parameters}")
        audio.export(
            output_path,
            format="mp3",
            bitrate=f"{target_bitrate}k",
            parameters=parameters
        )
        
        # Verify the file was created and has the correct bitrate
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                # Verify the output file has the correct bitrate
                output_info = mediainfo(output_path)
                print(f"Output file mediainfo: {output_info}")
                
                # Get the final bitrate to confirm conversion
                final_bitrate = None
                for key in ['bit_rate', 'bitrate', 'nominal_bit_rate']:
                    if key in output_info and output_info[key]:
                        final_bitrate = output_info[key]
                        break
                        
                if final_bitrate:
                    if isinstance(final_bitrate, str) and 'k' in final_bitrate.lower():
                        final_bitrate = int(final_bitrate.lower().replace('k', '').strip())
                    else:
                        final_bitrate = int(final_bitrate)
                        if final_bitrate > 1000:
                            final_bitrate = int(final_bitrate / 1000)
                    print(f"Output MP3 bitrate confirmed: {final_bitrate} kbps")
                    
                # Additional verification with ffprobe for 320kbps
                if FFPROBE_AVAILABLE and target_bitrate == 320:
                    actual_bitrate = get_mp3_bitrate_with_ffprobe(output_path)
                    print(f"Verified bitrate with ffprobe: {actual_bitrate} kbps (target was {target_bitrate} kbps)")
                    
                    # If bitrate is significantly lower, try direct FFmpeg approach
                    if actual_bitrate and actual_bitrate < target_bitrate * 0.95:
                        print(f"Bitrate is too low ({actual_bitrate}kbps), falling back to direct FFmpeg encoding")
                        return convert_to_mp3_with_ffmpeg(input_path, output_path, target_bitrate)
                
            except Exception as e:
                print(f"Could not verify output bitrate: {e}")
                
            print(f"MP3 bitrate conversion successful: {output_path}")
            return True
        else:
            print(f"MP3 output file missing or empty: {output_path}")
            return False
            
    except Exception as e:
        print(f"Error during MP3 bitrate conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Clean up potentially partially written output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Cleaned up partial output file: {output_path}")
            except Exception as e_clean:
                print(f"Error cleaning up partial output {output_path}: {e_clean}", file=sys.stderr)
        return False

# Function to convert any audio format to MP3
def convert_to_mp3_format(input_path, output_path, target_bitrate):
    """
    Converts any supported audio file to MP3 format with high quality settings
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the converted MP3 file
        target_bitrate: Target bitrate in kbps (e.g., 128, 192, 256, 320)
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    print(f"Converting to MP3: {input_path} -> {output_path} @ {target_bitrate}kbps")
    
    if not PYDUB_AVAILABLE:
        print("Error: pydub not available. Cannot convert to MP3 format.")
        return False
    
    try:
        # Load the audio file (pydub can handle various formats)
        print(f"Loading audio file: {input_path}")
        audio = AudioSegment.from_file(input_path)
        
        # Get some stats about the audio
        print(f"Audio stats: {len(audio)/1000}s, {audio.channels} channels, {audio.frame_rate}Hz, {audio.sample_width*8} bits")
        
        # Determine optimal encoding parameters based on target bitrate
        # For best quality with minimal loss when converting lossless to MP3
        parameters = []
        
        if target_bitrate >= 256:
            # For high bitrates, use constant bitrate (CBR) with strict enforcement
            parameters = [
                "-b:a", f"{target_bitrate}k",     # Target bitrate
                "-c:a", "libmp3lame",             # MP3 codec
                "-abr", "0",                      # Disable ABR mode
                "-codec:a", "libmp3lame",         # Explicitly set codec
                "-ac", str(audio.channels),       # Keep same number of channels
                "-ar", str(audio.frame_rate),     # Keep same sample rate
                "-q:a", "0",                      # Highest quality encoding
                "-compression_level", "0",        # Highest compression level
                "-application", "audio",          # Audio application type
                "-cutoff", "20000",               # Maximum frequency cutoff
            ]
            
            # Add CBR mode enforcement - key fix for 320kbps
            if target_bitrate == 320:
                parameters.extend([
                    "-vbr", "off",                # Disable VBR
                    "-minrate", f"{target_bitrate}k",  # Set minimum bitrate
                    "-maxrate", f"{target_bitrate}k",  # Set maximum bitrate
                    "-bufsize", f"{target_bitrate}k",  # Set buffer size
                    "-application", "audio",      # Audio application type
                    "-joint_stereo", "0",         # Disable joint stereo for maximum quality
                ])
                print(f"Using strict CBR encoding for {target_bitrate}kbps")
            else:
                # For other high bitrates (256kbps)
                parameters.extend([
                    "-joint_stereo", "1",         # Use joint stereo for good quality/size
                ])
                print(f"Using high-quality CBR encoding ({target_bitrate}kbps)")
        else:
            # For lower bitrates, use variable bitrate (VBR) for better quality/size ratio
            # Map bitrate to VBR quality setting (0=best, 9=worst)
            vbr_quality = 0  # Default high quality
            if target_bitrate <= 128:
                vbr_quality = 3  # Medium quality for lower bitrates
            elif target_bitrate <= 192:
                vbr_quality = 2  # Better quality for medium bitrates
                
            parameters = [
                "-q:a", str(vbr_quality),         # VBR quality level
                "-joint_stereo", "1"              # Use joint stereo
            ]
            print(f"Using VBR encoding with quality level {vbr_quality} for {target_bitrate}kbps")
            
        # Export with optimized settings
        print(f"Exporting MP3 with bitrate: {target_bitrate}kbps")
        print(f"Using FFmpeg parameters: {parameters}")
        audio.export(
            output_path,
            format="mp3",
            bitrate=f"{target_bitrate}k",
            parameters=parameters
        )
        
        # Verify the file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                # Verify the output file details
                output_info = mediainfo(output_path)
                print(f"Output file info: {output_info.get('format_name', 'unknown')}, {output_info.get('duration', 'unknown')}s")
                
                # Check actual bitrate
                if FFPROBE_AVAILABLE and target_bitrate == 320:
                    actual_bitrate = get_mp3_bitrate_with_ffprobe(output_path)
                    print(f"Verified bitrate: {actual_bitrate} kbps (target was {target_bitrate} kbps)")
                    
                    # If bitrate is significantly lower, try direct FFmpeg approach
                    if actual_bitrate and actual_bitrate < target_bitrate * 0.95:
                        print(f"Bitrate is too low ({actual_bitrate}kbps), falling back to direct FFmpeg encoding")
                        return convert_to_mp3_with_ffmpeg(input_path, output_path, target_bitrate)
                
            except Exception as e:
                print(f"Could not verify output details: {e}")
                
            print(f"MP3 conversion successful: {output_path}")
            return True
        else:
            print(f"MP3 output file missing or empty: {output_path}")
            return False
            
    except Exception as e:
        print(f"Error during conversion to MP3: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Clean up potentially partially written output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Cleaned up partial output file: {output_path}")
            except Exception as e_clean:
                print(f"Error cleaning up partial output {output_path}: {e_clean}", file=sys.stderr)
        return False

# Fallback function using direct FFmpeg for more control
def convert_to_mp3_with_ffmpeg(input_path, output_path, target_bitrate):
    """
    Convert audio to MP3 using direct FFmpeg command for maximum control
    This is a fallback for when pydub doesn't achieve the desired bitrate
    """
    print(f"Using direct FFmpeg for MP3 conversion at {target_bitrate}kbps")
    
    try:
        # Remove output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
            
        # Build FFmpeg command for strict CBR encoding
        cmd = [
            "ffmpeg",
            "-y",                           # Overwrite output
            "-i", input_path,               # Input file
            "-c:a", "libmp3lame",           # MP3 codec
            "-b:a", f"{target_bitrate}k",   # Target bitrate
            "-minrate", f"{target_bitrate}k", # Min bitrate
            "-maxrate", f"{target_bitrate}k", # Max bitrate
            "-bufsize", f"{target_bitrate}k", # Buffer size
            "-vbr", "off",                  # Disable VBR
            "-compression_level", "0",      # Max compression
            "-af", "apad=pad_dur=0",        # Avoid end trimming
            output_path                     # Output file
        ]
        
        print(f"FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
        # Verify the file was created and has correct bitrate
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if FFPROBE_AVAILABLE:
                actual_bitrate = get_mp3_bitrate_with_ffprobe(output_path)
                print(f"Verified FFmpeg bitrate: {actual_bitrate} kbps")
                
            print(f"Direct FFmpeg MP3 conversion successful: {output_path}")
            return True
        else:
            print(f"FFmpeg output file missing or empty: {output_path}")
            return False
    
    except Exception as e:
        print(f"Error during direct FFmpeg conversion: {e}")
        return False

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

@app.route('/download-all', methods=['POST'])
def download_all_files():
    """Creates a zip file containing all requested converted files and sends it."""
    try:
        # Get the list of filenames from the request
        data = request.get_json()
        if not data or 'filenames' not in data or not data['filenames']:
            return jsonify({'error': 'No filenames provided'}), 400
            
        filenames = data['filenames']
        print(f"Requested files for ZIP: {filenames}")
        
        # Validate filenames for security (no path traversal)
        for filename in filenames:
            if '..' in filename or filename.startswith('/'):
                return jsonify({'error': 'Invalid filename detected'}), 400
                
            # Also check if the file actually exists
            file_path = os.path.join(app.config['CONVERTED_FOLDER'], secure_filename(filename))
            if not os.path.isfile(file_path):
                return jsonify({'error': f'File not found: {filename}'}), 404
        
        # Create a temporary zip file
        timestamp = int(time.time())
        zip_filename = f"dithero_downloads_{timestamp}.zip"
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create the zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in filenames:
                safe_filename = secure_filename(filename)
                file_path = os.path.join(app.config['CONVERTED_FOLDER'], safe_filename)
                if os.path.isfile(file_path):
                    # Add the file to the zip with just its name (no path)
                    zipf.write(file_path, arcname=safe_filename)
                    print(f"Added {safe_filename} to zip")
                else:
                    print(f"Warning: File not found: {file_path}")
        
        # Send the zip file
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        print(f"Error creating zip file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to create zip file'}), 500

if __name__ == '__main__':
    print(f"Flask starting. Pydub available: {PYDUB_AVAILABLE}")
    # Consider setting debug=False when deploying
    app.run(debug=True, host='0.0.0.0', port=5001) 