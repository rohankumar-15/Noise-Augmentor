#!/usr/bin/env python3
"""
Audio Augmentor - Simulate telecall quality issues by mixing clean audio with noise.

This script adds realistic background noise to clean audio files while maintaining
speech intelligibility. Designed for creating training data for speech processing systems.
"""

import argparse
import gc
import logging
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
from tqdm import tqdm

# Constants
TARGET_SAMPLE_RATE = 8000
TARGET_BIT_DEPTH = 16
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a'}
MIN_NOISE_DURATION = 2.0  # seconds
RMS_TARGET_DB = -20.0
SILENCE_THRESHOLD_DB = -40.0
INTERMITTENT_CHUNK_MIN = 2.0  # seconds
INTERMITTENT_CHUNK_MAX = 8.0  # seconds
FADE_DURATION_MIN = 0.2  # seconds
FADE_DURATION_MAX = 0.5  # seconds
MIN_DYNAMIC_RANGE_DB = 10.0
MIN_SPEECH_ADVANTAGE_DB = 3.0
CHUNK_DURATION = 300  # 5 minutes in seconds for chunked processing


class NoiseType(Enum):
    CONTINUOUS = "continuous"
    INTERMITTENT = "intermittent"
    MIXED = "mixed"


@dataclass
class AudioFile:
    """Represents an audio file with metadata."""
    path: Path
    duration: float = 0.0
    sample_rate: int = 0
    is_valid: bool = False
    error_message: str = ""


@dataclass
class ProcessingStats:
    """Statistics for the processing run."""
    files_processed: int = 0
    files_failed: int = 0
    total_duration_processed: float = 0.0
    snr_values: List[float] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.snr_values is None:
            self.snr_values = []
        if self.warnings is None:
            self.warnings = []
    
    @property
    def average_snr(self) -> float:
        if not self.snr_values:
            return 0.0
        return sum(self.snr_values) / len(self.snr_values)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def db_to_linear(db: float) -> float:
    """Convert decibels to linear scale."""
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear scale to decibels."""
    if linear <= 0:
        return -np.inf
    return 20 * np.log10(linear)


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS value of audio signal."""
    return np.sqrt(np.mean(audio ** 2))


def calculate_rms_db(audio: np.ndarray) -> float:
    """Calculate RMS value in decibels."""
    rms = calculate_rms(audio)
    return linear_to_db(rms)


def validate_file_path(path: Union[str, Path]) -> Tuple[bool, Path, str]:
    """
    Validate that a file path exists and is readable.
    
    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    try:
        path = Path(path).resolve()
        if not path.exists():
            return False, path, f"File does not exist: {path}"
        if not path.is_file():
            return False, path, f"Path is not a file: {path}"
        if not os.access(path, os.R_OK):
            return False, path, f"File is not readable: {path}"
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return False, path, f"Unsupported format '{path.suffix}'. Supported: {SUPPORTED_FORMATS}"
        return True, path, ""
    except Exception as e:
        return False, Path(path), f"Error validating path: {str(e)}"


def validate_directory(path: Union[str, Path]) -> Tuple[bool, Path, str]:
    """
    Validate that a directory path exists and is readable.
    
    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    try:
        path = Path(path).resolve()
        if not path.exists():
            return False, path, f"Directory does not exist: {path}"
        if not path.is_dir():
            return False, path, f"Path is not a directory: {path}"
        if not os.access(path, os.R_OK):
            return False, path, f"Directory is not readable: {path}"
        return True, path, ""
    except Exception as e:
        return False, Path(path), f"Error validating directory: {str(e)}"


def get_audio_files(path: Union[str, Path], logger: logging.Logger) -> List[Path]:
    """
    Get list of audio files from path (single file or directory).
    
    Returns:
        List of valid audio file paths
    """
    path = Path(path).resolve()
    
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_FORMATS:
            return [path]
        return []
    
    if path.is_dir():
        audio_files = []
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(path.glob(f'*{ext}'))
            audio_files.extend(path.glob(f'*{ext.upper()}'))
        return sorted(set(audio_files))
    
    return []


def load_audio(
    path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[np.ndarray], int, str]:
    """
    Load audio file with resampling and mono conversion.
    
    Returns:
        Tuple of (audio_array, sample_rate, error_message)
    """
    try:
        audio, sr = librosa.load(path, sr=target_sr, mono=mono)
        return audio, sr, ""
    except Exception as e:
        error_msg = f"Failed to load audio file {path}: {str(e)}"
        if logger:
            logger.error(error_msg)
        return None, 0, error_msg


def validate_audio_quality(audio: np.ndarray, is_noise: bool = False) -> Tuple[bool, str]:
    """
    Validate audio quality for processing.
    
    Args:
        audio: Audio signal array
        is_noise: Whether this is a noise file (different validation)
    
    Returns:
        Tuple of (is_valid, warning_or_error_message)
    """
    # Check for completely silent audio
    rms_db = calculate_rms_db(audio)
    if rms_db < -60:
        return False, "Audio is completely silent or near-silent"
    
    # Check for clipping (more than 1% samples at max)
    clipped_samples = np.sum(np.abs(audio) >= 0.99)
    clip_ratio = clipped_samples / len(audio)
    if clip_ratio > 0.01:
        return False, f"Audio is severely clipped ({clip_ratio*100:.1f}% of samples)"
    
    if not is_noise:
        # Check dynamic range for clean audio
        peak_db = linear_to_db(np.max(np.abs(audio)))
        dynamic_range = peak_db - rms_db
        if dynamic_range < MIN_DYNAMIC_RANGE_DB:
            return True, f"Warning: Low dynamic range ({dynamic_range:.1f} dB). Audio may already be degraded."
    
    return True, ""


def remove_silence(
    audio: np.ndarray,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.
    
    Args:
        audio: Audio signal array
        threshold_db: Silence threshold in dB
        sr: Sample rate
    
    Returns:
        Trimmed audio array
    """
    threshold = db_to_linear(threshold_db)
    
    # Use librosa's trim function
    trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))
    
    return trimmed


def normalize_rms(audio: np.ndarray, target_db: float = RMS_TARGET_DB) -> np.ndarray:
    """
    Normalize audio to target RMS level.
    
    Args:
        audio: Audio signal array
        target_db: Target RMS level in dB
    
    Returns:
        Normalized audio array
    """
    current_rms = calculate_rms(audio)
    if current_rms < 1e-10:
        return audio  # Avoid division by zero for silent audio
    
    target_rms = db_to_linear(target_db)
    scale_factor = target_rms / current_rms
    
    return audio * scale_factor


def segment_noise(
    noise: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    min_duration: float = INTERMITTENT_CHUNK_MIN,
    max_duration: float = INTERMITTENT_CHUNK_MAX
) -> List[np.ndarray]:
    """
    Segment noise into random chunks for intermittent application.
    
    Args:
        noise: Noise audio array
        sr: Sample rate
        min_duration: Minimum chunk duration in seconds
        max_duration: Maximum chunk duration in seconds
    
    Returns:
        List of noise segments
    """
    segments = []
    total_samples = len(noise)
    min_samples = int(min_duration * sr)
    max_samples = int(max_duration * sr)
    
    pos = 0
    while pos < total_samples:
        # Random segment length
        segment_length = random.randint(min_samples, max_samples)
        end_pos = min(pos + segment_length, total_samples)
        
        if end_pos - pos >= min_samples:
            segments.append(noise[pos:end_pos])
        
        pos = end_pos
    
    return segments


def create_fade_envelope(length: int, fade_in: int, fade_out: int) -> np.ndarray:
    """
    Create fade-in/fade-out envelope.
    
    Args:
        length: Total length of envelope
        fade_in: Number of samples for fade-in
        fade_out: Number of samples for fade-out
    
    Returns:
        Envelope array
    """
    envelope = np.ones(length)
    
    if fade_in > 0:
        envelope[:fade_in] = np.linspace(0, 1, fade_in)
    
    if fade_out > 0:
        envelope[-fade_out:] = np.linspace(1, 0, fade_out)
    
    return envelope


def apply_soft_limiting(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Apply soft limiting to prevent clipping.
    
    Uses tanh-based soft clipping for natural sound.
    
    Args:
        audio: Audio signal array
        threshold: Threshold above which limiting kicks in
    
    Returns:
        Limited audio array
    """
    # Check if limiting is needed
    peak = np.max(np.abs(audio))
    if peak <= 1.0:
        return audio
    
    # Apply tanh-based soft limiting
    # Scale so that threshold maps to desired output
    scale = threshold / peak
    limited = np.tanh(audio * scale * 1.5) / np.tanh(1.5)
    
    return limited


def calculate_snr_scale(clean_rms: float, noise_rms: float, target_snr_db: float) -> float:
    """
    Calculate scale factor for noise to achieve target SNR.
    
    SNR = 20 * log10(signal_rms / noise_rms)
    Therefore: noise_scale = signal_rms / (noise_rms * 10^(snr/20))
    
    Args:
        clean_rms: RMS of clean signal
        noise_rms: RMS of noise signal
        target_snr_db: Target SNR in dB
    
    Returns:
        Scale factor to apply to noise
    """
    if noise_rms < 1e-10:
        return 0.0
    
    target_noise_rms = clean_rms / db_to_linear(target_snr_db)
    return target_noise_rms / noise_rms


def prepare_noise_continuous(
    noise: np.ndarray,
    target_length: int,
    sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """
    Prepare continuous noise to match target length.
    
    Loops noise if shorter, truncates if longer.
    
    Args:
        noise: Noise audio array
        target_length: Target length in samples
        sr: Sample rate
    
    Returns:
        Noise array matching target length
    """
    if len(noise) >= target_length:
        return noise[:target_length]
    
    # Loop noise with crossfade to avoid clicks
    result = np.zeros(target_length)
    pos = 0
    crossfade_samples = int(0.1 * sr)  # 100ms crossfade
    
    while pos < target_length:
        chunk_len = min(len(noise), target_length - pos)
        
        if pos == 0:
            result[pos:pos + chunk_len] = noise[:chunk_len]
        else:
            # Apply crossfade at loop point
            if crossfade_samples > 0 and pos >= crossfade_samples:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                result[pos - crossfade_samples:pos] *= fade_out
                result[pos - crossfade_samples:pos] += noise[:crossfade_samples] * fade_in
                
            result[pos:pos + chunk_len] = noise[:chunk_len]
        
        pos += len(noise) - crossfade_samples
    
    return result[:target_length]


def prepare_noise_intermittent(
    noise_segments: List[np.ndarray],
    target_length: int,
    sr: int = TARGET_SAMPLE_RATE,
    coverage: float = 0.3  # 30% of audio will have noise
) -> np.ndarray:
    """
    Prepare intermittent noise with random placement.
    
    Args:
        noise_segments: List of noise segment arrays
        target_length: Target length in samples
        sr: Sample rate
        coverage: Fraction of audio to cover with noise
    
    Returns:
        Noise array with intermittent segments
    """
    result = np.zeros(target_length)
    
    if not noise_segments:
        return result
    
    # Calculate how many segments to place
    total_noise_samples = int(target_length * coverage)
    placed_samples = 0
    
    # Random fade duration
    fade_samples = int(random.uniform(FADE_DURATION_MIN, FADE_DURATION_MAX) * sr)
    
    attempts = 0
    max_attempts = 100
    
    while placed_samples < total_noise_samples and attempts < max_attempts:
        attempts += 1
        
        # Select random segment
        segment = random.choice(noise_segments)
        segment_len = len(segment)
        
        # Random start position
        max_start = target_length - segment_len
        if max_start <= 0:
            continue
        
        start_pos = random.randint(0, max_start)
        
        # Create envelope for this segment
        envelope = create_fade_envelope(segment_len, fade_samples, fade_samples)
        
        # Add segment with envelope
        result[start_pos:start_pos + segment_len] += segment * envelope
        placed_samples += segment_len
    
    return result


def mix_audio(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, float]:
    """
    Mix clean audio with noise at specified SNR.
    
    Args:
        clean: Clean audio array
        noise: Noise audio array (same length as clean)
        snr_db: Target SNR in dB
        logger: Optional logger
    
    Returns:
        Tuple of (mixed_audio, actual_snr_db)
    """
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)
    
    if noise_rms < 1e-10:
        if logger:
            logger.warning("Noise has near-zero RMS, returning clean audio")
        return clean.copy(), np.inf
    
    # Calculate scale for target SNR
    noise_scale = calculate_snr_scale(clean_rms, noise_rms, snr_db)
    
    # Scale noise
    scaled_noise = noise * noise_scale
    
    # Mix
    mixed = clean + scaled_noise
    
    # Apply soft limiting if needed
    if np.max(np.abs(mixed)) > 1.0:
        mixed = apply_soft_limiting(mixed)
        if logger:
            logger.debug("Applied soft limiting to prevent clipping")
    
    # Calculate actual SNR achieved
    actual_noise_rms = calculate_rms(scaled_noise)
    if actual_noise_rms > 1e-10:
        actual_snr = linear_to_db(clean_rms / actual_noise_rms)
    else:
        actual_snr = np.inf
    
    return mixed, actual_snr


def validate_speech_intelligibility(
    clean: np.ndarray,
    mixed: np.ndarray,
    min_advantage_db: float = MIN_SPEECH_ADVANTAGE_DB
) -> Tuple[bool, str]:
    """
    Validate that speech remains intelligible after mixing.
    
    Uses a simple energy-based check to ensure speech segments
    maintain minimum advantage over noise.
    
    Args:
        clean: Original clean audio
        mixed: Mixed audio with noise
        min_advantage_db: Minimum required speech advantage
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Calculate correlation between clean and mixed
    # High correlation indicates speech is preserved
    if len(clean) != len(mixed):
        return False, "Length mismatch between clean and mixed audio"
    
    # Normalize for comparison
    clean_norm = clean / (np.max(np.abs(clean)) + 1e-10)
    mixed_norm = mixed / (np.max(np.abs(mixed)) + 1e-10)
    
    correlation = np.corrcoef(clean_norm, mixed_norm)[0, 1]
    
    if correlation < 0.7:  # Correlation threshold
        return False, f"Low correlation ({correlation:.2f}) suggests speech may not be intelligible"
    
    # Check that speech segments maintain energy advantage
    # Simple check: RMS of mixed should be reasonably close to clean
    clean_rms_db = calculate_rms_db(clean)
    mixed_rms_db = calculate_rms_db(mixed)
    
    rms_diff = abs(mixed_rms_db - clean_rms_db)
    if rms_diff > 6:  # If RMS changed by more than 6dB, something's wrong
        return False, f"Large RMS change ({rms_diff:.1f} dB) after mixing"
    
    return True, f"Speech intelligibility verified (correlation: {correlation:.2f})"


def save_audio(
    audio: np.ndarray,
    path: Path,
    sr: int = TARGET_SAMPLE_RATE,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """
    Save audio to WAV file.
    
    Args:
        audio: Audio array to save
        path: Output path
        sr: Sample rate
        logger: Optional logger
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        import soundfile as sf
        
        # Ensure output directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to 16-bit integer
        audio_int = np.clip(audio, -1.0, 1.0)
        audio_int = (audio_int * 32767).astype(np.int16)
        
        sf.write(str(path), audio_int, sr, subtype='PCM_16')
        
        return True, ""
    except ImportError:
        # Fallback to scipy if soundfile not available
        try:
            from scipy.io import wavfile
            audio_int = np.clip(audio, -1.0, 1.0)
            audio_int = (audio_int * 32767).astype(np.int16)
            wavfile.write(str(path), sr, audio_int)
            return True, ""
        except Exception as e:
            return False, f"Failed to save audio: {str(e)}"
    except Exception as e:
        return False, f"Failed to save audio: {str(e)}"


def preprocess_noise_file(
    path: Path,
    sr: int = TARGET_SAMPLE_RATE,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[np.ndarray], str]:
    """
    Load and preprocess a noise file.
    
    Args:
        path: Path to noise file
        sr: Target sample rate
        logger: Optional logger
    
    Returns:
        Tuple of (preprocessed_noise, error_message)
    """
    # Load audio
    audio, loaded_sr, error = load_audio(path, sr, mono=True, logger=logger)
    if audio is None:
        return None, error
    
    # Check duration
    duration = len(audio) / sr
    if duration < MIN_NOISE_DURATION:
        return None, f"Noise file too short ({duration:.1f}s). Minimum: {MIN_NOISE_DURATION}s"
    
    # Validate quality
    is_valid, message = validate_audio_quality(audio, is_noise=True)
    if not is_valid:
        return None, message
    
    # Remove silence
    audio = remove_silence(audio, SILENCE_THRESHOLD_DB, sr)
    
    # Check duration again after trimming
    duration = len(audio) / sr
    if duration < MIN_NOISE_DURATION:
        return None, f"Noise file too short after trimming ({duration:.1f}s)"
    
    # Normalize RMS
    audio = normalize_rms(audio, RMS_TARGET_DB)
    
    return audio, ""


def process_single_file(
    clean_path: Path,
    noise_files: List[Tuple[Path, np.ndarray]],
    output_dir: Path,
    noise_type: NoiseType,
    snr_min: float,
    snr_max: float,
    sr: int = TARGET_SAMPLE_RATE,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str, float]:
    """
    Process a single clean audio file with noise augmentation.
    
    Args:
        clean_path: Path to clean audio file
        noise_files: List of (path, preprocessed_noise) tuples
        output_dir: Output directory
        noise_type: Type of noise application
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        sr: Sample rate
        logger: Optional logger
    
    Returns:
        Tuple of (success, message, snr_used)
    """
    if logger:
        logger.debug(f"Processing: {clean_path.name}")
    
    # Load clean audio
    clean, _, error = load_audio(clean_path, sr, mono=True, logger=logger)
    if clean is None:
        return False, error, 0.0
    
    # Validate clean audio quality
    is_valid, message = validate_audio_quality(clean, is_noise=False)
    if not is_valid:
        return False, message, 0.0
    if message and logger:
        logger.warning(f"{clean_path.name}: {message}")
    
    clean_length = len(clean)
    clean_duration = clean_length / sr
    
    # Select random noise file
    noise_path, noise = random.choice(noise_files)
    
    # Select random SNR
    target_snr = random.uniform(snr_min, snr_max)
    
    if logger:
        logger.debug(f"Using noise: {noise_path.name}, Target SNR: {target_snr:.1f} dB")
    
    # Prepare noise based on type
    if noise_type == NoiseType.CONTINUOUS:
        prepared_noise = prepare_noise_continuous(noise, clean_length, sr)
    elif noise_type == NoiseType.INTERMITTENT:
        segments = segment_noise(noise, sr)
        prepared_noise = prepare_noise_intermittent(segments, clean_length, sr)
    else:  # MIXED
        # Apply both continuous (at lower level) and intermittent
        continuous = prepare_noise_continuous(noise, clean_length, sr) * 0.5
        
        # Use different noise for intermittent if available
        if len(noise_files) > 1:
            other_noise_path, other_noise = random.choice(
                [(p, n) for p, n in noise_files if p != noise_path]
            )
            segments = segment_noise(other_noise, sr)
        else:
            segments = segment_noise(noise, sr)
        
        intermittent = prepare_noise_intermittent(segments, clean_length, sr)
        prepared_noise = continuous + intermittent
        
        # Re-normalize combined noise
        prepared_noise = normalize_rms(prepared_noise, RMS_TARGET_DB)
    
    # Mix audio
    mixed, actual_snr = mix_audio(clean, prepared_noise, target_snr, logger)
    
    # Validate speech intelligibility
    is_valid, intel_message = validate_speech_intelligibility(clean, mixed)
    if not is_valid:
        # Try with higher SNR
        if logger:
            logger.warning(f"{clean_path.name}: {intel_message}. Retrying with higher SNR.")
        
        higher_snr = min(target_snr + 5, snr_max + 5)
        mixed, actual_snr = mix_audio(clean, prepared_noise, higher_snr, logger)
        
        is_valid, intel_message = validate_speech_intelligibility(clean, mixed)
        if not is_valid:
            return False, f"Could not maintain speech intelligibility: {intel_message}", 0.0
    
    # Generate output filename
    noise_type_str = noise_type.value
    output_filename = f"{clean_path.stem}_aug_{noise_type_str}.wav"
    output_path = output_dir / output_filename
    
    # Save output
    success, error = save_audio(mixed, output_path, sr, logger)
    if not success:
        return False, error, 0.0
    
    # Clean up
    del clean, prepared_noise, mixed
    gc.collect()
    
    return True, str(output_path), actual_snr


def process_chunked(
    clean_path: Path,
    noise_files: List[Tuple[Path, np.ndarray]],
    output_dir: Path,
    noise_type: NoiseType,
    snr_min: float,
    snr_max: float,
    sr: int = TARGET_SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str, float]:
    """
    Process a long audio file in chunks to manage memory.
    
    Args:
        clean_path: Path to clean audio file
        noise_files: List of (path, preprocessed_noise) tuples
        output_dir: Output directory
        noise_type: Type of noise application
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        logger: Optional logger
    
    Returns:
        Tuple of (success, message, snr_used)
    """
    if logger:
        logger.debug(f"Processing in chunks: {clean_path.name}")
    
    # Get total duration first
    try:
        total_duration = librosa.get_duration(path=str(clean_path))
    except Exception as e:
        return False, f"Could not get duration: {str(e)}", 0.0
    
    if total_duration <= chunk_duration:
        # Not actually long, process normally
        return process_single_file(
            clean_path, noise_files, output_dir, noise_type,
            snr_min, snr_max, sr, logger
        )
    
    # Select consistent parameters for all chunks
    noise_path, noise = random.choice(noise_files)
    target_snr = random.uniform(snr_min, snr_max)
    
    if logger:
        logger.debug(f"Long file ({total_duration:.1f}s), processing in {chunk_duration}s chunks")
    
    # Process in chunks and concatenate
    output_chunks = []
    chunk_samples = int(chunk_duration * sr)
    offset = 0
    
    while offset < total_duration:
        # Load chunk
        try:
            chunk, _ = librosa.load(
                str(clean_path),
                sr=sr,
                mono=True,
                offset=offset,
                duration=chunk_duration
            )
        except Exception as e:
            return False, f"Failed to load chunk at {offset}s: {str(e)}", 0.0
        
        chunk_length = len(chunk)
        
        # Prepare noise for this chunk
        if noise_type == NoiseType.CONTINUOUS:
            prepared_noise = prepare_noise_continuous(noise, chunk_length, sr)
        elif noise_type == NoiseType.INTERMITTENT:
            segments = segment_noise(noise, sr)
            prepared_noise = prepare_noise_intermittent(segments, chunk_length, sr)
        else:
            continuous = prepare_noise_continuous(noise, chunk_length, sr) * 0.5
            segments = segment_noise(noise, sr)
            intermittent = prepare_noise_intermittent(segments, chunk_length, sr)
            prepared_noise = continuous + intermittent
            prepared_noise = normalize_rms(prepared_noise, RMS_TARGET_DB)
        
        # Mix
        mixed_chunk, _ = mix_audio(chunk, prepared_noise, target_snr, logger)
        output_chunks.append(mixed_chunk)
        
        offset += chunk_duration
        
        # Clean up
        del chunk, prepared_noise, mixed_chunk
        gc.collect()
    
    # Concatenate all chunks
    full_output = np.concatenate(output_chunks)
    
    # Generate output filename
    noise_type_str = noise_type.value
    output_filename = f"{clean_path.stem}_aug_{noise_type_str}.wav"
    output_path = output_dir / output_filename
    
    # Save output
    success, error = save_audio(full_output, output_path, sr, logger)
    
    # Clean up
    del output_chunks, full_output
    gc.collect()
    
    if not success:
        return False, error, 0.0
    
    return True, str(output_path), target_snr


def run_augmentation(
    clean_input: Union[str, Path],
    noise_input: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    noise_type: str = "continuous",
    snr_min: float = 5.0,
    snr_max: float = 15.0,
    dry_run: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None
) -> ProcessingStats:
    """
    Main function to run audio augmentation.
    
    Args:
        clean_input: Path to clean audio file or directory
        noise_input: Path to noise audio file or directory
        output_dir: Output directory (optional)
        noise_type: Type of noise application
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        dry_run: If True, validate without processing
        verbose: Enable verbose logging
        seed: Random seed for reproducibility
    
    Returns:
        ProcessingStats object with results
    """
    logger = setup_logging(verbose)
    stats = ProcessingStats()
    
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")
    
    # Parse noise type
    try:
        noise_type_enum = NoiseType(noise_type.lower())
    except ValueError:
        logger.error(f"Invalid noise type: {noise_type}. Valid: continuous, intermittent, mixed")
        return stats
    
    # Validate SNR range
    if snr_min < -30 or snr_max > 30:
        logger.error("SNR range should be between -30 and +30 dB")
        return stats
    if snr_min > snr_max:
        logger.error("Minimum SNR cannot be greater than maximum SNR")
        return stats
    
    # Resolve input paths
    clean_input = Path(clean_input).resolve()
    noise_input = Path(noise_input).resolve()
    
    # Validate inputs exist
    if not clean_input.exists():
        logger.error(f"Clean input does not exist: {clean_input}")
        return stats
    if not noise_input.exists():
        logger.error(f"Noise input does not exist: {noise_input}")
        return stats
    
    # Get audio files
    clean_files = get_audio_files(clean_input, logger)
    noise_files_paths = get_audio_files(noise_input, logger)
    
    if not clean_files:
        logger.error(f"No valid audio files found in: {clean_input}")
        return stats
    if not noise_files_paths:
        logger.error(f"No valid audio files found in: {noise_input}")
        return stats
    
    logger.info(f"Found {len(clean_files)} clean audio file(s)")
    logger.info(f"Found {len(noise_files_paths)} noise audio file(s)")
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir).resolve()
    elif clean_input.is_dir():
        output_path = clean_input / "augmented"
    else:
        output_path = clean_input.parent
    
    logger.info(f"Output directory: {output_path}")
    
    # Pre-flight validation of all files
    logger.info("Validating input files...")
    
    valid_clean_files = []
    for path in clean_files:
        is_valid, resolved, error = validate_file_path(path)
        if is_valid:
            valid_clean_files.append(resolved)
        else:
            logger.warning(f"Skipping invalid clean file: {error}")
            stats.files_failed += 1
    
    if not valid_clean_files:
        logger.error("No valid clean audio files to process")
        return stats
    
    # Preprocess noise files
    logger.info("Preprocessing noise files...")
    noise_files: List[Tuple[Path, np.ndarray]] = []
    
    for path in tqdm(noise_files_paths, desc="Loading noise", disable=not verbose):
        noise_audio, error = preprocess_noise_file(path, TARGET_SAMPLE_RATE, logger)
        if noise_audio is not None:
            noise_files.append((path, noise_audio))
        else:
            logger.warning(f"Skipping noise file {path.name}: {error}")
    
    if not noise_files:
        logger.error("No valid noise files after preprocessing")
        return stats
    
    logger.info(f"Successfully preprocessed {len(noise_files)} noise file(s)")
    
    # Dry run mode
    if dry_run:
        logger.info("\n=== DRY RUN MODE ===")
        logger.info(f"Would process {len(valid_clean_files)} clean file(s)")
        logger.info(f"Using {len(noise_files)} noise file(s)")
        logger.info(f"Noise type: {noise_type_enum.value}")
        logger.info(f"SNR range: {snr_min:.1f} to {snr_max:.1f} dB")
        logger.info(f"Output directory: {output_path}")
        
        for i, f in enumerate(valid_clean_files[:10], 1):
            logger.info(f"  {i}. {f.name}")
        if len(valid_clean_files) > 10:
            logger.info(f"  ... and {len(valid_clean_files) - 10} more")
        
        return stats
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process files
    logger.info(f"Processing {len(valid_clean_files)} file(s)...")
    
    for clean_path in tqdm(valid_clean_files, desc="Augmenting"):
        try:
            # Check if file is long (>5 minutes)
            duration = librosa.get_duration(path=str(clean_path))
            
            if duration > CHUNK_DURATION:
                success, message, snr = process_chunked(
                    clean_path, noise_files, output_path, noise_type_enum,
                    snr_min, snr_max, TARGET_SAMPLE_RATE, CHUNK_DURATION, logger
                )
            else:
                success, message, snr = process_single_file(
                    clean_path, noise_files, output_path, noise_type_enum,
                    snr_min, snr_max, TARGET_SAMPLE_RATE, logger
                )
            
            if success:
                stats.files_processed += 1
                stats.snr_values.append(snr)
                stats.total_duration_processed += duration
                if verbose:
                    logger.debug(f"Created: {message} (SNR: {snr:.1f} dB)")
            else:
                stats.files_failed += 1
                stats.warnings.append(f"{clean_path.name}: {message}")
                logger.warning(f"Failed to process {clean_path.name}: {message}")
                
        except Exception as e:
            stats.files_failed += 1
            stats.warnings.append(f"{clean_path.name}: {str(e)}")
            logger.error(f"Error processing {clean_path.name}: {str(e)}")
        
        # Garbage collection after each file
        gc.collect()
    
    # Clean up noise arrays
    del noise_files
    gc.collect()
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Files processed successfully: {stats.files_processed}")
    logger.info(f"Files failed: {stats.files_failed}")
    logger.info(f"Total duration processed: {stats.total_duration_processed:.1f} seconds")
    
    if stats.snr_values:
        logger.info(f"Average SNR achieved: {stats.average_snr:.1f} dB")
        logger.info(f"SNR range: {min(stats.snr_values):.1f} to {max(stats.snr_values):.1f} dB")
    
    logger.info(f"Output saved to: {output_path}")
    
    if stats.warnings:
        logger.info(f"\nWarnings ({len(stats.warnings)}):")
        for warning in stats.warnings[:5]:
            logger.info(f"  - {warning}")
        if len(stats.warnings) > 5:
            logger.info(f"  ... and {len(stats.warnings) - 5} more warnings")
    
    return stats


def main():
    """Command-line interface for audio augmentor."""
    parser = argparse.ArgumentParser(
        description="Add realistic background noise to clean audio files for telecall simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with single noise
  python augmentor.py -c clean.wav -n noise.wav

  # Directory of files with directory of noise
  python augmentor.py -c ./clean_audio/ -n ./noise_samples/

  # Specify SNR range and noise type
  python augmentor.py -c audio.wav -n noise.wav --snr-min 8 --snr-max 12 --noise-type intermittent

  # Dry run to preview
  python augmentor.py -c ./audio/ -n ./noise/ --dry-run

  # Verbose with reproducible results
  python augmentor.py -c audio.wav -n noise.wav -v --seed 42
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-c', '--clean',
        required=True,
        help='Path to clean audio file or directory containing clean audio files'
    )
    parser.add_argument(
        '-n', '--noise',
        required=True,
        help='Path to noise audio file or directory containing noise audio files'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        help='Output directory (default: "augmented" subdirectory or same directory for single files)'
    )
    parser.add_argument(
        '--noise-type',
        choices=['continuous', 'intermittent', 'mixed'],
        default='continuous',
        help='Type of noise application (default: continuous)'
    )
    parser.add_argument(
        '--snr-min',
        type=float,
        default=5.0,
        help='Minimum SNR in dB (default: 5.0)'
    )
    parser.add_argument(
        '--snr-max',
        type=float,
        default=15.0,
        help='Maximum SNR in dB (default: 15.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs and show processing plan without actual processing'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging with detailed processing information'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    args = parser.parse_args()
    
    # Validate SNR range
    if args.snr_min < -30 or args.snr_max > 30:
        print("Error: SNR range must be between -30 and +30 dB")
        sys.exit(1)
    
    if args.snr_min > args.snr_max:
        print("Error: --snr-min cannot be greater than --snr-max")
        sys.exit(1)
    
    # Run augmentation
    stats = run_augmentation(
        clean_input=args.clean,
        noise_input=args.noise,
        output_dir=args.output,
        noise_type=args.noise_type,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        dry_run=args.dry_run,
        verbose=args.verbose,
        seed=args.seed
    )
    
    # Exit with appropriate code
    if stats.files_failed > 0 and stats.files_processed == 0:
        sys.exit(1)
    elif stats.files_failed > 0:
        sys.exit(2)  # Partial success
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
