import os
import json
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def load_mixed_channels_to_array(mix_id, target_dir):
    """
    Finds the metadata for a given mix_id, reads all corresponding 
    mono channel audio files, and returns them as a single stacked array.
    
    Returns:
        audio_matrix (np.ndarray): Shape (samples, n_channels)
        metadata (dict): Python dictionary containing sources and mixing matrix
    """
    # 1. Load the metadata file
    metadata_path = os.path.join(target_dir, f"{mix_id}_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    # Determine the number of channels from the column count of the mixing matrix
    mixing_matrix = np.array(metadata["mixing_matrix"])
    n_channels = mixing_matrix.shape[0]
    
    channel_data_list = []
    
    # 2. Iterate through and collect each mono channel file
    channel_filename = f"linear_{mix_id}.wav"
    channel_filepath = os.path.join(target_dir, channel_filename)
    
    data, samplerate = sf.read(channel_filepath)
    channel_data_list.append(data)
    
    # 3. Stack columns horizontally to form shape: (samples, n_channels)
    audio_matrix = np.column_stack(channel_data_list)

    
    return audio_matrix, mixing_matrix, metadata, samplerate


def save_matrix_to_audio(data_matrix, samplerate, output_filepath, target_variance=0.05):
    """
    Takes a multi-channel numpy array, scales each channel to have the exact 
    same target variance, clips values to safe audio boundaries, and saves it.
    
    Args:
        data_matrix (np.ndarray): Audio data of shape (samples, channels) or (samples,).
        samplerate (int): The sample rate of the audio.
        output_filepath (str): The destination file path.
        target_variance (float): The desired variance for every channel. 
                                  Default 0.05 is a sweet spot for speech.
    """
    # 1. Handle both 1D (mono) and 2D (multi-channel) structures uniformly
    is_1d = data_matrix.ndim == 1
    if is_1d:
        # Temporarily convert shape from (samples,) to (samples, 1)
        data_matrix = data_matrix[:, np.newaxis]
        
    # Create a float64 copy to maintain high calculation precision during scaling
    processed_matrix = data_matrix.astype(np.float64, copy=True)
    
    # 2. Iterate through columns (channels) and equalize variance    
    for c in range(processed_matrix.shape[1]):
        col = processed_matrix[:, c]

        current_variance = np.var(col)

        if current_variance > 1e-12:
        
            # Match target variance
            #scale_factor = np.sqrt(target_variance / current_variance)
            #col *= scale_factor
    
            # Optional amplitude normalization
            max_abs = np.max(np.abs(col))
            if max_abs > 1e-12:
                col /= max_abs*2
    
            ## Optional thresholding
            ##threshold = 0.3 * np.max(np.abs(col))
            #col[np.abs(col) < threshold] = 0
    
        
        else:
            print(
                f"Warning: Channel {c} is completely silent. "
                "Skipping variance scaling."
            )
    # 3. Apply a hard clip to enforce structural audio range constraints [-1.0, 1.0]
    # Choosing target_variance=0.05 ensures standard audio/speech distribution 
    # stays within bounds naturally, minimizing destructive peak clipping.
    processed_matrix = np.clip(processed_matrix, -1.0, 1.0)
    
    # 4. Revert back to 1D array if the input was originally a mono track
    if is_1d:
        processed_matrix = np.squeeze(processed_matrix)

    # 5. Write out the final normalized and clipped audio file
    sf.write(output_filepath, processed_matrix, samplerate)
    print(f"Successfully saved variance-normalized audio to: {output_filepath}")



def load_original_sources_from_metadata(mix_id, target_dir):
    """
    Parses a mix JSON metadata file, extracts the original source paths,
    loads each file, aligns them, and returns the pristine source matrix X.
    
    Args:
        mix_id (str): The ID of the mixture (e.g., 'mix_0000').
        target_dir (str): The folder containing the audio and JSON files.
        
    Returns:
        X (np.ndarray): The pristine source matrix of shape (samples, n_src).
        samplerate (int): The sample rate of the audio files.
    """
    # 1. Locate and parse the metadata JSON file
    metadata_path = os.path.join(target_dir, f"{mix_id}_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file missing: {metadata_path}")
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    source_paths = metadata["original_sources"]
    n_src = len(source_paths)
    
    # 2. Determine the minimum length among the original files to align them
    lengths = []
    for path in source_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Source file listed in metadata could not be found on disk: {path}\n"
                f"Ensure your LibriSpeech dataset directory has not been moved."
            )
        lengths.append(sf.info(path).frames)
        
    min_len = min(lengths)
    
    # 3. Read the files and populate the True Source Matrix X
    X = np.zeros((min_len, n_src))
    samplerate = None
    
    for idx, path in enumerate(source_paths):
        # Read up to min_len to match the exact truncation used during mixing
        data, sr = sf.read(path, frames=min_len)
        X[:, idx] = data
        if samplerate is None:
            samplerate = sr
            
    return X, samplerate