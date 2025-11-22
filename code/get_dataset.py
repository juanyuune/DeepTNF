import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="the path of input directory with embedding files")
parser.add_argument("-out", "--path_output", type=str, help="the path of output aggregated .npy file")
parser.add_argument("-dt", "--data_type", type=str, help="the data type of feature (.prottrans, .npy, .esm2, .tape)")
parser.add_argument("-maxseq", "--max_sequence", type=int, default=0, help="the maxseq length for padding/truncating (e.g., 20 or 512)")

def loadData(path):
    # It is usually safer to use mmap_mode='r' for large files if memory is a concern
    Data = np.load(path, allow_pickle=True)
    return Data

def saveData(path, data):
    # The output shape will be printed by the main function
    np.save(path, data)
    print(f"✅ Dataset saved to: {path}")

def get_series_feature(org_data, maxseq, dt):
    """
    Pads or truncates the embedding to a fixed maxseq length and reshapes it.
    """
    # 1. Determine channels and initialize 'data' array
    channels = 0
    if dt == '.prottrans' :
        channels = 1024
    elif dt == '.esm2':
        channels = 1280
    elif dt == '.tape'or dt == '.npy':
        channels = 768
    else:
        # Raise an error if the data type is unrecognized
        raise ValueError(f"Unsupported data type: {dt}. Check your -dt argument.")

    # Initialize 'data' array now that channels is defined
    # This prevents the UnboundLocalError
    data = np.zeros((maxseq, channels), dtype=np.float16)
    
    # 2. Apply padding or truncation
    data_len = len(org_data)
    # The length to copy is the minimum of the sequence length and the maxseq
    copy_len = min(data_len, maxseq)
    
    # Copy the required portion of org_data into the initialized 'data' array
    data[:copy_len, :] = org_data[:copy_len, :]
    
    # print(data.shape) # Removed intermediate print for cleaner output
    
    # 3. Reshape for CNN input (1, 1, max_seq, channels)
    data = data.reshape((1, 1, maxseq, channels))
    
    return data

def main(path_input, path_output, data_type, maxseq):
    if maxseq == 0:
        raise ValueError("The --max_sequence (-maxseq) argument must be greater than 0.")
        
    result = []
    input_files = os.listdir(path_input)
    
    for i in input_files:
        if i.endswith(data_type):
            file_name, _ = os.path.splitext(i)
            file_path = os.path.join(path_input, i)
            
            try:
                # print(f"Processing: {file_path}") # Optional: uncomment for verbose
                data = loadData(file_path)
                result.append(get_series_feature(data, maxseq, data_type))
            except Exception as e:
                print(f"❌ Error processing file {file_path}: {e}")
    
    if not result:
        print(f"❌ Error: No files with extension '{data_type}' found in {path_input}. The final dataset was not created.")
        return
        
    data = np.concatenate(result, axis=0)
    print(f"\n--- Aggregation Complete ---")
    print(f"There are {data.shape[0]} proteins processed.")
    print(f"Final dataset shape: {data.shape}")
    
    saveData(path_output, data)

if __name__ == "__main__":
    args = parser.parse_args()
    path_input = args.path_input
    path_output = args.path_output
    data_type = args.data_type
    maxseq = args.max_sequence
    main(path_input, path_output, data_type, maxseq)