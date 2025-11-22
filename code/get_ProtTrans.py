#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prottrans_generator_fixed.py
Author: Custom Fix (Based on user's script)
Description:
    Generates ProtT5 embeddings for all FASTA files in an input directory.
    Uses efficient binary .npy format and includes robust error handling.
"""

import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import torch
import time
import argparse
import os
import gc

# -----------------------------
# ARGUMENT PARSER
# -----------------------------
parser = argparse.ArgumentParser(description="Generate ProtT5 embeddings from FASTA files.")
parser.add_argument("-in", "--path_input", type=str, required=True, help="The path of the input directory containing FASTA files.")
parser.add_argument("-out", "--path_output", type=str, required=True, help="The path of the output directory for ProtTrans files (will use .npy extension).")

# -----------------------------
# GLOBAL SETUP
# -----------------------------
# Initialize device (must be done before loading model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"

# -----------------------------
# UNUSED CODE (Commented out for clean embedding generation)
# -----------------------------
# class ConvNet(torch.nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # ... (rest of the ConvNet definition)
#     def forward(self, x):
#         # ... (rest of the forward pass)
#         pass
# def load_sec_struct_model():
#     # ... (This is not needed for simple embedding generation)
#     pass
# -----------------------------

# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def read_fasta(fasta_path, split_char="!", id_field=0):
    """
    Reads a single FASTA file.
    FIXED: Aggressively removes all internal whitespace from the sequence.
    """
    seq = ''
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if not line.startswith('>'):
                # Aggressively strip, then replace ALL internal whitespace/tabs
                cleaned_line = line.strip().replace(' ', '').replace('\t', '')
                seq += cleaned_line

    seq_id = os.path.splitext(os.path.basename(fasta_path))[0]
    seqs = [(seq_id, seq)]
    return seqs


def get_T5_model(model_name=model_name):
    """Loads the T5 model and tokenizer once."""
    print(f"Loading ProtT5 model: {model_name} on {device}")
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    return model, tokenizer


def get_embeddings(model, tokenizer, seqs, max_residues=4000, max_seq_len=1000, max_batch=100):
    """Processes sequences in batches for embedding generation."""
    results = {"residue_embs": dict()}
    seq_dict = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
    start = time.time()
    batch = []

    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)
        
        # ProtT5 requires sequences to be space-separated: "A M I N O"
        seq_spaced = ' '.join(list(seq))
        batch.append((pdb_id, seq_spaced, seq_len))

        # Check batch conditions: max_batch, max_residues, or last sequence
        n_res_batch = sum([s_len for _, _, s_len in batch])
        # Note: The original code's n_res_batch calculation inside the loop was flawed.
        # This condition checks if adding the current sequence *plus* the current batch size 
        # exceeds the limits, or if the sequence is too long, or if it's the last sequence.
        if len(batch) >= max_batch or n_res_batch + seq_len >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            
            # --- Process Batch ---
            pdb_ids, seqs_spaced, seq_lens = zip(*batch)
            batch = [] # Reset batch

            token_encoding = tokenizer.batch_encode_plus(seqs_spaced, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError as e:
                print(f"RuntimeError during embedding for batch (starting with {pdb_ids[0]}): {e}")
                continue

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if "residue_embs" in results:
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
            # --- End Process Batch ---

    passed_time = time.time() - start
    if results["residue_embs"]:
        avg_time = passed_time / len(results["residue_embs"])
        print(f"\nAverage time per sequence: {avg_time:.2f} s")

    return results


def save_port_map(port_data, output_file):
    """
    FIXED: Uses numpy.save for much faster, smaller, binary file storage.
    The output file extension should be .npy for clarity.
    """
    # Ensure the output file uses the .npy extension
    if output_file.endswith(".prottrans"):
        output_file = output_file.replace(".prottrans", ".npy")
    
    np.save(output_file, port_data)
    print(f"Saved embedding to {output_file} (Shape: {port_data.shape})")


def main(fasta_file, output_file, model, tokenizer):
    """Main function to process a single FASTA file."""
    filename = os.path.splitext(os.path.basename(fasta_file))[0]
    seqs = read_fasta(fasta_file)
    
    # Handle potentially empty sequence set
    if not seqs or not seqs[0][1]:
         raise ValueError(f"FASTA file {fasta_file} contains no valid sequence.")

    results = get_embeddings(model, tokenizer, seqs)
    
    if filename in results['residue_embs']:
        embeddings = results['residue_embs'][filename]
        save_port_map(embeddings, output_file)
    else:
        raise RuntimeError(f"Embedding for {filename} was not generated (possibly skipped due to length or error).")

    
def save_no(path, opath):
    """Logs files that failed to process."""
    f = open('NO_OK.txt', 'a')
    # If the output file ends in .prottrans, log the intended .npy path instead
    if opath.endswith(".prottrans"):
        opath = opath.replace(".prottrans", ".npy")
        
    f.write(path + " > " + opath + "\n")
    f.close()


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # --- Initialize Model and Tokenizer ONCE ---
    model, tokenizer = get_T5_model()
    os.makedirs(args.path_output, exist_ok=True)
    input_files = os.listdir(args.path_input)
    
    j = 0
    print("--- Starting Embedding Generation ---")
    
    for i in input_files:
        if i.endswith(".fasta"):
            fasta_path = os.path.join(args.path_input, i)
            file_name, _ = os.path.splitext(i)
            output_file_base = os.path.join(args.path_output, file_name)
            
            print(f"Processing: {fasta_path}")
            
            try:
                # Pass the loaded model/tokenizer to the main function
                main(fasta_path, output_file_base + ".prottrans", model, tokenizer)
            
            # FIXED: Catch specific Exception and print the error before logging
            except Exception as e:
                print(f"❌ An error occurred during processing {fasta_path}: {e}")
                save_no(fasta_path, output_file_base + ".prottrans") 
            
            j += 1
            # Aggressively collect garbage after each file to help manage GPU memory
            gc.collect() 

    print(f"\n--- Finished. Total FASTA files processed (attempted): {j} ---")