import os
import glob
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def read_fasta(fasta_path):
    with open(fasta_path) as f:
        lines = f.read().splitlines()
    header = lines[0] if lines else ">unknown"
    seq = "".join(lines[1:]).strip().upper()
    return header, seq

def write_fasta(header, sequence, out_path):
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for i in range(0, len(sequence), 60):
            f.write(sequence[i:i + 60] + "\n")

def is_valid_sequence(seq, min_len=30):
    """Basic check for valid amino acids."""
    return len(seq) >= min_len and all(a in VALID_AMINO_ACIDS for a in seq)

def generate_candidates(model, tokenizer, input_seq, num_candidates=12, max_new_tokens=100):
    """Generate candidate sequences using ProLLaMA."""
    input_seq = input_seq[:512]  # truncate if too long
    prompt = f"Generate a biologically functional protein similar to this sequence but not identical:\n{input_seq}\nNew protein:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    config = GenerationConfig(
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=max_new_tokens
    )

    outputs = []
    for _ in range(num_candidates):
        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=config)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract only new part (after prompt)
        if "New protein:" in text:
            text = text.split("New protein:")[-1]
        seq = "".join([c for c in text if c.isalpha()]).upper()
        if is_valid_sequence(seq):
            outputs.append(seq)
        else:
            outputs.append("ACDEFGHIKLMNPQRSTVWY")  # fallback if generation fails
    return outputs

def main(input_folder="./input_fasta", output_folder="./output_candidates", model_name="GreatCaptainNemo/ProLLaMA"):
    os.makedirs(output_folder, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    model.eval()

    fasta_files = glob.glob(os.path.join(input_folder, "*.fasta")) + glob.glob(os.path.join(input_folder, "*.fa"))

    for fasta_path in fasta_files:
        base = Path(fasta_path).stem
        header, seq = read_fasta(fasta_path)
        print(f"\nProcessing {base} ({len(seq)} aa)")

        try:
            candidates = generate_candidates(model, tokenizer, seq, num_candidates=12, max_new_tokens=len(seq))
        except Exception as e:
            print(f"⚠️ Error generating for {base}: {e}")
            continue

        # Save all 12 generated sequences directly into output folder
        for i, cand in enumerate(candidates, 1):
            out_path = os.path.join(output_folder, f"{base}_cand{i}.fasta")
            write_fasta(f">{base}_cand{i}", cand, out_path)

        print(f"✅ Saved 12 new candidates for {base} in {output_folder}")

if __name__ == "__main__":
    # 👇 Set your own input/output paths
    input_folder = "C:/jupyter/juan/TNF/FASTA/data/pos_train"
    output_folder = "C:/jupyter/juan/TNF/FASTA/prollma_data"

    main(input_folder=input_folder, output_folder=output_folder)
