"""
Full Retraining Script for HANKSOME MoE v4
==========================================
Executes a complete training run of 10,000 epochs in 500-epoch chunks.
This ensures the new calibration parameters (Beta=0.86, Edu=5.0x) are fully learned.
"""

import sys
import os

# Add current directory to path so imports work
sys.path.append(os.path.join(os.getcwd(), 'HANKSOME_MoE_v4'))

from train_local_incremental import run_training_chunk

def train_full_model():
    total_epochs = 10000
    chunk_size = 500
    num_chunks = total_epochs // chunk_size
    
    print("="*60)
    print(f"STARTING FULL RETRAINING: {total_epochs} Epochs")
    print(f"Beta = 0.86 (High Impatience)")
    print(f"Tertiary Return = 5.0x (High Inequality)")
    print("="*60)
    
    for i in range(num_chunks):
        print(f"\n[Chunk {i+1}/{num_chunks}] Training epochs {i*chunk_size} to {(i+1)*chunk_size}...")
        run_training_chunk(chunk_size=chunk_size, checkpoint_path="hank_checkpoint.pt")
        
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    train_full_model()
