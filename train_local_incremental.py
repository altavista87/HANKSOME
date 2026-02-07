"""
Incremental Local Training Script for Malaysia HANK
===================================================
Trains the model in chunks (e.g., 500 epochs) to allow for safe interruption and resumption.
Designed for local execution on Mac (MPS).

Usage:
    python malaysia_hank/train_local_incremental.py
"""

import os
import torch
from malaysia_deep_hank_architecture import MalaysiaExtendedParams, MalaysiaExtendedHANK, HANKTrainer

def run_training_chunk(chunk_size: int = 500, target_epochs: int = 10000, checkpoint_path: str = "hank_checkpoint.pt"):
    """
    Runs training in a loop until target_epochs are reached.
    """
    # Initialize model and trainer
    params = MalaysiaExtendedParams()
    model = MalaysiaExtendedHANK(params)
    trainer = HANKTrainer(model)
    
    # Load existing checkpoint if available
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        # Note: In a real scenario, we'd want to save the epoch number in the checkpoint
        # For now, we'll just load weights and rely on external tracking or manual management
        # To make this robust, let's assume we want to run 'chunk_size' more epochs regardless of where we were
        trainer.load_checkpoint(checkpoint_path)
    else:
        print("Starting fresh training.")

    print(f"Training for {chunk_size} epochs...")
    trainer.train(n_epochs=chunk_size, print_every=50)
    
    # Save checkpoint
    trainer.save_checkpoint(checkpoint_path)
    print("Chunk complete and saved.")

if __name__ == "__main__":
    # Example: Run one chunk of 500 epochs
    # You can run this script multiple times to continue training
    run_training_chunk(chunk_size=500)
