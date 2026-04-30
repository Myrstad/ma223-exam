#!/usr/bin/env python3
"""
Aggregate Burn training logs into epoch-level CSV for R plotting.
"""

import os
import re
from pathlib import Path

def parse_log_file(filepath):
    """Parse a Burn log file (value,num_samples format)"""
    total = 0.0
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(',', 1)
            if len(parts) == 2:
                try:
                    value = float(parts[0].strip())
                    num = int(parts[1].strip())
                    total += value * num
                    count += num
                except ValueError:
                    continue
    return total / count if count > 0 else 0.0

base_dir = Path("./results/iam-classification")
train_dir = base_dir / "train"
valid_dir = base_dir / "valid"

# Get all epoch directories
epochs = sorted([int(re.search(r'epoch-(\d+)', d.name).group(1)) 
                 for d in train_dir.iterdir() if d.is_dir()])

import sys

# Hyperparameters (passed as args or hardcoded for now)
lr = 0.0001
batch_size = 64
dropout = 0.3

print("epoch,train_loss,train_acc,valid_loss,valid_acc,lr,batch_size,dropout")

for epoch in epochs:
    train_loss = parse_log_file(train_dir / f"epoch-{epoch}" / "Loss.log")
    train_acc = parse_log_file(train_dir / f"epoch-{epoch}" / "Accuracy.log")
    valid_loss = parse_log_file(valid_dir / f"epoch-{epoch}" / "Loss.log")
    valid_acc = parse_log_file(valid_dir / f"epoch-{epoch}" / "Accuracy.log")
    print(f"{epoch},{train_loss:.6f},{train_acc:.6f},{valid_loss:.6f},{valid_acc:.6f},{lr},{batch_size},{dropout}")