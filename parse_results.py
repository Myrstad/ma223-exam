#!/usr/bin/env python3
"""
Parse Burn training output and save to CSV for R plotting.
Usage: python parse_results.py < training_output.txt > results.csv
"""

import re
import sys

# Match lines like: Epoch 1/50 - Train: 0.5200/0.8100, Valid: 0.4800/0.8300
pattern = r'Epoch (\d+)/(\d+) - Train: ([\d.]+)/([\d.]+), Valid: ([\d.]+)/([\d.]+)'

print("epoch,train_loss,train_acc,valid_loss,valid_acc")

for line in sys.stdin:
    match = re.search(pattern, line)
    if match:
        epoch = match.group(1)
        train_loss = match.group(3)
        train_acc = match.group(4)
        valid_loss = match.group(5)
        valid_acc = match.group(6)
        print(f"{epoch},{train_loss},{train_acc},{valid_loss},{valid_acc}")