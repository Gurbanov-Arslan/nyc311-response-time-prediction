#!/usr/bin/env python3
import json
import pandas as pd

data = json.load(open('models/detailed_results.json'))
model = list(data.keys())[0]
print(f'First model: {model}')
print(f'Keys: {list(data[model].keys())}')
print(f'Val keys: {list(data[model]["val"].keys())}')

# Check the structure
if 'y_pred' in data[model]["val"]:
    print(f'y_pred shape: {len(data[model]["val"]["y_pred"])}')
if 'y_true' in data[model]["val"]:
    print(f'y_true shape: {len(data[model]["val"]["y_true"])}')
