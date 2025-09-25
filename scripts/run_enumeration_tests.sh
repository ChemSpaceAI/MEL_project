#!/bin/bash
echo "=== Running MEL Enumeration for All Configs ==="

CONFIGS=("configs/Config_For_Enumeration_bridge.json"
  "configs/Config_For_Enumeration_bridge_iter_2.json"
  "configs/Config_For_Enumeration_3_comp_iter_2.json"
  
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "=== Running config: $CONFIG ==="
  python -m mel_package.3_enumerate --config "$CONFIG"
done
