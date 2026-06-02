#!/bin/bash
mkdir -p models
curl -L -o models/gemma-4-E2B-it-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
