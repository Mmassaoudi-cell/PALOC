#!/bin/bash
# usage: jobs.sh <jobfile> <parallel> ; each line: args passed to run.py
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cat "$1" | xargs -P "$2" -I{} bash -c 'python run.py {} 2>&1 | grep -E --line-buffered "^\[|Error|Traceback|error" ' 
