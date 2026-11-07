"""Verify that vLLM is private and the other runtime packages are public."""
import os
from pathlib import Path
import sys
import torch
import transformers
import vllm
import vllm._C_stable_libtorch

source = Path.cwd() / 'vllm-src'
public = Path(os.environ['QWEN_PROJECT_ROOT']) / 'vllm-env'
private = Path(os.environ['QWEN_PRIVATE_ENV'])
assert Path(sys.prefix) == private, sys.prefix
for module in (vllm, vllm._C_stable_libtorch):
    location = Path(module.__file__).resolve()
    print(module.__name__, location, flush=True)
    assert location.is_relative_to(source.resolve()), location
for module in (torch, transformers):
    location = Path(module.__file__).resolve()
    print(module.__name__, location, flush=True)
    assert location.is_relative_to(public.resolve()), location
print('PRIVATE VLLM / PUBLIC DEPENDENCIES CHECK PASSED', flush=True)
