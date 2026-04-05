# Troubleshooting

## Ollama Not Running

```bash
# Start Ollama server
ollama serve

# Check status
curl http://localhost:11434/api/version
```

## Model Not Found

```bash
# Pull required model
ollama pull llama2:7b
```

## CUDA/GPU Issues (Jetson)

```bash
# Check GPU status
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Hailo NPU Issues (Raspberry Pi)

```bash
# Check Hailo device status
hailortcli fw-control identify

# List available backends
python -m benchmark backends

# List Hailo-supported models
python -m benchmark list-models --backend hailo

# Check HailoRT version
hailortcli --version

# View Hailo device info
hailortcli scan
```

**Common Hailo Issues:**

| Issue | Solution |
|-------|----------|
| "Hailo device not found" | Check PCIe connection, run `lspci \| grep Hailo` |
| "HailoRT not installed" | Install HailoRT SDK from Hailo Developer Zone |
| "Model compilation failed" | Check disk space, ensure Dataflow Compiler is installed |
| "Unsupported task" | Use detection or classification only (see [Hailo docs](hailo.md)) |
| "CPU fallback error" | This is expected - Hailo backend requires NPU hardware |

## Insufficient Memory

- Use smaller model sizes (n, s)
- Run workloads separately instead of `all`
- Close other applications
