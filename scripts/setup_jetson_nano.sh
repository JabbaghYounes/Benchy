#!/bin/bash
# Edge AI Benchmark Suite - Jetson Nano Setup Script
# Platform: NVIDIA Jetson Nano Developer Kit
#
# This script sets up the complete benchmark environment on a Jetson Nano.
# It is idempotent and can be run multiple times safely.
#
# Requirements:
#   - NVIDIA Jetson Nano Developer Kit
#   - JetPack 4.6+ installed
#   - Internet connection
#   - Sufficient storage (at least 20GB free recommended)
#
# Usage: sudo ./setup_jetson_nano.sh [--pull-models]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/venv"
LOG_FILE="${PROJECT_ROOT}/setup_jetson_nano.log"
PULL_MODELS=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

info() { log "INFO" "${BLUE}$*${NC}"; }
success() { log "SUCCESS" "${GREEN}$*${NC}"; }
warn() { log "WARN" "${YELLOW}$*${NC}"; }
error() { log "ERROR" "${RED}$*${NC}"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pull-models)
            PULL_MODELS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--pull-models]"
            echo ""
            echo "Options:"
            echo "  --pull-models    Download YOLO and LLM models after setup"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Platform detection
detect_platform() {
    info "Detecting platform..."

    # Check for Jetson-specific files
    if [[ ! -f /etc/nv_tegra_release ]]; then
        error "This script is intended for NVIDIA Jetson devices."
        error "Could not find /etc/nv_tegra_release"
        exit 1
    fi

    # Parse Jetson information
    local tegra_release
    tegra_release=$(cat /etc/nv_tegra_release)
    info "Tegra release: $tegra_release"

    # Check for Jetson Nano specifically
    if command -v jetson_release &> /dev/null; then
        info "Jetson release info:"
        jetson_release 2>/dev/null || true
    fi

    # Verify CUDA availability
    if [[ -d /usr/local/cuda ]]; then
        local cuda_version
        cuda_version=$(cat /usr/local/cuda/version.txt 2>/dev/null || echo "unknown")
        info "CUDA version: $cuda_version"
    else
        warn "CUDA not found at /usr/local/cuda"
    fi

    success "Platform detected: NVIDIA Jetson"
}

# Check if running as root (required for some operations)
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        warn "Not running as root. Some operations may require sudo."
    fi
}

# Install system dependencies
install_system_deps() {
    info "Installing system dependencies..."

    # Update package lists
    sudo apt-get update

    # Install required packages
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        curl \
        wget \
        build-essential \
        cmake \
        pkg-config \
        libhdf5-dev \
        libhdf5-serial-dev \
        hdf5-tools \
        libatlas-base-dev \
        libjpeg-dev \
        libpng-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libopenblas-base \
        libopenmpi-dev \
        libomp-dev \
        libfreetype6-dev \
        libjpeg8-dev \
        zlib1g-dev

    success "System dependencies installed"
}

# Set up Jetson performance mode
setup_jetson_performance() {
    info "Configuring Jetson performance settings..."

    # Set to maximum performance mode
    if command -v nvpmodel &> /dev/null; then
        info "Setting power mode to MAXN (maximum performance)..."
        sudo nvpmodel -m 0 2>/dev/null || warn "Could not set power mode"

        # Show current mode
        nvpmodel -q 2>/dev/null || true
    fi

    # Enable all CPU cores at maximum frequency
    if command -v jetson_clocks &> /dev/null; then
        info "Enabling maximum clock speeds..."
        sudo jetson_clocks 2>/dev/null || warn "Could not set jetson_clocks"
    fi

    success "Jetson performance configured"
}

# Create Python virtual environment
setup_venv() {
    info "Setting up Python virtual environment..."

    if [[ -d "$VENV_DIR" ]]; then
        info "Virtual environment already exists at $VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
        success "Virtual environment created at $VENV_DIR"
    fi

    # Activate and upgrade pip
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip wheel setuptools

    success "Python virtual environment ready"
}

# Install Python dependencies
install_python_deps() {
    info "Installing Python dependencies..."

    source "$VENV_DIR/bin/activate"

    # Install numpy first (required for many packages)
    pip install numpy

    # Install PyTorch for Jetson (use NVIDIA's wheel)
    # Check JetPack version to determine correct wheel
    local jetpack_version
    jetpack_version=$(cat /etc/nv_tegra_release | grep -oP 'R\d+' | head -1 || echo "R32")

    info "Detected JetPack base: $jetpack_version"

    # Install torch from NVIDIA's index for JetPack 4.x / 5.x compatibility
    if [[ "$jetpack_version" == "R32" ]] || [[ "$jetpack_version" == "R34" ]]; then
        # JetPack 4.x series
        info "Installing PyTorch for JetPack 4.x..."
        pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu102
    elif [[ "$jetpack_version" == "R35" ]] || [[ "$jetpack_version" == "R36" ]]; then
        # JetPack 5.x / 6.x series
        info "Installing PyTorch for JetPack 5.x/6.x..."
        pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    else
        warn "Unknown JetPack version, attempting standard PyTorch install..."
        pip install torch torchvision
    fi

    # Install Ultralytics YOLO
    info "Installing Ultralytics..."
    pip install ultralytics

    # Install benchmark suite dependencies
    info "Installing benchmark dependencies..."
    pip install \
        psutil \
        requests \
        pyyaml \
        numpy \
        opencv-python-headless

    # Install the benchmark package itself
    if [[ -f "$PROJECT_ROOT/setup.py" ]] || [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        info "Installing benchmark package..."
        pip install -e "$PROJECT_ROOT"
    fi

    success "Python dependencies installed"
}

# Install Ollama
install_ollama() {
    info "Installing Ollama..."

    if command -v ollama &> /dev/null; then
        info "Ollama already installed"
        ollama --version 2>/dev/null || true
    else
        info "Downloading and installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        success "Ollama installed"
    fi

    # Ensure Ollama service is enabled
    if systemctl is-active --quiet ollama 2>/dev/null; then
        info "Ollama service is running"
    else
        info "Starting Ollama service..."
        sudo systemctl enable ollama 2>/dev/null || true
        sudo systemctl start ollama 2>/dev/null || true

        # Wait for service to start
        sleep 5
    fi

    # Verify Ollama is responding
    local max_attempts=10
    local attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            success "Ollama is running and responding"
            return 0
        fi
        info "Waiting for Ollama to start (attempt $attempt/$max_attempts)..."
        sleep 2
        ((attempt++))
    done

    warn "Ollama may not be responding. Try running 'ollama serve' manually."
}

# Pull benchmark models
pull_models() {
    info "Pulling benchmark models..."

    source "$VENV_DIR/bin/activate"

    # Pull a small YOLO model to verify setup
    info "Downloading YOLOv8n for verification..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || true

    # Pull LLM models for Ollama
    info "Pulling LLM models (this may take a while)..."

    # 7B models
    local models=(
        "llama2:7b"
        "mistral:7b"
    )

    for model in "${models[@]}"; do
        info "Pulling $model..."
        ollama pull "$model" 2>/dev/null || warn "Failed to pull $model"
    done

    success "Model download complete"
}

# Verify installation
verify_installation() {
    info "Verifying installation..."

    source "$VENV_DIR/bin/activate"

    local errors=0

    # Check Python
    info "Checking Python..."
    python3 --version || ((errors++))

    # Check PyTorch
    info "Checking PyTorch..."
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || ((errors++))

    # Check Ultralytics
    info "Checking Ultralytics..."
    python3 -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')" || ((errors++))

    # Check benchmark module
    info "Checking benchmark module..."
    python3 -c "import benchmark; print('Benchmark module: OK')" || ((errors++))

    # Check Ollama
    info "Checking Ollama..."
    if command -v ollama &> /dev/null; then
        ollama --version 2>/dev/null || true
    else
        warn "Ollama command not found"
        ((errors++))
    fi

    # Summary
    if [[ $errors -eq 0 ]]; then
        success "All components verified successfully!"
    else
        warn "Some components may have issues. Check the log for details."
    fi

    return $errors
}

# Print usage instructions
print_usage_instructions() {
    echo ""
    echo "=========================================="
    echo "  Edge AI Benchmark Suite Setup Complete"
    echo "=========================================="
    echo ""
    echo "To activate the virtual environment:"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
    echo "To run benchmarks:"
    echo "  # Default profile (minimal)"
    echo "  python -m benchmark run all --profile default"
    echo ""
    echo "  # Full profile (comprehensive)"
    echo "  python -m benchmark run all --profile full"
    echo ""
    echo "  # YOLO only"
    echo "  python -m benchmark run yolo --profile default"
    echo ""
    echo "  # LLM only"
    echo "  python -m benchmark run llm --profile default"
    echo ""
    echo "To show system information:"
    echo "  python -m benchmark info"
    echo ""
    echo "To generate reports:"
    echo "  python -m benchmark report --input results --output results/report"
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""
}

# Main execution
main() {
    info "Starting Jetson Nano setup for Edge AI Benchmark Suite"
    info "Log file: $LOG_FILE"

    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"

    detect_platform
    check_privileges
    install_system_deps
    setup_jetson_performance
    setup_venv
    install_python_deps
    install_ollama

    if [[ "$PULL_MODELS" == true ]]; then
        pull_models
    fi

    verify_installation
    print_usage_instructions

    success "Setup complete!"
}

# Run main function
main "$@"
