#!/bin/bash
# Edge AI Benchmark Suite - Raspberry Pi AI HAT+ Setup Script
# Platform: Raspberry Pi 4/5 with AI HAT+ (Hailo-8L NPU)
#
# This script sets up the complete benchmark environment on a Raspberry Pi
# with the AI HAT+ accelerator (featuring the Hailo-8L NPU).
# It is idempotent and can be run multiple times safely.
#
# Requirements:
#   - Raspberry Pi 4 or 5
#   - AI HAT+ properly connected
#   - Raspberry Pi OS (64-bit) Bookworm or later
#   - Internet connection
#   - Sufficient storage (at least 20GB free recommended)
#
# Usage: sudo ./setup_rpi_ai_hat_plus.sh [--pull-models]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/venv"
LOG_FILE="${PROJECT_ROOT}/setup_rpi_ai_hat_plus.log"
PULL_MODELS=false

# Hailo configuration
HAILO_DEVICE="hailo8l"  # AI HAT+ uses Hailo-8L

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

    # Check for Raspberry Pi
    if [[ ! -f /proc/device-tree/model ]]; then
        error "Could not detect device model. Is this a Raspberry Pi?"
        exit 1
    fi

    local model
    model=$(cat /proc/device-tree/model | tr -d '\0')
    info "Device model: $model"

    # Verify it's a supported Pi
    if [[ ! "$model" =~ "Raspberry Pi" ]]; then
        error "This script is intended for Raspberry Pi devices."
        exit 1
    fi

    # Check architecture
    local arch
    arch=$(uname -m)
    info "Architecture: $arch"

    if [[ "$arch" != "aarch64" ]]; then
        warn "64-bit OS recommended for best performance"
    fi

    # Check for Hailo device
    info "Checking for Hailo accelerator..."
    if lspci 2>/dev/null | grep -qi "hailo"; then
        success "Hailo device detected via PCIe"
    elif ls /dev/hailo* 2>/dev/null; then
        success "Hailo device node found"
    else
        warn "Hailo device not detected. Ensure AI HAT+ is properly connected."
        warn "The HAT may need to be enabled in config.txt"
    fi

    success "Platform detected: Raspberry Pi with AI HAT+"
}

# Check if running as root
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (sudo)"
        exit 1
    fi
}

# Configure Raspberry Pi settings
configure_raspberry_pi() {
    info "Configuring Raspberry Pi settings..."

    # Check and update config.txt for PCIe if needed
    local config_file="/boot/firmware/config.txt"
    if [[ ! -f "$config_file" ]]; then
        config_file="/boot/config.txt"
    fi

    if [[ -f "$config_file" ]]; then
        # Enable PCIe external connector for Pi 5
        if grep -q "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
            if ! grep -q "dtparam=pciex1" "$config_file"; then
                info "Enabling PCIe for Raspberry Pi 5..."
                echo "" >> "$config_file"
                echo "# Enable PCIe for AI HAT+" >> "$config_file"
                echo "dtparam=pciex1" >> "$config_file"
                echo "dtparam=pciex1_gen=3" >> "$config_file"
                warn "PCIe enabled. A reboot may be required."
            fi
        fi
    fi

    success "Raspberry Pi configuration complete"
}

# Install system dependencies
install_system_deps() {
    info "Installing system dependencies..."

    # Update package lists
    apt-get update

    # Install required packages
    apt-get install -y \
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
        libatlas-base-dev \
        libjpeg-dev \
        libpng-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libfreetype6-dev \
        libopenblas-dev \
        pciutils \
        udev

    success "System dependencies installed"
}

# Install Hailo runtime and drivers
install_hailo_runtime() {
    info "Installing Hailo runtime..."

    # Check if Hailo repository is already added
    if [[ ! -f /etc/apt/sources.list.d/hailo.list ]]; then
        info "Adding Hailo repository..."

        # Import Hailo GPG key
        curl -fsSL https://hailo.ai/deb/hailo.gpg.key | gpg --dearmor -o /usr/share/keyrings/hailo-archive-keyring.gpg

        # Add repository
        echo "deb [signed-by=/usr/share/keyrings/hailo-archive-keyring.gpg] https://hailo.ai/deb stable main" > /etc/apt/sources.list.d/hailo.list

        apt-get update
    fi

    # Install Hailo packages
    info "Installing Hailo packages..."
    apt-get install -y \
        hailort \
        hailo-firmware \
        hailo-pcie-driver \
        hailort-pcie \
        || warn "Some Hailo packages may not be available"

    # Alternative: Install from Raspberry Pi repository if available
    if ! command -v hailortcli &> /dev/null; then
        info "Trying Raspberry Pi Hailo packages..."
        apt-get install -y \
            hailo-all \
            || warn "hailo-all package not found"
    fi

    # Load Hailo driver
    if [[ -f /lib/modules/$(uname -r)/extra/hailo_pci.ko ]]; then
        info "Loading Hailo PCIe driver..."
        modprobe hailo_pci 2>/dev/null || true
    fi

    # Verify Hailo installation
    if command -v hailortcli &> /dev/null; then
        info "Hailo CLI version:"
        hailortcli --version 2>/dev/null || true

        info "Hailo device info:"
        hailortcli scan 2>/dev/null || warn "Could not scan for Hailo devices"
    else
        warn "Hailo CLI not found. Manual installation may be required."
        warn "Visit: https://hailo.ai/developer-zone/documentation/"
    fi

    success "Hailo runtime installation complete"
}

# Create Python virtual environment
setup_venv() {
    info "Setting up Python virtual environment..."

    # Get the actual user (not root)
    local actual_user
    actual_user=${SUDO_USER:-$USER}

    if [[ -d "$VENV_DIR" ]]; then
        info "Virtual environment already exists at $VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
        success "Virtual environment created at $VENV_DIR"
    fi

    # Activate and upgrade pip
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip wheel setuptools

    # Fix ownership
    chown -R "$actual_user:$actual_user" "$VENV_DIR"

    success "Python virtual environment ready"
}

# Install Python dependencies
install_python_deps() {
    info "Installing Python dependencies..."

    source "$VENV_DIR/bin/activate"

    # Install numpy first
    pip install numpy

    # Install PyTorch for ARM64
    info "Installing PyTorch for ARM64..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

    # Install Ultralytics YOLO
    info "Installing Ultralytics..."
    pip install ultralytics

    # Install Hailo Python bindings if available
    info "Installing Hailo Python bindings..."
    pip install hailo-platform 2>/dev/null || warn "Hailo Python package not found in PyPI"

    # Try installing from system packages
    if [[ -d /usr/lib/python3/dist-packages/hailo_platform ]]; then
        info "Linking system Hailo packages to venv..."
        ln -sf /usr/lib/python3/dist-packages/hailo_platform "$VENV_DIR/lib/python3.*/site-packages/" 2>/dev/null || true
    fi

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

    # Fix ownership
    local actual_user
    actual_user=${SUDO_USER:-$USER}
    chown -R "$actual_user:$actual_user" "$VENV_DIR"

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
        systemctl enable ollama 2>/dev/null || true
        systemctl start ollama 2>/dev/null || true

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

# Set up udev rules for Hailo
setup_udev_rules() {
    info "Setting up udev rules for Hailo..."

    local rules_file="/etc/udev/rules.d/99-hailo.rules"

    if [[ ! -f "$rules_file" ]]; then
        cat > "$rules_file" << 'EOF'
# Hailo PCIe device permissions
SUBSYSTEM=="hailo", MODE="0666"
KERNEL=="hailo*", MODE="0666"
EOF
        udevadm control --reload-rules
        udevadm trigger
        success "Hailo udev rules installed"
    else
        info "Hailo udev rules already exist"
    fi
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

    # Start with smaller models suitable for Pi
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
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || ((errors++))

    # Check Ultralytics
    info "Checking Ultralytics..."
    python3 -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')" || ((errors++))

    # Check benchmark module
    info "Checking benchmark module..."
    python3 -c "import benchmark; print('Benchmark module: OK')" || ((errors++))

    # Check Hailo
    info "Checking Hailo..."
    if command -v hailortcli &> /dev/null; then
        hailortcli scan 2>/dev/null || warn "Could not scan for Hailo devices"
    else
        warn "Hailo CLI not found"
    fi

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
    local actual_user
    actual_user=${SUDO_USER:-$USER}

    echo ""
    echo "=========================================="
    echo "  Edge AI Benchmark Suite Setup Complete"
    echo "  Platform: Raspberry Pi AI HAT+"
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
    echo "Note: For best Hailo NPU performance, consider exporting YOLO models"
    echo "to Hailo format using the Ultralytics export function."
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""

    # Check if reboot needed
    if [[ -f /var/run/reboot-required ]]; then
        warn "A system reboot is required to complete setup."
        warn "Please reboot with: sudo reboot"
    fi
}

# Main execution
main() {
    info "Starting Raspberry Pi AI HAT+ setup for Edge AI Benchmark Suite"
    info "Log file: $LOG_FILE"

    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"

    check_privileges
    detect_platform
    configure_raspberry_pi
    install_system_deps
    install_hailo_runtime
    setup_udev_rules
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
