#!/bin/bash
# Edge AI Benchmark Suite - Raspberry Pi AI HAT+ 2 Setup Script
# Platform: Raspberry Pi 5 with AI HAT+ 2 (Hailo-10H NPU)
#
# This script sets up the complete benchmark environment on a Raspberry Pi 5
# with the AI HAT+ 2 accelerator (featuring the Hailo-10H NPU).
# It is idempotent and can be run multiple times safely.
#
# Requirements:
#   - Raspberry Pi 5 (required for AI HAT+ 2)
#   - AI HAT+ 2 properly connected
#   - Raspberry Pi OS (64-bit) Bookworm or later
#   - Internet connection
#   - Sufficient storage (at least 20GB free recommended)
#
# Usage: sudo ./setup_rpi_ai_hat_plus_2.sh [--pull-models]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/venv"
LOG_FILE="${PROJECT_ROOT}/setup_rpi_ai_hat_plus_2.log"
PULL_MODELS=false

# Hailo configuration
HAILO_DEVICE="hailo10h"  # AI HAT+ 2 uses Hailo-10H

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

    # Verify it's a Raspberry Pi 5 (required for AI HAT+ 2)
    if [[ ! "$model" =~ "Raspberry Pi 5" ]]; then
        error "AI HAT+ 2 requires Raspberry Pi 5."
        error "Detected: $model"
        error "If you have AI HAT+ (not HAT+ 2), use setup_rpi_ai_hat_plus.sh instead."
        exit 1
    fi

    # Check architecture
    local arch
    arch=$(uname -m)
    info "Architecture: $arch"

    if [[ "$arch" != "aarch64" ]]; then
        error "64-bit OS is required for AI HAT+ 2"
        exit 1
    fi

    # Check for Hailo-10H device
    info "Checking for Hailo-10H accelerator..."
    local hailo_found=false

    if lspci 2>/dev/null | grep -qi "hailo"; then
        success "Hailo device detected via PCIe"
        lspci | grep -i hailo || true
        hailo_found=true
    fi

    if ls /dev/hailo* 2>/dev/null; then
        success "Hailo device node found"
        hailo_found=true
    fi

    if [[ "$hailo_found" == false ]]; then
        warn "Hailo device not detected. Ensure AI HAT+ 2 is properly connected."
        warn "The HAT may need PCIe to be enabled in config.txt"
    fi

    success "Platform detected: Raspberry Pi 5 with AI HAT+ 2"
}

# Check if running as root
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (sudo)"
        exit 1
    fi
}

# Configure Raspberry Pi 5 settings for AI HAT+ 2
configure_raspberry_pi() {
    info "Configuring Raspberry Pi 5 settings for AI HAT+ 2..."

    local config_file="/boot/firmware/config.txt"
    if [[ ! -f "$config_file" ]]; then
        config_file="/boot/config.txt"
    fi

    if [[ -f "$config_file" ]]; then
        local config_changed=false

        # Enable PCIe external connector (required for AI HAT+ 2)
        if ! grep -q "^dtparam=pciex1" "$config_file"; then
            info "Enabling PCIe external connector..."
            echo "" >> "$config_file"
            echo "# Enable PCIe for AI HAT+ 2" >> "$config_file"
            echo "dtparam=pciex1" >> "$config_file"
            config_changed=true
        fi

        # Set PCIe to Gen 3 for maximum performance
        if ! grep -q "^dtparam=pciex1_gen=3" "$config_file"; then
            info "Setting PCIe to Gen 3 mode..."
            echo "dtparam=pciex1_gen=3" >> "$config_file"
            config_changed=true
        fi

        # Enable Hailo overlay if available
        if [[ -f /boot/firmware/overlays/hailo-8.dtbo ]] || [[ -f /boot/overlays/hailo-8.dtbo ]]; then
            if ! grep -q "^dtoverlay=hailo-8" "$config_file"; then
                info "Enabling Hailo-8 device tree overlay..."
                echo "dtoverlay=hailo-8" >> "$config_file"
                config_changed=true
            fi
        fi

        if [[ "$config_changed" == true ]]; then
            warn "Configuration changed. A reboot will be required."
        fi
    fi

    # Configure GPU memory split for optimal AI performance
    if [[ -f "$config_file" ]]; then
        if ! grep -q "^gpu_mem=" "$config_file"; then
            info "Setting GPU memory allocation..."
            echo "gpu_mem=256" >> "$config_file"
        fi
    fi

    success "Raspberry Pi 5 configuration complete"
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
        udev \
        i2c-tools

    success "System dependencies installed"
}

# Install Hailo-10H runtime and drivers
install_hailo_runtime() {
    info "Installing Hailo-10H runtime for AI HAT+ 2..."

    # For Raspberry Pi AI HAT+ 2, Hailo-10H packages are available from the official
    # Raspberry Pi repository (archive.raspberrypi.com), not from hailo.ai
    # The hailo.ai repository is deprecated/unavailable for RPi users.
    # AI HAT+ 2 uses Hailo-10H which requires hailo-h10-all package.

    # Remove any stale Hailo repository that may cause apt errors
    if [[ -f /etc/apt/sources.list.d/hailo.list ]]; then
        info "Removing outdated Hailo repository..."
        rm -f /etc/apt/sources.list.d/hailo.list
        rm -f /usr/share/keyrings/hailo-archive-keyring.gpg
        apt-get update
    fi

    # Install Hailo-10H packages from Raspberry Pi repository
    # Note: hailo-h10-all is specifically for Hailo-10H (AI HAT+ 2)
    # whereas hailo-all is for Hailo-8/8L (AI HAT+)
    info "Installing Hailo-10H packages from Raspberry Pi repository..."
    apt-get install -y hailo-h10-all || {
        warn "hailo-h10-all package not found, trying individual packages..."

        apt-get install -y \
            hailort \
            hailo-h10-firmware \
            2>/dev/null || warn "Some Hailo packages may not be available. Ensure your OS is up to date."
    }

    # Install Hailo TAPPAS for optimized pipelines (optional)
    apt-get install -y hailo-tappas-core 2>/dev/null || info "TAPPAS not available (optional)"

    # Reload Hailo driver to pick up new firmware
    info "Loading Hailo PCIe driver..."
    rmmod hailo_pci 2>/dev/null || true
    if modprobe hailo_pci 2>/dev/null; then
        success "Hailo PCIe driver loaded"
    else
        warn "Could not load Hailo driver. May need reboot."
    fi

    # Verify Hailo installation
    if command -v hailortcli &> /dev/null; then
        info "Hailo CLI version:"
        hailortcli --version 2>/dev/null || true

        info "Scanning for Hailo devices..."
        hailortcli scan 2>/dev/null || warn "Could not scan for Hailo devices"

        # Show device info if available
        if hailortcli scan 2>/dev/null | grep -q "hailo"; then
            info "Hailo device information:"
            hailortcli fw-control identify 2>/dev/null || true
        fi
    else
        warn "Hailo CLI not found. Manual installation may be required."
        warn "Ensure your Raspberry Pi OS is up to date: sudo apt update && sudo apt full-upgrade"
    fi

    success "Hailo-10H runtime installation complete"
}

# Set up udev rules for Hailo
setup_udev_rules() {
    info "Setting up udev rules for Hailo-10H..."

    local rules_file="/etc/udev/rules.d/99-hailo.rules"

    cat > "$rules_file" << 'EOF'
# Hailo-10H PCIe device permissions for AI HAT+ 2
SUBSYSTEM=="hailo", MODE="0666"
KERNEL=="hailo*", MODE="0666"

# Grant access to video group
SUBSYSTEM=="hailo", GROUP="video", MODE="0660"
EOF

    udevadm control --reload-rules
    udevadm trigger

    # Add user to video group
    local actual_user
    actual_user=${SUDO_USER:-$USER}
    if [[ "$actual_user" != "root" ]]; then
        usermod -aG video "$actual_user" 2>/dev/null || true
    fi

    success "Hailo udev rules installed"
}

# Create Python virtual environment
setup_venv() {
    info "Setting up Python virtual environment..."

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

    # Install Hailo Python bindings
    info "Installing Hailo Python bindings..."

    # Try pip install first
    pip install hailo-platform 2>/dev/null || {
        warn "Hailo Python package not found in PyPI"

        # Try to link system packages
        if [[ -d /usr/lib/python3/dist-packages/hailo_platform ]]; then
            info "Linking system Hailo packages to venv..."
            local site_packages
            site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])")
            ln -sf /usr/lib/python3/dist-packages/hailo_platform "$site_packages/" 2>/dev/null || true
            ln -sf /usr/lib/python3/dist-packages/hailort "$site_packages/" 2>/dev/null || true
        fi
    }

    # Install Hailo Model Zoo utilities if available
    pip install hailo-model-zoo 2>/dev/null || warn "Hailo Model Zoo not available"

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

# Configure performance settings
configure_performance() {
    info "Configuring system for optimal performance..."

    # Disable CPU frequency scaling (use performance governor)
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -f "$cpu" ]]; then
            echo "performance" > "$cpu" 2>/dev/null || true
        fi
    done

    # Increase kernel parameters for better performance
    cat > /etc/sysctl.d/99-ai-benchmark.conf << 'EOF'
# Optimize for AI benchmark workloads
vm.swappiness=10
vm.dirty_ratio=60
vm.dirty_background_ratio=2
net.core.rmem_max=16777216
net.core.wmem_max=16777216
EOF

    sysctl -p /etc/sysctl.d/99-ai-benchmark.conf 2>/dev/null || true

    success "Performance settings configured"
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

    # Models suitable for Pi 5 with AI HAT+ 2
    local models=(
        "llama2:7b"
        "mistral:7b"
        "gemma2:9b"
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
    info "Checking Hailo-8..."
    if command -v hailortcli &> /dev/null; then
        hailortcli --version 2>/dev/null || true

        if hailortcli scan 2>/dev/null | grep -qi "hailo"; then
            success "Hailo-8 device detected and responsive"
        else
            warn "Hailo-8 device not responding"
        fi
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
    echo "  Platform: Raspberry Pi 5 AI HAT+ 2"
    echo "  Accelerator: Hailo-10H"
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
    echo "Hailo-10H NPU Notes:"
    echo "  - The Hailo-10H is optimized for edge AI workloads"
    echo "  - For optimal YOLO performance, export models to Hailo format"
    echo "  - Example: yolo export model=yolov8n.pt format=hailo"
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""

    # Check if reboot needed
    if [[ -f /var/run/reboot-required ]] || grep -q "dtparam=pciex1" "$LOG_FILE" 2>/dev/null; then
        warn "A system reboot is recommended to complete setup."
        warn "Please reboot with: sudo reboot"
    fi
}

# Main execution
main() {
    info "Starting Raspberry Pi 5 AI HAT+ 2 setup for Edge AI Benchmark Suite"
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
    configure_performance
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
