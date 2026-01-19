#!/bin/bash
# Edge AI Benchmark Suite - Common Setup Functions
# This file contains shared functions used by all platform-specific setup scripts.
#
# Usage: source this file from platform-specific scripts
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Common configuration
BENCHMARK_PYTHON_VERSION="3.10"
BENCHMARK_MIN_STORAGE_GB=10

# Color codes for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export NC='\033[0m' # No Color

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}"
}

info() { log "INFO" "${BLUE}$*${NC}"; }
success() { log "SUCCESS" "${GREEN}$*${NC}"; }
warn() { log "WARN" "${YELLOW}$*${NC}"; }
error() { log "ERROR" "${RED}$*${NC}"; }

# Check available disk space
check_disk_space() {
    local min_gb="${1:-$BENCHMARK_MIN_STORAGE_GB}"
    local available_gb
    available_gb=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')

    if [[ "$available_gb" -lt "$min_gb" ]]; then
        warn "Low disk space: ${available_gb}GB available, ${min_gb}GB recommended"
        return 1
    fi

    info "Disk space check passed: ${available_gb}GB available"
    return 0
}

# Check Python version
check_python_version() {
    local min_version="${1:-3.9}"

    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found"
        return 1
    fi

    local current_version
    current_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if [[ "$(printf '%s\n' "$min_version" "$current_version" | sort -V | head -n1)" != "$min_version" ]]; then
        error "Python $min_version or higher required, found $current_version"
        return 1
    fi

    info "Python version check passed: $current_version"
    return 0
}

# Check internet connectivity
check_internet() {
    local test_url="${1:-https://pypi.org}"

    if curl -s --head --max-time 5 "$test_url" > /dev/null 2>&1; then
        info "Internet connectivity: OK"
        return 0
    else
        warn "No internet connectivity detected"
        return 1
    fi
}

# Install common Python dependencies
install_common_python_deps() {
    local venv_dir="$1"

    if [[ -z "$venv_dir" ]] || [[ ! -d "$venv_dir" ]]; then
        error "Virtual environment directory not found: $venv_dir"
        return 1
    fi

    source "$venv_dir/bin/activate"

    info "Installing common Python dependencies..."

    pip install --upgrade pip wheel setuptools

    # Core dependencies
    pip install \
        numpy \
        psutil \
        requests \
        pyyaml \
        opencv-python-headless

    success "Common Python dependencies installed"
}

# Verify Ollama installation
verify_ollama() {
    local max_wait="${1:-30}"
    local attempt=1

    if ! command -v ollama &> /dev/null; then
        warn "Ollama not installed"
        return 1
    fi

    # Check if Ollama server is responding
    while [[ $attempt -le $max_wait ]]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            success "Ollama server is responding"
            return 0
        fi
        sleep 1
        ((attempt++))
    done

    warn "Ollama server not responding after ${max_wait}s"
    return 1
}

# Get OS information
get_os_info() {
    local os_name=""
    local os_version=""

    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        os_name="$NAME"
        os_version="$VERSION_ID"
    elif [[ -f /etc/lsb-release ]]; then
        source /etc/lsb-release
        os_name="$DISTRIB_ID"
        os_version="$DISTRIB_RELEASE"
    fi

    echo "$os_name $os_version"
}

# Get kernel version
get_kernel_version() {
    uname -r
}

# Get CPU model
get_cpu_model() {
    if [[ -f /proc/cpuinfo ]]; then
        grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs
    else
        uname -p
    fi
}

# Get total RAM in GB
get_ram_gb() {
    local ram_kb
    ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    echo $((ram_kb / 1024 / 1024))
}

# Get storage type (SSD/HDD/SD)
get_storage_type() {
    local root_device
    root_device=$(df / | tail -1 | awk '{print $1}')

    # Extract base device
    local base_device
    base_device=$(echo "$root_device" | sed 's/[0-9]*$//' | sed 's/p$//')

    # Check if it's an SD card
    if [[ "$base_device" =~ "mmcblk" ]]; then
        echo "SD Card"
        return
    fi

    # Check rotational flag for SSD/HDD
    local rotational_file="/sys/block/$(basename "$base_device")/queue/rotational"
    if [[ -f "$rotational_file" ]]; then
        if [[ $(cat "$rotational_file") -eq 0 ]]; then
            echo "SSD"
        else
            echo "HDD"
        fi
    else
        echo "Unknown"
    fi
}

# Print system summary
print_system_summary() {
    echo ""
    echo "System Summary"
    echo "=============="
    echo "OS: $(get_os_info)"
    echo "Kernel: $(get_kernel_version)"
    echo "CPU: $(get_cpu_model)"
    echo "RAM: $(get_ram_gb) GB"
    echo "Storage: $(get_storage_type)"
    echo ""
}

# Create backup of a file
backup_file() {
    local file="$1"
    local backup_dir="${2:-/tmp/benchmark_backups}"

    if [[ -f "$file" ]]; then
        mkdir -p "$backup_dir"
        local backup_name
        backup_name="$(basename "$file").$(date +%Y%m%d_%H%M%S).bak"
        cp "$file" "$backup_dir/$backup_name"
        info "Backed up $file to $backup_dir/$backup_name"
    fi
}

# Check if running on ARM64
is_arm64() {
    [[ "$(uname -m)" == "aarch64" ]]
}

# Check if running on x86_64
is_x86_64() {
    [[ "$(uname -m)" == "x86_64" ]]
}

# Cleanup function to call on exit
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        warn "Script exited with code $exit_code"
    fi
}

# Set trap for cleanup
trap cleanup_on_exit EXIT
