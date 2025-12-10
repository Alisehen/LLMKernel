#!/bin/bash
# Fix NVIDIA GPU Performance Counter permissions for NCU

echo "=========================================="
echo "Fix NCU Profiling Permissions"
echo "=========================================="

echo ""
echo "Creating NVIDIA driver configuration..."

# Create modprobe.d config file
sudo bash -c 'cat > /etc/modprobe.d/nvidia-profiling.conf << EOF
# Allow all users to access NVIDIA GPU Performance Counters
options nvidia "NVreg_RestrictProfilingToAdminUsers=0"
EOF'

echo "✓ Configuration file created: /etc/modprobe.d/nvidia-profiling.conf"
echo ""
echo "Content:"
cat /etc/modprobe.d/nvidia-profiling.conf

echo ""
echo "=========================================="
echo "IMPORTANT: You need to reboot for changes to take effect!"
echo "=========================================="
echo ""
echo "After reboot, verify with:"
echo "  cat /proc/driver/nvidia/params | grep RestrictProfiling"
echo ""
echo "Expected output:"
echo "  RestrictProfilingToAdminUsers: 0"
echo ""
