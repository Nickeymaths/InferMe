#!/usr/bin/env bash
# One-time host setup: Docker + NVIDIA Container Toolkit. Run: sudo ./setup-host.sh
set -euo pipefail
[ "$EUID" -eq 0 ] || { echo "Run with sudo"; exit 1; }
USER_NAME="${SUDO_USER:-$USER}"

apt-get update
apt-get install -y docker.io docker-compose-v2 curl gnupg ca-certificates

# NVIDIA Container Toolkit (generic deb repo)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker
usermod -aG docker "$USER_NAME"

echo "Done. Log out/in (or run 'newgrp docker'), then: docker compose run --rm cuda ./run.sh"
