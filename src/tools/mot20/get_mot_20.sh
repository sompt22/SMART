#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/mot20"
ARCHIVE_PATH="${DATA_DIR}/MOT20.zip"

mkdir -p "${DATA_DIR}"

if [ ! -f "${ARCHIVE_PATH}" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L "https://motchallenge.net/data/MOT20.zip" -o "${ARCHIVE_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE_PATH}" "https://motchallenge.net/data/MOT20.zip"
  else
    echo "Neither curl nor wget is available. Please download MOT20.zip manually into ${DATA_DIR}."
    exit 1
  fi
fi

unzip -o "${ARCHIVE_PATH}" -d "${DATA_DIR}"
mkdir -p "${DATA_DIR}/images" "${DATA_DIR}/annotations"

if [ -d "${DATA_DIR}/train" ] && [ ! -d "${DATA_DIR}/images/train" ]; then
  mv "${DATA_DIR}/train" "${DATA_DIR}/images/train"
fi
if [ -d "${DATA_DIR}/test" ] && [ ! -d "${DATA_DIR}/images/test" ]; then
  mv "${DATA_DIR}/test" "${DATA_DIR}/images/test"
fi

python3 "${REPO_ROOT}/src/tools/mot20/convert_mot_to_coco.py"
echo "MOT20 dataset prepared under ${DATA_DIR}"
