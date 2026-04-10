#!/usr/bin/env bash
# Download and extract FreiHAND and/or HO3D datasets.
#
# FreiHAND requires registration:
#   https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
#
# HO3D requires registration:
#   https://www.tugraz.at/index.php?id=40231
#
# Usage:
#   bash scripts/download_data.sh --freihand /data/freihand
#   bash scripts/download_data.sh --ho3d /data/ho3d
#   bash scripts/download_data.sh --freihand /data/freihand --ho3d /data/ho3d

set -euo pipefail

FREIHAND_DIR=""
HO3D_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --freihand) FREIHAND_DIR="$2"; shift 2 ;;
        --ho3d)     HO3D_DIR="$2";     shift 2 ;;
        *) echo "Unknown arg: $1. Usage: $0 [--freihand DIR] [--ho3d DIR]"; exit 1 ;;
    esac
done

if [[ -z "$FREIHAND_DIR" && -z "$HO3D_DIR" ]]; then
    echo "Specify at least one of --freihand or --ho3d."
    exit 1
fi

# -------------------------------------------------------------------------
# FreiHAND
# -------------------------------------------------------------------------
if [[ -n "$FREIHAND_DIR" ]]; then
    echo "=== FreiHAND Setup ==="
    mkdir -p "$FREIHAND_DIR"

    echo ""
    echo "FreiHAND requires manual download (registration required)."
    echo "Download the following files from:"
    echo "  https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html"
    echo ""
    echo "Expected files to place in $FREIHAND_DIR:"
    echo "  FreiHAND_pub_v2.zip          (training + evaluation RGB)"
    echo "  FreiHAND_pub_v2_eval.zip     (evaluation annotations)"
    echo ""

    FREIHAND_ZIP="$FREIHAND_DIR/FreiHAND_pub_v2.zip"
    if [[ -f "$FREIHAND_ZIP" ]]; then
        echo "Found $FREIHAND_ZIP — extracting..."
        cd "$FREIHAND_DIR"
        unzip -q FreiHAND_pub_v2.zip
        # Rename extracted dirs to expected structure if needed
        if [[ -d "FreiHAND_pub_v2" ]]; then
            mv FreiHAND_pub_v2/* . 2>/dev/null || true
            rmdir FreiHAND_pub_v2 2>/dev/null || true
        fi
        echo "FreiHAND extracted to $FREIHAND_DIR"
        echo ""
        echo "Expected structure:"
        echo "  $FREIHAND_DIR/"
        echo "    training/{rgb,mask}/"
        echo "    evaluation/rgb/"
        echo "    training_K.json"
        echo "    training_xyz.json"
        echo "    evaluation_K.json"
        echo "    evaluation_xyz.json"
    else
        echo "File not found: $FREIHAND_ZIP"
        echo "Please download manually and re-run."
    fi
fi

# -------------------------------------------------------------------------
# HO3D
# -------------------------------------------------------------------------
if [[ -n "$HO3D_DIR" ]]; then
    echo "=== HO3D Setup ==="
    mkdir -p "$HO3D_DIR"

    echo ""
    echo "HO3D requires manual download (registration required)."
    echo "Download from: https://www.tugraz.at/index.php?id=40231"
    echo ""
    echo "Expected files to place in $HO3D_DIR:"
    echo "  HO3D_v3.zip"
    echo ""

    HO3D_ZIP="$HO3D_DIR/HO3D_v3.zip"
    if [[ -f "$HO3D_ZIP" ]]; then
        echo "Found $HO3D_ZIP — extracting..."
        cd "$HO3D_DIR"
        unzip -q HO3D_v3.zip
        echo "HO3D extracted to $HO3D_DIR"
        echo ""
        echo "Expected structure:"
        echo "  $HO3D_DIR/"
        echo "    train/{ABF10,...}/{rgb,meta}/"
        echo "    evaluation/{...}/"
        echo "    train.txt"
        echo "    evaluation.txt"
    else
        echo "File not found: $HO3D_ZIP"
        echo "Please download manually and re-run."
    fi
fi

echo ""
echo "Done. Verify dataset structure before running training."
