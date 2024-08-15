#!/bin/bash

# Function to download and extract files from HuggingFace
download_and_extract() {
    local repo_name=$1
    local filename=$2
    local url="https://huggingface.co/${repo_name}/resolve/main/${filename}?download=true"
    
    echo "Downloading ${filename}..."
    wget -O "${filename}" "${url}"
    
    echo "Extracting ${filename}..."
    unzip "${filename}"
    
    echo "Removing ${filename}..."
    rm "${filename}"
}

# Install requirements
pip install -r requirements.txt
pip install --upgrade transformers
pip install -e .

# HuggingFace repository name
REPO_NAME="canrager/ravel"

# List of files to download
FILES=(
    "data-tinyllama-gemma2.zip"
)

# Download and extract each file
for file in "${FILES[@]}"; do
    download_and_extract "$REPO_NAME" "$file"
done