#!/bin/bash

# Define variables
URL="https://zenodo.org/records/15032263/files/test_structures.tar.gz?download=1"
FILE_NAME="test_structures.tar.gz"

download_dir="../data"
mkdir -p "$download_dir"
cd $download_dir

# Download the file using wget
# wget -O "$FILE_NAME" "$URL"
curl -L -o "$FILE_NAME" "$URL"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Download failed!"
    exit 1
fi

# Extract the tar.gz file
tar -xzvf "$FILE_NAME"

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Extraction failed!"
    exit 1
fi

rm -rf "$FILE_NAME"

echo "Download and extraction completed successfully."
