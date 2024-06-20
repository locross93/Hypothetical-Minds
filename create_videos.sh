#!/bin/bash

# Check if target folder is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_folder>"
    exit 1
fi

TARGET_FOLDER="$1"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg could not be found. Please install it to continue."
    exit 1
fi

# Get absolute path of the target folder
ABS_TARGET_FOLDER="$(realpath "$TARGET_FOLDER")"

# Loop through each subdirectory and create a video
for dir in "$ABS_TARGET_FOLDER"/*/; do
    # Remove the trailing slash to get the directory name
    dir_name=$(basename "$dir")
    echo "Processing directory: $dir_name"

    # Create a temporary file listing all frames in sorted order
    temp_file=$(mktemp)
    for f in $(ls "$dir"/*_[0-9]*.png | sort -V); do
        echo "file '$f'" >> "$temp_file"
    done

    # Run ffmpeg to create the video at 10 fps using the sorted list of frames
    ffmpeg -r 10 -f concat -safe 0 -i "$temp_file" -c:v libx264 -r 10 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "${ABS_TARGET_FOLDER}/${dir_name}.mp4"

    # Clean up the temporary file
    rm "$temp_file"
done

echo "Video creation complete."
