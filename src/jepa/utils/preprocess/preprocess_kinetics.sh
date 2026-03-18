#!/usr/bin/env bash

SOURCE_DIR="data/kinetics-dataset/k700-2020"
TARGET_DIR="data/kinetics-dataset/k700-2020-processed-v3"
LOG_FILE="output.log"

mkdir -p "$TARGET_DIR"
cp -r "$SOURCE_DIR/annotations" "$TARGET_DIR"

preprocess_split() {
  local split="$1"

  # Recreate directory structure in the target
  find "$SOURCE_DIR/$split" -type d | while read -r d; do
    mkdir -p "$TARGET_DIR/${d#$SOURCE_DIR/}"
  done

  # Count total MP4 files via ripgrep
  files_total=$(rg -g '*.mp4' --files "$SOURCE_DIR/$split" | wc -l)
  echo "Found $files_total total mp4 files in $split"

  needed_files=$(
    rg -g '*.mp4' --files "$SOURCE_DIR/$split" |
      pv -l -s "$files_total"
  )

  echo "Transcoding $(echo "$needed_files" | wc -l) files in $split..."

  echo "$needed_files" | parallel --bar -j 16 "
    out={= s:$SOURCE_DIR:$TARGET_DIR: =};
    ffmpeg -hide_banner -loglevel error \
      -i {} \
      -vf \"scale='if(gt(iw,ih),-2,240)':'if(gt(iw,ih),240,-2)',fps=4\" \
      -c:v libx265 \
      -preset slow \
      -crf 28 \
      -x265-params "keyint=4:min-keyint=4:scenecut=0:no-open-gop=1:log-level=quiet" \
      -threads 4 \
      -an \
      -y \"\$out\"
  "
}

preprocess_split train
preprocess_split val
