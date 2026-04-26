#!/bin/bash

# Config
OUTBA_DIR="outba"
RES_FRAMES_DIR="res/frames"
AUDIO="res/audio.mp3"
OUTPUT="output.mp4"
FRAMERATE=30
TMPDIR="tmp_frames"
BLANK_COLOR="black"
PARALLEL_JOBS=8

mkdir -p "$TMPDIR"

# Get dimensions from first outba frame
FIRST_OUTBA=$(ls "$OUTBA_DIR"/*.png | head -1)
WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$FIRST_OUTBA")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$FIRST_OUTBA")

echo "Frame size: ${WIDTH}x${HEIGHT}"

# Generate blank frame once
BLANK="$TMPDIR/blank.png"
ffmpeg -y -loglevel error \
    -f lavfi -i "color=${BLANK_COLOR}:size=${WIDTH}x${HEIGHT}:rate=1" \
    -frames:v 1 "$BLANK"

compose_frame() {
    RES_FRAME="$1"
    OUTBA_DIR="$2"
    TMPDIR="$3"
    WIDTH="$4"
    HEIGHT="$5"
    BLANK="$6"

    BASENAME=$(basename "$RES_FRAME")
    NUMBER=$(echo "$BASENAME" | grep -oE '[0-9]+' | tail -1)

    OUTBA_FRAME=$(ls "$OUTBA_DIR"/*_${NUMBER}.png 2>/dev/null | head -1)
    COMPOSED="$TMPDIR/composed_${NUMBER}.png"

    if [ -n "$OUTBA_FRAME" ]; then
        ffmpeg -y -loglevel error \
            -i "$RES_FRAME" -i "$OUTBA_FRAME" \
            -filter_complex "[0:v]scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black[left];[left][1:v]hstack=inputs=2" \
            "$COMPOSED"
    else
        ffmpeg -y -loglevel error \
            -i "$RES_FRAME" -i "$BLANK" \
            -filter_complex "[0:v]scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black[left];[left][1:v]hstack=inputs=2" \
            "$COMPOSED"
    fi

    echo "Composed frame $NUMBER"
}

export -f compose_frame

# Process frames in parallel
for RES_FRAME in "$RES_FRAMES_DIR"/output_*.jpg; do
    compose_frame "$RES_FRAME" "$OUTBA_DIR" "$TMPDIR" "$WIDTH" "$HEIGHT" "$BLANK" &
    while [ $(jobs -r | wc -l) -ge $PARALLEL_JOBS ]; do sleep 0.05; done
done
wait

echo "All frames composed, encoding video..."

# Assemble video from composed frames + audio
ffmpeg -y \
    -framerate $FRAMERATE \
    -pattern_type glob -i "$TMPDIR/composed_*.png" \
    -i "$AUDIO" \
    -c:v libx264 -pix_fmt yuv420p \
    -c:a aac \
    -shortest \
    "$OUTPUT"

echo "Done: $OUTPUT"
