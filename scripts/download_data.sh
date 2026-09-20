#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
IMAGES_DIR="${DATA_DIR}/images"
ANNOTATIONS_DIR="${DATA_DIR}/annotations"

mkdir -p "${IMAGES_DIR}" "${ANNOTATIONS_DIR}"

IMAGES_BASE_URL="https://vizwiz.cs.colorado.edu/VizWiz_final/images"
ANNOTATIONS_BASE_URL="https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality"
EVAL_CSV_URL="https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VizWiz_quality_issues_train_val_test.csv"

echo "Downloading image splits into ${IMAGES_DIR} ..."
for split in train val test; do
    wget -c "${IMAGES_BASE_URL}/${split}.zip" -O "${IMAGES_DIR}/${split}.zip"
done

echo "Downloading annotations into ${ANNOTATIONS_DIR} ..."
wget -c "${ANNOTATIONS_BASE_URL}/annotations.zip" -O "${ANNOTATIONS_DIR}/annotations.zip"
wget -c "${EVAL_CSV_URL}" -O "${ANNOTATIONS_DIR}/VizWiz_quality_issues_train_val_test.csv"

echo "Unzipping image splits ..."
for split in train val test; do
    unzip -o -q "${IMAGES_DIR}/${split}.zip" -d "${IMAGES_DIR}"
done

echo "Unzipping annotations ..."
unzip -o -q "${ANNOTATIONS_DIR}/annotations.zip" -d "${ANNOTATIONS_DIR}"

echo ""
echo "=== .jpg counts per split (expected: train 23431 / val 7750 / test 8000) ==="
declare -A expected=( [train]=23431 [val]=7750 [test]=8000 )
for split in train val test; do
    count=$(find "${IMAGES_DIR}" -iname "*${split}*.jpg" -type f | wc -l)
    printf "%-6s actual=%-8s expected=%s\n" "${split}" "${count}" "${expected[${split}]}"
done
