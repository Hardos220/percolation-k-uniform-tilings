#!/usr/bin/env bash
# Fetch the k-uniform part (k = 1..10) of the tessellation catalog of Čtrnáct and Kopczyński
# at the commit used in the paper into data/tes/ (layout: data/tes/<k>/<orbit structure>/<name>.tes).
# usage: bash tools/fetch_catalog.sh
set -euo pipefail
COMMIT=6772c7c6c1e57fbc1fa07e740b8ae4da7529189b
TMP=$(mktemp -d)
git clone --filter=blob:none --no-checkout https://github.com/zenorogue/tes-catalog.git "$TMP"
git -C "$TMP" sparse-checkout set files/tessellations/k-uniform
git -C "$TMP" checkout -q "$COMMIT"
mkdir -p data/tes
for k in 01 02 03 04 05 06 07 08 09 10; do
  cp -r "$TMP/files/tessellations/k-uniform/$k" data/tes/
done
echo "$COMMIT" > data/tes/CATALOG_COMMIT.txt
rm -rf "$TMP"
echo "catalog k = 1..10 copied to data/tes"
