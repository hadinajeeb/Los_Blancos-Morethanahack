#!/bin/bash
# SkillCorner opendata, 2024/25 A-League, 20 matches (github.com/SkillCorner/opendata, MIT, credit SkillCorner).
# The tracking files are git LFS objects; fetch them over HTTPS and check them against their LFS pointers.
set -u
OUT=${MTAH_ROOT:-.}/external/skillcorner/raw
mkdir -p "$OUT/lfs"; cd "$OUT"
[ -d opendata ] || GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/SkillCorner/opendata.git
(cd opendata && git checkout -q 4340d27)
for d in opendata/data/matches/*/; do
  id=$(basename "$d")
  f="lfs/${id}_tracking_extrapolated.jsonl"
  want=$(grep -o 'sha256:[0-9a-f]*' "$d/${id}_tracking_extrapolated.jsonl" | cut -d: -f2)
  if [ ! -f "$f" ] || [ "$(sha256sum "$f" | cut -d' ' -f1)" != "$want" ]; then
    curl -sL --retry 5 -o "$f" "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/${id}/${id}_tracking_extrapolated.jsonl"
  fi
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want" ] && echo "OK $id" || echo "SHA_MISMATCH $id"
done
echo SKILLCORNER_DOWNLOAD_DONE
