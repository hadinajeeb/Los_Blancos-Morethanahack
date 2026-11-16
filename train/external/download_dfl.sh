#!/bin/bash
# Download the DFL/Sportec open dataset (figshare 28196177, CC BY 4.0) and verify MD5.
set -u
OUT=${MTAH_ROOT:-.}/external/dfl/raw
mkdir -p "$OUT"; cd "$OUT"
curl -s -m 60 https://api.figshare.com/v2/articles/28196177 > article.json
python3 - <<'PY' > files.tsv
import json
d=json.load(open("article.json"))
for f in d["files"]:
    print(f["name"], f["download_url"], f["supplied_md5"], f["size"], sep="\t")
PY
dl() {
  name=$1; url=$2; md5=$3
  if [ -f "$name" ] && [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ]; then echo "OK(cached) $name"; return; fi
  curl -sL --retry 5 -o "$name" "$url"
  got=$(md5sum "$name" | cut -d' ' -f1)
  if [ "$got" = "$md5" ]; then echo "OK $name"; else echo "MD5_MISMATCH $name $got $md5"; fi
}
export -f dl
cat files.tsv | xargs -P 4 -L 1 bash -c 'dl "$0" "$1" "$2"'
echo DFL_DOWNLOAD_DONE
