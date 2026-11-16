#!/bin/bash
P=${MTAH_ROOT:-.}
cd "$(dirname "$0")"
for m in J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9; do
  uv run -q --with numpy python convert_dfl.py $P/external/dfl/raw DFL-MAT-$m $P/external/dfl/common/DFL-MAT-$m.npz > convert_$m.log 2>&1 &
done
wait
cat convert_J*.log
