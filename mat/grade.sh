#!/usr/bin/env bash
# mat/grade.sh EXAM_DIR BASE.json TUNED.json OUT_DIR : paired grading (both sets side by side, same grader) with Claude Code
set -euo pipefail
E=$(realpath "$1"); A=$(realpath "$2"); B=$(realpath "$3"); mkdir -p "$4"; O=$(realpath "$4"); x=$(basename "$E")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
claude -p --model opus --allowedTools Read Write --add-dir "$E" "$(dirname "$A")" "$(dirname "$B")" "$O" -- "Follow the instructions in $ROOT/mat/GRADER.md with these inputs:
- exam: $E/exam.json (image paths are relative to $E; look at the images when an item depends on them)
- key: $E/key.json
Grade TWO answer sets side by side, item by item, applying the same standard to both (paired grading):
- A: $A -> write $O/${x}_base.json
- B: $B -> write $O/${x}_tuned.json
Each output is JSON {\"total\",\"max\",\"items\":[{\"id\",\"points\",\"max\",\"why\"}]}. Do not modify any other files. Reply with only the A total and the B total."
