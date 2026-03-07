#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  replace_inductor.sh [SOURCE_INDUCTOR_DIR]

Examples:
  ./replace_inductor.sh
  ./replace_inductor.sh /home/user/pytorch/torch/_inductor
  PYTHON_BIN=/path/to/python ./replace_inductor.sh /home/user/pytorch/torch/_inductor

Behavior:
  - If SOURCE_INDUCTOR_DIR is omitted:
      1) if current directory name is "_inductor", use current directory
      2) else if "./_inductor" exists, use that
      3) else fail
  - Finds installed torch path from the active python
  - Backs up installed torch/_inductor
  - Replaces it with the source _inductor
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# 1) source _inductor 결정
if [[ $# -ge 1 ]]; then
  SRC_INDUCTOR="$(cd "$1" && pwd)"
else
  if [[ "$(basename "$PWD")" == "_inductor" ]]; then
    SRC_INDUCTOR="$PWD"
  elif [[ -d "$PWD/_inductor" ]]; then
    SRC_INDUCTOR="$PWD/_inductor"
  else
    echo "[ERROR] SOURCE_INDUCTOR_DIR를 찾을 수 없습니다."
    echo "        현재 디렉토리가 _inductor 이거나, ./_inductor 가 있어야 합니다."
    echo "        또는 인자로 경로를 직접 넘겨주세요."
    usage
    exit 1
  fi
fi

if [[ ! -d "$SRC_INDUCTOR" ]]; then
  echo "[ERROR] source _inductor directory not found: $SRC_INDUCTOR"
  exit 1
fi

if [[ "$(basename "$SRC_INDUCTOR")" != "_inductor" ]]; then
  echo "[ERROR] source directory name must be '_inductor': $SRC_INDUCTOR"
  exit 1
fi

# 2) torch 설치 경로 찾기
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] python not found: $PYTHON_BIN"
  exit 1
fi

TORCH_DIR="$("$PYTHON_BIN" - <<'PY'
import os
import sys
try:
    import torch
except Exception as e:
    print(f"[PYTHON ERROR] failed to import torch: {e}", file=sys.stderr)
    sys.exit(1)
print(os.path.dirname(torch.__file__))
PY
)"

DST_INDUCTOR="${TORCH_DIR}/_inductor"

if [[ ! -d "$DST_INDUCTOR" ]]; then
  echo "[ERROR] installed torch _inductor not found: $DST_INDUCTOR"
  exit 1
fi

# 3) 자기 자신 복사 방지
SRC_REAL="$(readlink -f "$SRC_INDUCTOR")"
DST_REAL="$(readlink -f "$DST_INDUCTOR")"

if [[ "$SRC_REAL" == "$DST_REAL" ]]; then
  echo "[INFO] source and destination are already the same:"
  echo "       $SRC_REAL"
  exit 0
fi

# 4) 백업
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${DST_INDUCTOR}.backup.${STAMP}"

echo "[INFO] python      : $(command -v "$PYTHON_BIN")"
echo "[INFO] torch dir   : $TORCH_DIR"
echo "[INFO] source      : $SRC_INDUCTOR"
echo "[INFO] destination : $DST_INDUCTOR"
echo "[INFO] backup      : $BACKUP_DIR"

mv "$DST_INDUCTOR" "$BACKUP_DIR"

# 5) 복사
cp -a "$SRC_INDUCTOR" "$DST_INDUCTOR"

# 6) 검증
echo "[INFO] verification..."
"$PYTHON_BIN" - <<PY
import os
import torch
dst = os.path.join(os.path.dirname(torch.__file__), "_inductor")
print("torch.__file__ =", torch.__file__)
print("installed _inductor =", dst)
print("exists =", os.path.isdir(dst))
PY

echo "[DONE] installed torch/_inductor has been replaced."
echo "[DONE] backup saved at: $BACKUP_DIR"
