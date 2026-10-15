#!/usr/bin/env bash
# Apply patches/skim/toolqa.patch to the pinned SKIM submodule (idempotent).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SKIM="${REPO}/third_party/SKIM"
PATCH="${REPO}/patches/skim/toolqa.patch"
git -C "${REPO}" submodule update --init third_party/SKIM
if git -C "${SKIM}" apply --check "${PATCH}" 2>/dev/null; then
  git -C "${SKIM}" apply "${PATCH}" && echo "applied $(basename "${PATCH}")"
elif git -C "${SKIM}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "already applied $(basename "${PATCH}")"
else
  echo "cannot apply ${PATCH}" >&2; exit 1
fi
