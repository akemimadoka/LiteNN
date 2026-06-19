#!/bin/sh
set -eu

ROOT="$(git rev-parse --show-toplevel)"
SOURCE="$ROOT/scripts/hooks/pre-commit"
TARGET_DIR="$ROOT/.git/hooks"
TARGET="$TARGET_DIR/pre-commit"

if [ ! -f "$SOURCE" ]; then
	echo "Missing hook template: $SOURCE" >&2
	exit 1
fi

mkdir -p "$TARGET_DIR"
cp "$SOURCE" "$TARGET"
chmod +x "$TARGET"

echo "Installed LiteNN pre-commit hook: $TARGET"
