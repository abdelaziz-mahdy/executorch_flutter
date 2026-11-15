#!/usr/bin/env bash
# Prints the "What's changed" part of the GitHub release notes for VERSION:
# each package's CHANGELOG section for that version, with its pub.dev link.
# A package without a section for VERSION is left out (packages can release
# on their own). Run from the repo root:  scripts/release_notes.sh 0.9.1
set -euo pipefail

VERSION="${1:?usage: release_notes.sh VERSION}"
found=0

for pkg in executorch_flutter executorch_dart; do
  # The lines after "## VERSION" (exactly that version, not 0.9.10 for
  # 0.9.1) up to the next "## " heading, without blank lines at either end.
  section=$(awk -v v="$VERSION" '
    !inside && $0 == "## " v { inside = 1; next }
    inside && /^## / { exit }
    inside { print }
  ' "packages/$pkg/CHANGELOG.md" | sed -e '/./,$!d' | sed -e ':a' -e '/^\n*$/{$d;N;ba' -e '}')

  if [ -n "$section" ]; then
    found=1
    echo "## $pkg $VERSION"
    echo
    echo "$section"
    echo
    echo "📦 [$pkg on pub.dev](https://pub.dev/packages/$pkg/versions/$VERSION)"
    echo
  fi
done

if [ "$found" = 0 ]; then
  echo "Release version $VERSION"
fi
