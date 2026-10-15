#!/bin/zsh
# Arma tesis_overleaf.zip con los .tex, la bibliografía y solo las figuras usadas.
# En Overleaf: New Project > Upload Project, y en Menu > Compiler elegir XeLaTeX.
set -euo pipefail
cd "$(dirname "$0")"

tmp=$(mktemp -d)
mkdir -p "$tmp/tesis/figuras"
cp main.tex referencias.bib latexmkrc "$tmp/tesis/"
mkdir -p "$tmp/tesis/capitulos" && cp capitulos/*.tex "$tmp/tesis/capitulos/"

for fig in $(rg -o --no-filename 'includegraphics\[[^]]*\]\{[^}]+\}' capitulos | sed -E 's/.*\{([^}]+)\}$/\1/' | sort -u); do
  if [[ -f "figuras/$fig" ]]; then
    cp "figuras/$fig" "$tmp/tesis/figuras/"
  else
    cp "../paper_figuras/$fig" "$tmp/tesis/figuras/"
  fi
done

rm -f tesis_overleaf.zip
(cd "$tmp/tesis" && zip -qr "$OLDPWD/tesis_overleaf.zip" .)
rm -rf "$tmp"
echo "Listo: $(pwd)/tesis_overleaf.zip"
