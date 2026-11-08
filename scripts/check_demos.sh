#!/usr/bin/env bash
# Teste de abertura das demos interativas de docs/: verifica a sintaxe de todo
# .js com `node --check`, serve docs/ num servidor estático e abre cada página
# (a inicial e cada demo) num Chromium headless, falhando se o console
# registrar erro de execução, falha de inicialização de seção ou recurso 404.
# Uso: bash scripts/check_demos.sh   (do repo root; BROWSER e PORT opcionais)
set -euo pipefail
cd "$(dirname "$0")/.."

BROWSER="${BROWSER:-$(command -v chromium || command -v google-chrome || command -v google-chrome-stable || command -v chromium-browser || true)}"
if [ -z "$BROWSER" ]; then echo "nenhum Chromium/Chrome encontrado; defina BROWSER" >&2; exit 2; fi
PORT="${PORT:-8765}"

status=0
while IFS= read -r f; do
  node --check "$f" || { echo "::error file=$f::erro de sintaxe"; status=1; }
done < <(find docs -name '*.js' | sort)

python3 -m http.server "$PORT" --directory docs >/dev/null 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do curl -fs -o /dev/null "http://localhost:$PORT/" && break; sleep 0.5; done

pages=("")
for dir in docs/*/; do
  [ -f "$dir/index.html" ] && pages+=("$(basename "$dir")/")
done

for page in "${pages[@]}"; do
  log=$("$BROWSER" --headless=new --disable-gpu --no-sandbox --enable-logging=stderr --v=0 \
        --virtual-time-budget=8000 --window-size=1280,900 "http://localhost:$PORT/$page" 2>&1 >/dev/null \
        | grep 'CONSOLE' || true)
  errors=$(printf '%s\n' "$log" | grep -iE 'Falha ao iniciar|Uncaught|ReferenceError|TypeError|RangeError|SyntaxError|Failed to load resource|status of 404' || true)
  if [ -n "$errors" ]; then
    printf '::error::docs/%s: %s\n' "$page" "$errors"
    status=1
  else
    echo "ok docs/${page:-index.html}"
  fi
done
exit $status
