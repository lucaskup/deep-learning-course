#!/usr/bin/env bash
# Teste de abertura das demos interativas de docs/: verifica a sintaxe de todo
# .js com `node --check`, serve docs/ num servidor estático e abre cada página
# (a inicial e cada demo) num Chromium headless, falhando se o console
# registrar erro de execução, falha de inicialização de seção ou recurso 404.
# Uso: bash scripts/check_demos.sh   (do repo root; BROWSER e PORT opcionais)
# Saída: 0 tudo certo; 1 erro em alguma página ou .js; 2 sem navegador;
# 3 o servidor de teste não subiu na porta (porta ocupada).
set -euo pipefail
cd "$(dirname "$0")/.."

BROWSER="${BROWSER:-$(command -v chromium || command -v google-chrome || command -v google-chrome-stable || command -v chromium-browser || true)}"
if [ -z "$BROWSER" ]; then echo "nenhum Chromium/Chrome encontrado; defina BROWSER" >&2; exit 2; fi
PORT="${PORT:-8765}"

status=0
while IFS= read -r f; do
  node --check "$f" || { echo "::error file=$f::erro de sintaxe"; status=1; }
done < <(find docs -name '*.js' | sort)

# Servidor próprio, só em 127.0.0.1. Se a porta já estiver ocupada, o python
# encerra com erro e o teste para aqui: nunca se testa um servidor alheio.
BASE="http://127.0.0.1:$PORT"
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory docs >/dev/null 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

server_fail() {
  echo "::error::$1 (PORT=$PORT). Libere a porta ou rode com outra: PORT=8766 bash scripts/check_demos.sh" >&2
  exit 3
}
up=0
for _ in $(seq 1 20); do
  kill -0 "$SERVER" 2>/dev/null || server_fail "o servidor de teste não subiu; a porta provavelmente está em uso"
  if curl -fs -o /dev/null "$BASE/"; then up=1; break; fi
  sleep 0.5
done
[ "$up" = 1 ] || server_fail "o servidor de teste não respondeu em 10 s"
# Outro processo pode ter respondido enquanto o nosso ainda tentava abrir a
# porta: confirma que o nosso continua vivo e que quem responde serve este docs/.
sleep 0.5
kill -0 "$SERVER" 2>/dev/null || server_fail "o servidor de teste encerrou logo após iniciar; a porta provavelmente está em uso"
curl -fs "$BASE/style.css" | cmp -s - docs/style.css || server_fail "quem responde na porta não serve o docs/ deste repositório"

pages=("")
for dir in docs/*/; do
  [ -f "$dir/index.html" ] && pages+=("$(basename "$dir")/")
done

for page in "${pages[@]}"; do
  log=$("$BROWSER" --headless=new --disable-gpu --no-sandbox --enable-logging=stderr --v=0 \
        --virtual-time-budget=8000 --window-size=1280,900 "$BASE/$page" 2>&1 >/dev/null \
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
