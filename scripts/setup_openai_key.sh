#!/usr/bin/env bash
set -euo pipefail
mkdir -p .streamlit
printf "OpenAI API-Key eingeben (Eingabe wird nicht angezeigt): "
stty -echo
read OPENAI_KEY
stty echo
printf "\n"
cat > .streamlit/secrets.toml <<TOML
OPENAI_API_KEY = "$OPENAI_KEY"
OPENAI_MODEL = "gpt-4.1-mini"
TOML
chmod 600 .streamlit/secrets.toml
printf "Key wurde lokal in .streamlit/secrets.toml gespeichert.\n"
