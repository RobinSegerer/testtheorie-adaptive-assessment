# Testtheorie-Kompetenztest

Streamlit-Prototyp mit drei schlanken Modi:

1. **Volltest: Routing → Phase 1 → Phase 2**
   - Startet mit zufällig stratifizierter Routingphase.
   - Danach adaptive Phase 1.
   - Springt automatisch in Phase 2, wenn das Gate offen ist (`theta >= 3.00` und `SE <= 0.35`).
   - Online-Generierung für Phase 1 und Phase 2 ist separat zuschaltbar.

2. **Nur Phase 1 (mit Online-Items)**
   - Routing + adaptive Phase 1.
   - Kein Sprung zu Phase 2.
   - Online-Items nach dem Routing zuschaltbar.

3. **Nur Phase 2 (mit Online-Items)**
   - Direkt offene Fallstudien.
   - Online-Fallstudien zuschaltbar.

Phase 1 zeigt nach jeder Antwort richtig/falsch, Begründung, theta und SE.
Phase 2 nutzt rubriziertes Scoring.

## Installation

```bash
cd ~/Desktop
rm -rf testtheorie_adaptive_prototype
unzip testtheorie_adaptive_prototype_clean_sidebar.zip
cd testtheorie_adaptive_prototype

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## OpenAI-Key im Hintergrund hinterlegen

```bash
mkdir -p .streamlit

printf "OpenAI API Key: "
stty -echo
read OPENAI_KEY
stty echo
printf "\n"

printf '\nOPENAI_API_KEY = "%s"\n' "$OPENAI_KEY" > .streamlit/secrets.toml
chmod 600 .streamlit/secrets.toml

echo "Key gespeichert."
```

## Start

```bash
streamlit run app.py
```
