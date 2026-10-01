# Personalised Floor Plan Furniture POC

Streamlit app: upload a floor-plan PDF, detect spaces (GPT-4o), pick up to 3 furniture items per space from the Dezinebox quiz catalog, then generate one clean 2D furnished plan with **gpt-image-2** (no dimensions/labels).

## Setup

```bash
cd personalised-fp-ui
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## Run

```bash
streamlit run app.py
```

## Flow

1. Upload PDF → preview
2. Detect spaces (or skip to use full catalog for UI checks)
3. Select multiple spaces + up to 3 furniture each
4. Generate → clean plan → one final furnished image
5. Sidebar shows estimated API cost (USD + INR at $1=₹95) with Hide/Show

## Catalog

Space and furniture names come from `QUIZ FOR REQURIEMENT_DEZINEBOX .xlsx` (names only). Snapshot: `data/furniture_catalog.json`. Regenerate:

```bash
python scripts/build_catalog.py
```
