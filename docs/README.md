# CAPC_UI

## Test locally

From the **project root** (`CAPC_UI/`):

```bash
pip install -r code/requirements.txt   # once
python run_local.py
```

Then open **http://localhost:5000** in your browser. The app serves both the UI and the API on the same origin, so the chat works without CORS or a separate docs server. Stop with Ctrl+C.

**Or run the backend only:**

```bash
python code/app.py
```

Then open http://localhost:5000.

---

## RAG integration

The chat UI uses the local RAG pipeline (Chroma + `nlp.answer`) instead of the Google Gemini API.

### Run the RAG backend only

From the **project root** (`CAPC_UI/`):

```bash
pip install -r code/requirements.txt   # if needed
python code/app.py
```

Backend runs at `http://localhost:5000`. It loads the Chroma DB from the `code` folder and serves `POST /api/chat` with body `{ "query": "..." }` and response `{ "response": "..." }`.

### UI and API on one server

The Flask app serves the UI at `/` and the API at `/api/chat`, so you only need to run one server and open http://localhost:5000.

### GitHub Pages

GitHub Pages serves **only static files** (HTML, CSS, JS). It does not run the RAG backend (Flask, Chroma, Ollama). So:

- **If you host only the `docs` folder on GitHub Pages:** The page will load, but the chat will show a message that the backend is not available. Visitors need to run the backend locally and use the UI from a local server, or you can deploy the backend elsewhere (see below).
- **To have the chat work from your GitHub Pages URL:** Deploy the RAG backend to a separate service (e.g. Railway, Render, Fly.io) that can run Python and your stack. Then set the frontend to use that URL by adding a script before `script.js` in `index.html` that sets `window.CAPC_RAG_API_URL = 'https://your-backend.example.com/api/chat';`. The UI on GitHub Pages will then call your deployed backend.