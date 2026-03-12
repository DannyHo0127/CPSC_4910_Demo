# =============================================================================
# app.py
# Flask backend: exposes RAG (nlp.answer) as POST /api/chat for the CAPC UI.
# Run from project root: python code/app.py
# =============================================================================

import os
import sys
from pathlib import Path

# Ensure "code" is on path so we can import nlp and retrieval
CODE_DIR = Path(__file__).resolve().parent
DOCS_DIR = CODE_DIR.parent / "docs"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.chdir(CODE_DIR)

import flask
import nlp
import retrieval

app = flask.Flask(__name__)

# -----------------------------------------------------------------------------
# Load RAG data once at startup (local Chroma + sparse from Chroma)
# -----------------------------------------------------------------------------
CHROMA_PATH = os.environ.get("CAPC_CHROMA_PATH", str(CODE_DIR))
sparse_collection = None
dense_collection = None


def load_rag():
    global sparse_collection, dense_collection
    dense_collection = retrieval.load_chromadb_local(CHROMA_PATH)
    sparse_collection = retrieval.build_sparse_from_chroma(dense_collection)


# RAG is loaded on first /api/chat request inside the view (so we can return a proper error response)


# -----------------------------------------------------------------------------
# CORS for frontend (docs/ served from same or different origin)
# -----------------------------------------------------------------------------
@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/health", methods=["GET"])
def health():
    return flask.jsonify({"status": "ok"}), 200


@app.route("/api/chat", methods=["OPTIONS"])
def chat_options():
    return "", 204


# -----------------------------------------------------------------------------
# POST /api/chat — body: { "query": "..." } → { "response": "..." }
# -----------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    global sparse_collection, dense_collection
    try:
        body = flask.request.get_json(force=True, silent=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return flask.jsonify({"error": "Missing or empty 'query'"}), 400

        if dense_collection is None or sparse_collection is None:
            try:
                load_rag()
            except Exception as e:
                return flask.jsonify({"error": "RAG failed to load: " + str(e)}), 500

        response = nlp.answer(
            query,
            sparse_collection=sparse_collection,
            dense_collection=dense_collection,
            hybrid_top_k=15,
            rerank_top_n=7,
        )
        return flask.jsonify({"response": response})
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# Serve UI from same origin (avoids CORS / file:// issues)
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return flask.send_from_directory(DOCS_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    if path.startswith("api/"):
        flask.abort(404)
    return flask.send_from_directory(DOCS_DIR, path)


# -----------------------------------------------------------------------------
# Run with: python code/app.py  (from project root) or python app.py (from code/)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Bind immediately; RAG loads on first /api/chat request
    port = int(os.environ.get("PORT", 5000))
    print("Backend starting at http://localhost:%s (RAG loads on first chat)" % port)
    app.run(host="0.0.0.0", port=port, debug=True)
