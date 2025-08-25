import os
import json # os, json = gestion fichiers et JSON
from flask_cors import CORS # CORS = autorise l’accès depuis GitHub Pages (cross-origin)
from datetime import datetime # datetime = date/heure pour logs et PDF
from io import BytesIO # BytesIO = fichier en mémoire (PDF)

import requests # requests = appel API Mistral
from flask import Flask, request, jsonify, send_file, abort # flask = serveur web + API
from werkzeug.utils import safe_join # safe_join = sécurise les chemins fichiers (évite des accès non autorisés)

from dotenv import load_dotenv  # Charge les variables secrètes depuis backend/.env (ex. MISTRAL_API_KEY).
# Charge le .env situé à côté de app.py (backend/.env)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env")) # ← charge le .env AVANT de lire les variables


# --- Config basique --- 
# Définit les chemins vers tes dossiers :
# frontend/ pour l’UI,
# data/QR.txt pour stocker questions/réponses,
# prompts.txt pour les instructions IA.

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
QR_PATH = os.path.join(DATA_DIR, "QR.txt")
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts.txt")

# Récupère ta clé API (depuis .env),
# Prépare l’URL et le modèle IA à utiliser.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")  # facultatif pour dev sans réseau
print("MISTRAL_API_KEY chargée:", "OK" if bool(MISTRAL_API_KEY) else "ABSENTE") #Lance l’app. Tu dois voir OK en console.
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"  # simple et économique

# Crée le serveur Flask
# Active CORS pour que GitHub Pages puisse l’appeler.
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Utilitaires ---
# Fonctions utilitaires
# ensure_dirs() → crée data/ si manquant.
# read_prompts() → lit prompts.txt ou donne un texte par défaut.
# mistral_generate_question() → construit le prompt, appelle l’API Mistral, nettoie la réponse.
# append_qr_block() → ajoute les Q/R dans QR.txt avec horodatage.
# build_pdf_from_qr() → lit QR.txt et génère un PDF en mémoire.
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def read_prompts():
    if os.path.isfile(PROMPTS_PATH):
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return (
        "Tu es un générateur de questions concises pour un entretien."
        " Pose UNE seule question pertinente à la fois, sans préambule."
    )

def mistral_generate_question(prompt_text: str, previous_questions=None) -> str:
    """
    Retourne une (1) question.
    Utilise previous_questions pour éviter les répétitions et varier l'angle.
    """
    previous_questions = previous_questions or []
    if not MISTRAL_API_KEY:
        # Mode dégradé local : variation simple côté serveur
        # On évite de répéter exactement la dernière question
        base = "Pouvez-vous préciser votre objectif principal ?"
        if previous_questions and base in previous_questions:
            return "Quel résultat concret voulez-vous obtenir en premier ?"
        return base

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    # Contexte anti-répétition + consignes de diversité
    previous_blob = "\n".join(f"- {q}" for q in previous_questions if q)
    user_msg = (
        (prompt_text or "Génère une seule question pertinente.") +
        (
            f"\n\nQuestions déjà posées (à NE PAS répéter ni paraphraser):\n{previous_blob}"
            if previous_blob else ""
        ) +
        "\n\nContraintes:\n"
        "- Propose une question NOUVELLE, couvrant un angle non traité.\n"
        "- Une seule phrase. Pas de préambule. 5–18 mots.\n"
    )
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": read_prompts()},
            {"role": "user", "content": user_msg}
        ],
        # ↑ diversité raisonnable :
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 64,
    }
    resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # Format Mistral: data["choices"][0]["message"]["content"]
    content = (
        data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
    )
    # Nettoyage simple si le modèle renvoie du superflu
    # => On prend la première ligne non vide.
    for line in content.splitlines():
        line = line.strip(" \t-–—•:").strip()
        if line:
            return line
    return content or "Pouvez-vous développer votre contexte ?"

def mistral_code_text(text: str, scheme: str) -> str:
    """
    Code un texte clinique selon le schéma demandé: 'cim10' | 'ccam' | 'ghm'.
    Retourne un bloc texte structuré (codes + libellés + justification).
    """
    assert scheme in {"cim10", "ccam", "ghm"}
    if not text or not text.strip():
        return "Erreur: texte vide."

    # Mode dégradé si pas de clé
    if not MISTRAL_API_KEY:
        return f"[MODE LOCAL] {scheme.upper()} — exemple de sortie:\n- CODE: XXX.XX — Libellé: Exemple\n- Justification: Indices dans le texte."

    # Règles spécifiques
    if scheme == "cim10":
        instructions = (
            "Tu es un codeur médical. Fais le codage CIM-10 (diagnostics) strictement.\n"
            "- Donne 3 parties: 1) Principaux codes (avec libellés), 2) Codes associés/secondaires, 3) Justification par extraits.\n"
            "- Format: liste à puces, 'CODE — Libellé'.\n"
            "- Ne crée pas de codes inexistants. Si ambigu, propose 2-3 alternatives plausibles avec conditions.\n"
            "- Pas d'explications générales: uniquement le résultat structuré."
        )
    elif scheme == "ccam":
        instructions = (
            "Tu es un codeur médical. Fais le codage CCAM (actes) strictement.\n"
            "- Donne 3 parties: 1) Actes principaux (CODE — Libellé), 2) Actes associés, 3) Justification par extraits.\n"
            "- Ajoute côté actes, si pertinent: latéralité, guidage imagerie, voie d'abord.\n"
            "- Ne crée pas de codes inexistants. Si ambigu, alternatives plausibles + conditions."
        )
    else:  # ghm
        instructions = (
            "Tu es un codeur médical. Propose le GHM le plus probable.\n"
            "- Donne 3 parties: 1) GHM candidat(s) (CODE — Libellé), 2) Diagnostics/actes clés motivants, 3) Justification par extraits.\n"
            "- Si l'information est insuffisante, indique précisément ce qu'il manque."
        )

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Texte clinique à coder ({scheme.upper()}):\n\n{text.strip()}"}
        ],
        "temperature": 0.2,     # codage = plutôt déterministe
        "top_p": 0.9,
        "max_tokens": 512
    }
    resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    content = (
        data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
    )
    return content or "Aucun résultat renvoyé par le modèle."


def append_qr_block(pairs, timestamp_iso):
    """
    pairs = [{id, question, answer}]
    Ecrit un bloc lisible dans QR.txt
    """
    ensure_dirs()
    lines = []
    lines.append(f"=== Session @ {timestamp_iso} ===")
    for item in pairs:
        q = (item.get("question") or "").replace("\n", " ").strip()
        a = (item.get("answer") or "").replace("\n", " ").strip()
        i = item.get("id", "?")
        lines.append(f"Q{i}: {q}")
        lines.append(f"R{i}: {a if a else '(vide)'}")
    lines.append("---")
    lines.append("")  # newline final

    with open(QR_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _latin1_safe(s: str) -> str:
    """Remplace les caractères hors Latin-1 par des équivalents ASCII."""
    if not s:
        return ""
    # remplacements ciblés fréquents
    replacements = {
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2012": "-",  # figure dash
        "\u2212": "-",  # minus math
        "\u00A0": " ",  # espace insécable
        "\u2018": "'", "\u2019": "'",  # apostrophes typographiques
        "\u201C": '"', "\u201D": '"',  # guillemets typographiques
        "\u2022": "*", "\u00B7": "*",  # puces
        "\u2026": "...",               # points de suspension
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Dernier filet de sécurité: remplace le reste par '?'
    return s.encode("latin-1", "replace").decode("latin-1")

def build_pdf_from_qr() -> BytesIO:
    """
    Lit QR.txt et construit un PDF simple en mémoire avec FPDF.
    """
    from fpdf import FPDF

    ensure_dirs()
    text = ""
    if os.path.isfile(QR_PATH):
        with open(QR_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = "Aucune donnée disponible (QR.txt inexistant)."

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Police de base (intégrée), pas besoin d'ajouter de fichier .ttf
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Questions / Réponses", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", size=12)
    for line in text.splitlines():
        pdf.multi_cell(0, 7, line if line.strip() else " ")

    # ✅ sortie mémoire correcte pour pyfpdf (1.x)
    buf = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buf.write(pdf_bytes)
    buf.seek(0)
    return buf

def build_pdf_from_text(title: str, subtitle: str, body: str) -> BytesIO:
    """
    Construit un PDF simple en mémoire (titre, sous-titre, corps multi-lignes).
    """
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 🔽 normalisation ici
    title = _latin1_safe(title)
    subtitle = _latin1_safe(subtitle)
    body = _latin1_safe(body)

    # Titre
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, title or "Document", ln=True)

    # Sous-titre + horodatage
    pdf.set_font("Arial", size=10)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subtitle = (subtitle or "").strip()
    if subtitle:
        pdf.multi_cell(0, 7, subtitle)
    pdf.cell(0, 8, f"Généré le {ts}", ln=True)
    pdf.ln(4)

    # Corps
    pdf.set_font("Arial", size=12)
    body = (body or "").replace("\r\n", "\n")
    for line in body.split("\n"):
        pdf.multi_cell(0, 7, line if line.strip() else " ")

    buf = BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    buf.write(pdf_bytes)
    buf.seek(0)
    return buf





# --- Routes Frontend (sert index.html depuis /) ---
@app.route("/")
def serve_index():
    path = safe_join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(path):
        abort(404, "frontend/index.html introuvable")
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return html

# Optionnel : servir d'autres assets si tu en ajoutes plus tard
@app.route("/frontend/<path:asset>")
def serve_asset(asset):
    path = safe_join(FRONTEND_DIR, asset)
    if not os.path.isfile(path):
        abort(404)
    ext = os.path.splitext(asset)[1].lower()
    mime = {
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        data = f.read()
    return data, 200, {"Content-Type": mime}

# --- API attendue par le frontend ---
# Génère une question Mistral, en tenant compte des questions déjà posées.
@app.route("/ask", methods=["GET", "POST"])
def api_ask():
    """
    Renvoie une question générée par Mistral selon prompts.txt
    Réponse JSON: { "question": "..." }
    """
    try:
        previous = []
        if request.method == "POST":
            payload = request.get_json(force=True, silent=True) or {}
            previous = payload.get("previous_questions", []) or []

        question = mistral_generate_question(
            prompt_text="Génère UNE question.",
            previous_questions=previous
        )
        return jsonify({"question": question})
    except requests.HTTPError as e:
        return jsonify({"error": f"HTTP {e.response.status_code} depuis Mistral"}), 502
    except Exception as e:
        return jsonify({
            "question": "Quelle est la priorité n°1 de votre projet ?",
            "warning": str(e)
        }), 200

# Sauvegarde les questions/réponses reçues dans QR.txt.
@app.route("/save", methods=["POST"])
def api_save():
    """
    Reçoit: { pairs:[{id,question,answer}], timestamp }
    Ecrit dans data/QR.txt
    """
    try:
        payload = request.get_json(force=True, silent=False)
        pairs = payload.get("pairs", [])
        timestamp = payload.get("timestamp") or datetime.utcnow().isoformat()
        if not isinstance(pairs, list):
            return jsonify({"error": "Format 'pairs' invalide"}), 400
        append_qr_block(pairs, timestamp)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/code", methods=["POST"])
def api_code():
    """
    Reçoit: { scheme: 'cim10'|'ccam'|'ghm', text: '...' }
    Renvoie: { result: '...' }
    """
    try:
        payload = request.get_json(force=True, silent=False)
        scheme = (payload.get("scheme") or "").lower().strip()
        text = payload.get("text") or ""
        if scheme not in {"cim10", "ccam", "ghm"}:
            return jsonify({"error": "Paramètre 'scheme' invalide (cim10|ccam|ghm)"}), 400
        if not text.strip():
            return jsonify({"error": "Paramètre 'text' vide"}), 400

        result = mistral_code_text(text, scheme)
        return jsonify({"result": result})
    except requests.HTTPError as e:
        return jsonify({"error": f"HTTP {e.response.status_code} depuis Mistral"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Génère un PDF et l’envoie directement au navigateur.
@app.route("/pdf", methods=["GET"])
def api_pdf():
    """
    Génére un PDF à partir de data/QR.txt et renvoie le binaire.
    """
    try:
        buf = build_pdf_from_qr()
        filename = "questions_reponses.pdf"
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
            max_age=0,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pdf_content", methods=["POST"])
def api_pdf_content():
    """
    Reçoit JSON: { title: str, subtitle: str, content: str }
    Renvoie un PDF à télécharger.
    """
    try:
        payload = request.get_json(force=True, silent=False)
        title = payload.get("title", "Document")
        subtitle = payload.get("subtitle", "")
        content = payload.get("content", "")
        buf = build_pdf_from_text(title, subtitle, content)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="cotation.pdf",
            max_age=0,
        )
    except Exception as e:
        # Log utile pour comprendre si ça recasse
        print("Erreur /pdf_content:", repr(e))
        return jsonify({"error": str(e)}), 500


# --- Lancement ---
# Lance le serveur, écoute le port (Render fournit automatiquement PORT).
if __name__ == "__main__":
    ensure_dirs()
    port = int(os.getenv("PORT", "5000"))   # Render fournit PORT
    app.run(host="0.0.0.0", port=port, debug=False)
