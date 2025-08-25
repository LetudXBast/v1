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

# --- Fonctions utilitaires ---
# ensure_dirs() → crée data/ si manquant.
# _latin1_safe() → remplace les caractères hors Latin-1 par des équivalents ASCII.
# mistral_code_text() → construit le prompt, appelle l’API Mistral, nettoie la réponse.
# build_pdf_from_text() → génère un PDF à partir d'un texte donné.

# def ensure_dirs():
#     os.makedirs(DATA_DIR, exist_ok=True)

# mistral_code_text() → construit le prompt, appelle l’API Mistral, nettoie la réponse.
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
            "Tu es expert en codage PMSI hospitalier. À partir des observations médicales ci-dessous, produis une **proposition de codage PMSI** structurée comme suit : \n"
             " - Identifier le **Diagnostic Principal (DP)** suggéré selon les règles PMSI (pathologie ayant motivé l’hospitalisation et consommé l’essentiel des ressources), en privilégiant celui qui maximise la sévérité.\n"
             " - Identifier les **Diagnostics Associés (DS)** pertinents.  "
        )
    elif scheme == "ccam":
        instructions = (
            "Tu es expert en codage PMSI hospitalier. À partir des observations médicales ci-dessous, produis une **proposition de codage PMSI** structurée comme suit :  \n"
            "- Proposer les codages **CCAM** des actes réalisés (imagerie, gestes diagnostiques, thérapeutiques, soins liés à stomie, etc.)."
        )
    else:  # ghm
        instructions = (
            "Tu es expert en codage PMSI hospitalier. À partir des observations médicales ci-dessous, produis une **proposition de codage PMSI** structurée comme suit :  \n"
            "- Proposer le **GHM** correspondant, avec explication du choix.  \n"
            "- Si plusieurs options sont possibles (ex : trouble hydro-électrolytique vs infection abdominale), proposer l’alternative et préciser laquelle maximise la sévérité."
        )
    # # Règles spécifiques
    # if scheme == "cim10":
    #     instructions = (
    #         "Tu es un codeur médical. Fais le codage CIM-10 (diagnostics) strictement.\n"
    #         "- Donne 3 parties: 1) Principaux codes (avec libellés), 2) Codes associés/secondaires, 3) Justification par extraits.\n"
    #         "- Format: liste à puces, 'CODE — Libellé'.\n"
    #         "- Ne crée pas de codes inexistants. Si ambigu, propose 2-3 alternatives plausibles avec conditions.\n"
    #         "- Pas d'explications générales: uniquement le résultat structuré."
    #     )
    # elif scheme == "ccam":
    #     instructions = (
    #         "Tu es un codeur médical. Fais le codage CCAM (actes) strictement.\n"
    #         "- Donne 3 parties: 1) Actes principaux (CODE — Libellé), 2) Actes associés, 3) Justification par extraits.\n"
    #         "- Ajoute côté actes, si pertinent: latéralité, guidage imagerie, voie d'abord.\n"
    #         "- Ne crée pas de codes inexistants. Si ambigu, alternatives plausibles + conditions."
    #     )
    # else:  # ghm
    #     instructions = (
    #         "Tu es un codeur médical. Propose le GHM le plus probable.\n"
    #         "- Donne 3 parties: 1) GHM candidat(s) (CODE — Libellé), 2) Diagnostics/actes clés motivants, 3) Justification par extraits.\n"
    #         "- Si l'information est insuffisante, indique précisément ce qu'il manque."
    #     )

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

# _latin1_safe() → remplace les caractères hors Latin-1 par des équivalents ASCII.
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

# build_pdf_from_text() → génère un PDF à partir d'un texte donné.
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
# @app.route("/frontend/<path:asset>")
# def serve_asset(asset):
#     path = safe_join(FRONTEND_DIR, asset)
#     if not os.path.isfile(path):
#         abort(404)
#     ext = os.path.splitext(asset)[1].lower()
#     mime = {
#         ".js": "application/javascript; charset=utf-8",
#         ".css": "text/css; charset=utf-8",
#         ".html": "text/html; charset=utf-8",
#         ".png": "image/png",
#         ".jpg": "image/jpeg",
#         ".svg": "image/svg+xml",
#     }.get(ext, "application/octet-stream")
#     with open(path, "rb") as f:
#         data = f.read()
#     return data, 200, {"Content-Type": mime}


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


@app.route("/pdf_content", methods=["POST"])
# Génère un pdf : cotation.pdf
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
    # ensure_dirs()
    port = int(os.getenv("PORT", "5000"))   # Render fournit PORT
    app.run(host="0.0.0.0", port=port, debug=False)
