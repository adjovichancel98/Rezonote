from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
from transformers import pipeline
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import torch

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Whisper pour la transcription
model_whisper = whisper.load_model("medium", device=device)

# 🚀 Nouveau pipeline ROBUSTE et RAPIDE pour résumé français
summarizer_pipeline = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    device=device
)

# 🚀 Fonction stable pour résumé rapide
def generer_resume(texte):
    resultat = summarizer_pipeline(
        texte,
        max_length=150,
        min_length=40,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )
    return resultat[0]['summary_text'] if resultat else "Résumé indisponible."

@app.route('/transcrire', methods=['POST'])
def transcrire():
    if 'audio' not in request.files:
        return jsonify({"error": "❌ Aucun fichier audio trouvé"}), 400

    audio = request.files['audio']
    ext = audio.filename.rsplit('.', 1)[-1].lower()
    if ext not in ['mp3', 'wav', 'm4a']:
        return jsonify({"error": "❌ Format non supporté. (mp3, wav, m4a uniquement)"}), 400

    chemin = f"temp_audio.{ext}"
    audio.save(chemin)

    try:
        result = model_whisper.transcribe(chemin, language="fr", beam_size=3, fp16=torch.cuda.is_available())
        texte_transcrit = result.get("text", "").strip()

        if len(texte_transcrit.split()) < 10:
            return jsonify({
                "texte": texte_transcrit,
                "resume": "⚠️ Trop peu de texte pour générer un résumé pertinent."
            })

        phrases = sent_tokenize(texte_transcrit, language='french')
        texte_segmente, temp_chunk, compteur_mots = [], "", 0
        for phrase in phrases:
            longueur_phrase = len(phrase.split())
            if compteur_mots + longueur_phrase <= 400:
                temp_chunk += " " + phrase
                compteur_mots += longueur_phrase
            else:
                texte_segmente.append(temp_chunk.strip())
                temp_chunk, compteur_mots = phrase, longueur_phrase
        if temp_chunk:
            texte_segmente.append(temp_chunk.strip())

        resume_segments = []
        for chunk in texte_segmente:
            resume_chunk = generer_resume(chunk)
            resume_segments.append(resume_chunk)

        resume_final = " ".join(resume_segments).strip()

        return jsonify({
            "texte": texte_transcrit,
            "resume": resume_final
        })

    except Exception as e:
        print(f"❌ Erreur serveur : {type(e).__name__} – {str(e)}")
        return jsonify({
            "texte": "❌ Erreur pendant la transcription/résumé.",
            "resume": f"{type(e).__name__}: {str(e)}"
        }), 500

    finally:
        if os.path.exists(chemin):
            os.remove(chemin)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001)
