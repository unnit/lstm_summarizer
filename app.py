from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from utils import build_vocab, vectorize, detokenize
from model import Encoder, Decoder
import os

app = Flask(__name__)

# Load model and vocab
checkpoint = torch.load("model/seq2seq_model.pth", map_location=torch.device("cpu"))
text_vocab = checkpoint["text_vocab"]
summary_vocab = checkpoint["summary_vocab"]
inv_summary_vocab = checkpoint["inv_summary_vocab"]

encoder = Encoder(len(text_vocab), 128, 256)
decoder = Decoder(len(summary_vocab), 128, 256)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])
encoder.eval()
decoder.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()
    summary_type = data.get("type", "paragraph")

    if not text or summary_type not in ["paragraph", "bullets", "two_paragraphs", "paragraph_bullet"]:
        return jsonify({"error": "Invalid input"}), 400

    format_token = f"format_{summary_type}"
    input_tensor = vectorize(text, text_vocab, add_tokens=False).unsqueeze(0)

    with torch.no_grad():
        h, c = encoder(input_tensor)
        dec_input = torch.tensor([summary_vocab[format_token]])
        output_tokens = []
        for _ in range(150):
            out, h, c = decoder(dec_input, h, c)
            pred_token = out.argmax(dim=-1).item()
            if pred_token == summary_vocab.get("<EOS>", 1):
                break
            output_tokens.append(pred_token)
            dec_input = torch.tensor([pred_token])

    summary = detokenize(output_tokens, inv_summary_vocab)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True, port=5003)
