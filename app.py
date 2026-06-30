import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from sqlalchemy import func, text
from models import db, UploadSession, Sample

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(BASE_DIR, "water.db")
)
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "connect_args": {"timeout": 30, "check_same_thread": False},
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()
    with db.engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.commit()

# ── Neural Network Definitions ───────────────────────────────────────────────
class WaterNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class DiseaseNet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):
        return self.net(x)

# ── Load Models ──────────────────────────────────────────────────────────────
device = torch.device('cpu')

def load_potability_model():
    dl_path = os.path.join("notebooks", "dl_potability_model.pt")
    if os.path.exists(dl_path):
        try:
            ckpt   = torch.load(dl_path, map_location=device)
            net    = WaterNet(ckpt['input_dim']).to(device)
            net.load_state_dict(ckpt['model_state'])
            net.eval()
            scaler = joblib.load(os.path.join("notebooks", "dl_potability_scaler.pkl"))
            feats  = joblib.load(os.path.join("notebooks", "features.pkl"))
            print("✅ Loaded DL potability model")
            return net, scaler, feats, 'dl'
        except Exception as e:
            print(f"DL potability load failed: {e}")
    rf = torch.load(os.path.join("notebooks", "dl_potability_model.pt"))
    scaler = joblib.load(os.path.join("notebooks", "dl_potability_scaler.pkl"))
    feats  = joblib.load(os.path.join("notebooks", "features.pkl"))
    print("✅ Loaded RF potability model")
    return rf, scaler, feats, 'rf'

def load_disease_model():
    dl_path = os.path.join("notebooks", "dl_disease_model.pt")
    if os.path.exists(dl_path):
        try:
            ckpt    = torch.load(dl_path, map_location=device)
            net     = DiseaseNet(ckpt['input_dim'], ckpt['n_classes']).to(device)
            net.load_state_dict(ckpt['model_state'])
            net.eval()
            scaler  = joblib.load(os.path.join("notebooks", "dl_disease_scaler.pkl"))
            encoder = joblib.load(os.path.join("notebooks", "dl_disease_encoder.pkl"))
            print("✅ Loaded DL disease model")
            return net, scaler, encoder, 'dl'
        except Exception as e:
            print(f"DL disease load failed: {e}")
    rf      = joblib.load(os.path.join("notebooks", "disease_model.pkl"))
    scaler  = joblib.load(os.path.join("notebooks", "disease_scaler.pkl"))
    encoder = joblib.load(os.path.join("notebooks", "disease_encoder.pkl"))
    print("✅ Loaded RF disease model")
    return rf, scaler, encoder, 'rf'

pot_model,     pot_scaler,     FEATURE_NAMES, pot_type     = load_potability_model()
disease_model, disease_scaler, disease_enc,   disease_type = load_disease_model()

RAW_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon",
    "Trihalomethanes", "Turbidity",
]

DISEASE_COLORS = {
    "None": "#34d399", "Cholera": "#f87171", "Typhoid": "#fb923c",
    "Dysentery": "#f87171", "Fluorosis": "#a78bfa", "Arsenicosis": "#f59e0b",
    "Lead Poisoning": "#ef4444", "Hepatitis A": "#f97316",
}

CRITICAL_DISEASES = ['Cholera', 'Typhoid', 'Hepatitis A']
HIGH_DISEASES     = ['Dysentery', 'Arsenicosis', 'Lead Poisoning']


# ── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    clip_bounds = {
        "ph": (0.5, 13.0), "Hardness": (60.0, 310.0), "Solids": (320.0, 56500.0),
        "Chloramines": (1.0, 13.0), "Sulfate": (150.0, 480.0),
        "Conductivity": (180.0, 750.0), "Organic_carbon": (2.5, 26.0),
        "Trihalomethanes": (8.0, 120.0), "Turbidity": (1.8, 6.5),
    }
    for col, (lo, hi) in clip_bounds.items():
        df[col] = df[col].clip(lo, hi)
    df["ph_safe_range"]       = ((df["ph"] >= 6.5) & (df["ph"] <= 8.5)).astype(int)
    df["ph_squared"]          = df["ph"] ** 2
    df["ph_cubed"]            = df["ph"] ** 3
    df["log_solids"]          = np.log1p(df["Solids"])
    df["log_sulfate"]         = np.log1p(df["Sulfate"])
    df["solids_per_cond"]     = df["Solids"] / (df["Conductivity"] + 1)
    df["chlor_turb"]          = df["Chloramines"] * df["Turbidity"]
    df["thm_carbon"]          = df["Trihalomethanes"] / (df["Organic_carbon"] + 1)
    df["sulfate_hardness"]    = df["Sulfate"] / (df["Hardness"] + 1)
    df["total_contamination"] = df["Chloramines"] + df["Trihalomethanes"] + df["Turbidity"]
    df["mineral_load"]        = df["Hardness"] + df["Solids"] + df["Sulfate"]
    df["hardness_cond"]       = df["Hardness"] / (df["Conductivity"] + 1)
    df["ph_chlor"]            = df["ph"] * df["Chloramines"]
    return df[FEATURE_NAMES]


# ── Prediction Helpers ────────────────────────────────────────────────────────
def predict_potability(df_raw):
    X = pot_scaler.transform(engineer_features(df_raw))
    if pot_type == 'dl':
        with torch.no_grad():
            logits = pot_model(torch.FloatTensor(X))
            probs  = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        confs = np.where(preds == 1, probs, 1 - probs) * 100
    else:
        preds = pot_model.predict(X)
        probs = pot_model.predict_proba(X)
        confs = np.max(probs, axis=1) * 100
    return preds, confs.round(1)


def predict_disease(df_raw):
    X = disease_scaler.transform(df_raw[RAW_COLUMNS])
    if disease_type == 'dl':
        with torch.no_grad():
            logits   = disease_model(torch.FloatTensor(X))
            probs    = torch.softmax(logits, dim=1).numpy()
        preds    = probs.argmax(axis=1)
        diseases = disease_enc.inverse_transform(preds)
        confs    = probs.max(axis=1) * 100
    else:
        preds    = disease_model.predict(X)
        probs    = disease_model.predict_proba(X)
        diseases = disease_enc.inverse_transform(preds)
        confs    = np.max(probs, axis=1) * 100
    return diseases, confs.round(1)


def severity_for(disease):
    if disease in CRITICAL_DISEASES: return 'Critical'
    if disease in HIGH_DISEASES:     return 'High'
    if disease == 'Fluorosis':       return 'Moderate'
    return 'No Risk'

def severity_score_for(disease):
    if disease in CRITICAL_DISEASES: return 3
    if disease in HIGH_DISEASES:     return 2
    if disease == 'Fluorosis':       return 1
    return 0


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def dashboard():
    table_data   = None
    safe_count   = 0
    unsafe_count = 0
    session_id   = None
    model_info   = f"Potability: {pot_type.upper()} | Disease: {disease_type.upper()}"

    total_sessions = UploadSession.query.count()
    all_safe       = db.session.query(func.sum(UploadSession.safe_count)).scalar()   or 0
    all_unsafe     = db.session.query(func.sum(UploadSession.unsafe_count)).scalar() or 0

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            df_raw = pd.read_csv(file)[RAW_COLUMNS]
            for col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            df_raw = df_raw.fillna(df_raw.median(numeric_only=True))

            preds, confs      = predict_potability(df_raw)
            diseases, d_confs = predict_disease(df_raw)

            df_raw["Prediction"]       = ["Safe" if p == 1 else "Unsafe" for p in preds]
            df_raw["Confidence (%)"]   = confs
            df_raw["Disease"]          = diseases
            df_raw["Disease Conf (%)"] = d_confs
            df_raw.loc[df_raw["Prediction"] == "Safe", "Disease"] = "None"

            safe_count   = int((df_raw["Prediction"] == "Safe").sum())
            unsafe_count = int((df_raw["Prediction"] == "Unsafe").sum())

            try:
                sess = UploadSession(filename=file.filename,
                                     safe_count=safe_count,
                                     unsafe_count=unsafe_count)
                db.session.add(sess)
                db.session.flush()
                for _, row in df_raw.iterrows():
                    db.session.add(Sample(
                        session_id=sess.id, ph=row["ph"],
                        hardness=row["Hardness"], solids=row["Solids"],
                        chloramines=row["Chloramines"], sulfate=row["Sulfate"],
                        conductivity=row["Conductivity"],
                        organic_carbon=row["Organic_carbon"],
                        trihalomethanes=row["Trihalomethanes"],
                        turbidity=row["Turbidity"], prediction=row["Prediction"],
                        confidence=row["Confidence (%)"], disease=row["Disease"],
                        disease_confidence=row["Disease Conf (%)"],
                    ))
                db.session.commit()
                session_id = sess.id
            except Exception as e:
                db.session.rollback()
                print(f"DB Error: {e}")

            table_data     = df_raw.to_dict(orient="records")
            total_sessions = UploadSession.query.count()
            all_safe       = db.session.query(func.sum(UploadSession.safe_count)).scalar()   or 0
            all_unsafe     = db.session.query(func.sum(UploadSession.unsafe_count)).scalar() or 0

    return render_template("dashboard.html",
        table_data=table_data, safe_count=safe_count, unsafe_count=unsafe_count,
        session_id=session_id, total_sessions=total_sessions,
        all_safe=all_safe, all_unsafe=all_unsafe,
        disease_colors=DISEASE_COLORS, model_info=model_info)


@app.route("/history")
def history():
    sessions = UploadSession.query.order_by(UploadSession.uploaded_at.desc()).all()
    return render_template("history.html", sessions=sessions)


@app.route("/session/<int:session_id>")
def session_detail(session_id):
    sess    = UploadSession.query.get_or_404(session_id)
    samples = Sample.query.filter_by(session_id=session_id).all()
    return render_template("session_detail.html", sess=sess,
                           samples=samples, disease_colors=DISEASE_COLORS)


@app.route("/session/<int:session_id>/delete", methods=["POST"])
def delete_session(session_id):
    sess = UploadSession.query.get_or_404(session_id)
    db.session.delete(sess)
    db.session.commit()
    return redirect(url_for("history"))


@app.route("/assistant")
def assistant():
    return render_template("assistant.html")


@app.route("/ai-chat", methods=["POST"])
def ai_chat():
    try:
        data    = request.get_json()
        message = data.get("message", "")
        system_prompt = """You are AquaAI Assistant, an expert in water quality and waterborne diseases.
Help users understand water quality parameters, waterborne diseases, test results, prevention, treatment, and WHO guidelines.
Keep answers concise, clear and helpful. Use simple language non-experts can understand.
Format responses with clear paragraphs. Use **bold** for key terms. Use bullet points for lists."""

        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            reply  = result["content"][0]["text"]
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"AI Chat error: {e}")
        return jsonify({"reply": "Sorry, unable to respond right now."}), 500


@app.route("/stats")
def stats():
    total_safe     = db.session.query(func.sum(UploadSession.safe_count)).scalar()   or 0
    total_unsafe   = db.session.query(func.sum(UploadSession.unsafe_count)).scalar() or 0
    total_sessions = UploadSession.query.count()

    disease_counts = db.session.query(
        Sample.disease, func.count(Sample.id).label('count')
    ).group_by(Sample.disease).order_by(func.count(Sample.id).desc()).all()
    disease_data = [{'disease': r.disease, 'count': r.count} for r in disease_counts]

    sessions       = UploadSession.query.order_by(UploadSession.uploaded_at.asc()).all()
    session_labels = [s.filename[:15] for s in sessions]
    session_safe   = [s.safe_count    for s in sessions]
    session_unsafe = [s.unsafe_count  for s in sessions]

    sorted_sessions = sorted(sessions,
                             key=lambda s: s.unsafe_count / max(s.total, 1))
    best_sessions   = sorted_sessions[:3]
    worst_sessions  = sorted_sessions[-3:][::-1]

    session_diseases = []
    for s in sessions[-10:]:
        top = db.session.query(
            Sample.disease, func.count(Sample.id).label('cnt')
        ).filter(Sample.session_id == s.id, Sample.disease != 'None'
        ).group_by(Sample.disease).order_by(func.count(Sample.id).desc()).first()
        session_diseases.append({
            'filename': s.filename[:20],
            'disease':  top.disease if top else 'None',
            'count':    top.cnt     if top else 0,
            'total':    s.total
        })

    return render_template('stats.html',
        total_safe=total_safe, total_unsafe=total_unsafe,
        total_sessions=total_sessions, disease_data=disease_data,
        session_labels=session_labels, session_safe=session_safe,
        session_unsafe=session_unsafe, best_sessions=best_sessions,
        worst_sessions=worst_sessions, session_diseases=session_diseases)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    result = None
    if request.method == "POST":
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        if file1 and file2:
            def process_file(f):
                df = pd.read_csv(f)[RAW_COLUMNS]
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.fillna(df.median(numeric_only=True))
                preds, _          = predict_potability(df)
                diseases, _       = predict_disease(df)
                pot_labels = ["Safe" if p == 1 else "Unsafe" for p in preds]
                safe   = sum(1 for p in pot_labels if p == 'Safe')
                unsafe = len(pot_labels) - safe
                total  = len(pot_labels)
                from collections import Counter
                disease_list    = ['None' if pot_labels[i] == 'Safe'
                                   else diseases[i] for i in range(total)]
                disease_counter = Counter(d for d in disease_list if d != 'None')
                top             = disease_counter.most_common(1)
                return {
                    'safe': safe, 'unsafe': unsafe, 'total': total,
                    'unsafe_pct': round(unsafe / max(total,1) * 100, 1),
                    'top_disease': top[0][0] if top else 'None',
                    'diseases': dict(disease_counter),
                    'severity_score': sum(severity_score_for(d) for d in disease_list),
                }
            r1 = process_file(file1)
            r2 = process_file(file2)
            if r1['severity_score'] > r2['severity_score']:
                verdict = {'worse': file1.filename, 'better': file2.filename}
            elif r2['severity_score'] > r1['severity_score']:
                verdict = {'worse': file2.filename, 'better': file1.filename}
            else:
                verdict = {'worse': 'Equal', 'better': 'Equal'}
            result = {'file1': {'name': file1.filename, **r1},
                      'file2': {'name': file2.filename, **r2},
                      'verdict': verdict}
    return render_template('compare.html', result=result)


@app.route("/report/<int:session_id>")
def download_report(session_id):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Table,
                                        TableStyle, Paragraph, Spacer)
        from reportlab.lib.styles import ParagraphStyle
        from collections import Counter

        session = UploadSession.query.get_or_404(session_id)
        samples = Sample.query.filter_by(session_id=session_id).all()
        total   = session.total

        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4,
                                   rightMargin=2*cm, leftMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        story  = []

        def para(text, size=10, color='#cccccc', bold=False, space=6):
            return Paragraph(text, ParagraphStyle('p',
                fontSize=size,
                textColor=colors.HexColor(color),
                fontName='Helvetica-Bold' if bold else 'Helvetica',
                spaceAfter=space))

        story.append(para('AquaAI Water Quality Report', 20, '#34d399', bold=True, space=4))
        story.append(para(f'File: {session.filename}', 10, '#94a3b8', space=2))
        story.append(para(f'Date: {session.uploaded_at.strftime("%d %b %Y %H:%M")}', 10, '#94a3b8', space=2))
        story.append(para(f'Total Samples: {total}', 10, '#94a3b8', space=12))

        disease_counts = Counter(s.disease for s in samples
                                 if s.disease and s.disease != 'None')
        summary_data = [
            ['Metric', 'Value'],
            ['Safe Samples',   f'{session.safe_count} ({session.safe_pct}%)'],
            ['Unsafe Samples', f'{session.unsafe_count} ({round(100-session.safe_pct,1)}%)'],
        ]
        for disease, count in disease_counts.most_common():
            summary_data.append([f'Disease: {disease}', str(count)])

        t = Table(summary_data, colWidths=[9*cm, 7*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',     (0,0), (-1,0),  colors.HexColor('#1a1d25')),
            ('TEXTCOLOR',      (0,0), (-1,0),  colors.HexColor('#34d399')),
            ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',       (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#12141a'),
                                                colors.HexColor('#1a1d25')]),
            ('TEXTCOLOR',      (0,1), (-1,-1), colors.HexColor('#e2e8f0')),
            ('GRID',           (0,0), (-1,-1), 0.4, colors.HexColor('#2a2d35')),
            ('PADDING',        (0,0), (-1,-1), 7),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.6*cm))
        story.append(para('Sample Predictions', 13, '#e2e8f0', bold=True, space=8))

        table_data = [['pH','Hardness','Chloramines','Turbidity','Result','Disease','Severity']]
        for s in samples:
            table_data.append([
                f'{s.ph:.1f}'          if s.ph          else '-',
                f'{s.hardness:.0f}'    if s.hardness    else '-',
                f'{s.chloramines:.1f}' if s.chloramines else '-',
                f'{s.turbidity:.2f}'   if s.turbidity   else '-',
                s.prediction or '-',
                s.disease    or '-',
                severity_for(s.disease or 'None'),
            ])

        pt = Table(table_data,
                   colWidths=[2*cm, 2.5*cm, 2.5*cm, 2.2*cm, 2*cm, 3*cm, 2*cm])
        pt.setStyle(TableStyle([
            ('BACKGROUND',     (0,0), (-1,0),  colors.HexColor('#1a1d25')),
            ('TEXTCOLOR',      (0,0), (-1,0),  colors.HexColor('#34d399')),
            ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',       (0,0), (-1,-1), 7),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#12141a'),
                                                colors.HexColor('#1a1d25')]),
            ('TEXTCOLOR',      (0,1), (-1,-1), colors.HexColor('#e2e8f0')),
            ('GRID',           (0,0), (-1,-1), 0.3, colors.HexColor('#2a2d35')),
            ('PADDING',        (0,0), (-1,-1), 4),
        ]))
        story.append(pt)
        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True,
                         download_name=f'AquaAI_{session.filename}_{session_id}.pdf',
                         mimetype='application/pdf')
    except ImportError:
        return "Run: pip install reportlab", 500


if __name__ == "__main__":
    app.run(debug=True)