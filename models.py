from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class UploadSession(db.Model):
    __tablename__ = 'upload_session'
    id           = db.Column(db.Integer, primary_key=True)
    filename     = db.Column(db.String(255))
    safe_count   = db.Column(db.Integer, default=0)
    unsafe_count = db.Column(db.Integer, default=0)
    uploaded_at  = db.Column(db.DateTime, default=datetime.utcnow)
    samples      = db.relationship('Sample', backref='session',
                                   cascade='all, delete-orphan', lazy=True)

    @property
    def total(self):
        return self.safe_count + self.unsafe_count

    @property
    def safe_pct(self):
        return round(self.safe_count / max(self.total, 1) * 100, 1)


class Sample(db.Model):
    __tablename__ = 'sample'
    id                 = db.Column(db.Integer, primary_key=True)
    session_id         = db.Column(db.Integer, db.ForeignKey('upload_session.id'), nullable=False)
    ph                 = db.Column(db.Float)
    hardness           = db.Column(db.Float)
    solids             = db.Column(db.Float)
    chloramines        = db.Column(db.Float)
    sulfate            = db.Column(db.Float)
    conductivity       = db.Column(db.Float)
    organic_carbon     = db.Column(db.Float)
    trihalomethanes    = db.Column(db.Float)
    turbidity          = db.Column(db.Float)
    prediction         = db.Column(db.String(10))
    confidence         = db.Column(db.Float)
    disease            = db.Column(db.String(50))
    disease_confidence = db.Column(db.Float)