# db_schema.py
from extensions import db
from datetime import datetime


class ModelStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), unique=True)
    is_downloaded = db.Column(db.Boolean, default=False)
    version = db.Column(db.String(20))
    path = db.Column(db.String(255))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

def get_model_status(model_name):
    """Get the current status of a model from the database"""
    model = ModelStatus.query.filter_by(model_name=model_name).first()
    if model:
        return {
            'is_downloaded': model.is_downloaded,
            'version': model.version,
            'path': model.path,
            'last_updated': model.last_updated
        }
    return {
        'is_downloaded': False,
        'version': None,
        'path': None,
        'last_updated': None
    }

def set_model_status(model_name, is_downloaded, version, path):
    """Update or create a model status in the database"""
    model = ModelStatus.query.filter_by(model_name=model_name).first()
    if model:
        model.is_downloaded = is_downloaded
        model.version = version
        model.path = path
        model.last_updated = datetime.utcnow()
    else:
        model = ModelStatus(
            model_name=model_name,
            is_downloaded=is_downloaded,
            version=version,
            path=path
        )
        db.session.add(model)
    
    db.session.commit()