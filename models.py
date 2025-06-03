from extensions import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


def ist_now():
    return datetime.now(IST).replace(tzinfo=None)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    chats = db.relationship('Chat', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    collection_id = db.Column(db.String(36), nullable=True)  # Make nullable for diary chats
    memory_id = db.Column(db.String(36), nullable=True)
    diary_id = db.Column(db.Integer, db.ForeignKey('diary.id'), nullable=True)  # Add this line
    title = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=ist_now
)
    updated_at = db.Column(db.DateTime, nullable=False, default=ist_now
)
    messages = db.relationship('ChatMessage', backref='chat', lazy=True, cascade="all, delete-orphan")
    
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, nullable=False, default=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=ist_now
)
    relevant_memory_ids = db.Column(db.String(200))  # Comma-separated memory IDs

class Diary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=ist_now
)
    updated_at = db.Column(db.DateTime, nullable=False, default=ist_now
)
    entries = db.relationship('DiaryEntry', backref='diary', lazy=True, cascade="all, delete-orphan")

class DiaryEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    diary_id = db.Column(db.Integer, db.ForeignKey('diary.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    text = db.Column(db.Text, nullable=False)
    caption = db.Column(db.String(500))
    image_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, nullable=False, default=ist_now
)
    updated_at = db.Column(db.DateTime, nullable=False, default=ist_now
)


class ConversationContextData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    graph_data = db.Column(db.Text, nullable=False)  # Serialized graph data
    updated_at = db.Column(db.DateTime, nullable=False, default=ist_now
)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))