import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, redirect, url_for, send_from_directory, jsonify
from flask_login import login_required, current_user
from services import get_collection
from diary_blueprint import diary_bp
from models import Chat

from extensions import db, login_manager
from auth_blueprint import auth_bp
from collections_blueprint import collections_bp
from memory_blueprint import memory_bp
from chat_blueprint import chat_bp, general_chat_bp, diary_chat_bp
from model_download import model_download_bp
from model_downloader import model_downloader_bp
from upload_blueprint import upload_bp

def create_app():
    app = Flask(__name__, static_folder='static')
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///memory_vault.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(diary_chat_bp)

    app.register_blueprint(upload_bp)
    app.register_blueprint(collections_bp)
    app.register_blueprint(memory_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(general_chat_bp)  # Register the new general chat blueprint

    app.register_blueprint(model_download_bp)
    app.register_blueprint(model_downloader_bp)
    app.register_blueprint(diary_bp)

    with app.app_context():
        db.create_all()
    
    @app.route('/')
    def index():
        from flask_login import current_user
        if current_user.is_authenticated:
            return send_from_directory('static', 'index.html')
        return redirect(url_for('auth.login'))

    @app.route('/collections')
    def collections_index():
        return send_from_directory('static', 'collections-index.html')

    @app.route('/collections/<collection_id>')
    @login_required
    def collection_view(collection_id):
        """Serve the collection view page for a specific collection"""
        # Check if collection exists and belongs to the user
        collection = get_collection(current_user.id, collection_id)
        if not collection:
            flash('Collection not found', 'error')
            return redirect(url_for('collections_index'))
        
        return send_from_directory('static', 'collection-view.html')

    @app.route('/collections/<collection_id>/upload')
    @login_required
    def upload_memory_view(collection_id):
        collection = get_collection(current_user.id, collection_id)
        if not collection:
            return redirect(url_for('index'))
        
        return send_from_directory('static', 'upload-memory.html')

    @app.route('/collections/<collection_id>/chat')
    @app.route('/collections/<collection_id>/chat/<chat_id>')
    @login_required
    def collection_chat(collection_id, chat_id=None):
        collection = get_collection(current_user.id, collection_id)
        if not collection:
            return redirect(url_for('collections_index'))
        
        return send_from_directory('static', 'chat-view.html')

    @app.route('/collections/<collection_id>/memory/<memory_id>/chat')
    @app.route('/collections/<collection_id>/memory/<memory_id>/chat/<chat_id>')
    @login_required
    def memory_chat(collection_id, memory_id, chat_id=None):
        collection = get_collection(current_user.id, collection_id)
        if not collection:
            return redirect(url_for('collections_index'))
        
        # Additional check for memory existence could be added here
        
        return send_from_directory('static', 'chat-view.html')
    

    @app.route('/diary')
    @login_required
    def diary_index():
        """Serve the diary index page"""
        return send_from_directory('static', 'diary.html')

    @app.route('/diary/<int:diary_id>')
    @login_required
    def diary_view(diary_id):
        """Serve the diary view page for a specific diary"""
        from flask_login import current_user
        from diary_services import get_diary
        
        # Check if diary exists and belongs to the user
        diary_data = get_diary(current_user.id, diary_id)
        if not diary_data:
            flash('Diary not found', 'error')
            return redirect(url_for('diary_index'))
        
        return send_from_directory('static', 'diary-view.html')

    @app.route('/diary/<int:diary_id>/chat')
    @app.route('/diary/<int:diary_id>/chat/<int:chat_id>')
    @login_required
    def diary_chat(diary_id, chat_id=None):
        """Serve the chat view page for diary chats"""
        # Check if diary exists and belongs to the user
        from diary_services import get_diary
        
        diary_data = get_diary(current_user.id, diary_id)
        if not diary_data:
            flash('Diary not found', 'error')
            return redirect(url_for('diary_index'))
        
        # Use the same chat-view.html for diary chats
        return send_from_directory('static', 'chat-view.html')


    @app.route('/diary/<int:diary_id>/chat')
    @app.route('/diary/<int:diary_id>/chat/<int:chat_id>')
    @login_required
    def diary_chat_view(diary_id, chat_id=None):
        """Serve the chat view page for diary chats"""
        # Check if diary exists and belongs to the user
        from diary_services import get_diary
        
        diary_data = get_diary(current_user.id, diary_id)
        if not diary_data:
            flash('Diary not found', 'error')
            return redirect(url_for('diary_index'))
        
        # Use the same chat-view.html for diary chats
        return send_from_directory('static', 'chat-view.html')


    @app.route('/chats/<chat_id>')
    @login_required
    def view_general_chat(chat_id):
        """Serve the chat view page for any type of chat"""
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        
        if not chat:
            flash('Chat not found', 'error')
            return redirect(url_for('chat_index'))
        
        # Determine the appropriate redirect based on chat type
        if chat.collection_id and chat.memory_id:
            # Memory chat
            return redirect(url_for('memory_chat', 
                                collection_id=chat.collection_id, 
                                memory_id=chat.memory_id, 
                                chat_id=chat.id))
        elif chat.collection_id:
            # Collection chat
            return redirect(url_for('collection_chat', 
                                collection_id=chat.collection_id, 
                                chat_id=chat.id))
        elif chat.diary_id:
            # Diary chat
            return redirect(url_for('diary_chat', 
                                diary_id=chat.diary_id, 
                                chat_id=chat.id))
        else:
            flash('Invalid chat type', 'error')
            return redirect(url_for('chat_index'))



# Add this route to your app.py in the create_app() function or at the appropriate location

    @app.route('/chats')
    @login_required
    def chat_index():
        return send_from_directory('static', 'chat-index.html')

    @app.route('/static/<path:path>')
    def serve_static(path):
        return send_from_directory('static', path)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)


