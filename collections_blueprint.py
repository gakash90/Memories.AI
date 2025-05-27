from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from services import (
    get_all_collections, 
    get_collection, 
    create_collection, 
    delete_collection,
    get_collection_metadata_path
)
from chat_services import (
    create_diary_chat_session,
    get_diary_chat_sessions,
    process_diary_chat_query,
    get_chat_messages
)
from models import Chat
import json

collections_bp = Blueprint('collections', __name__, url_prefix='/api/collections')

diary_chat_bp = Blueprint('diary_chat', __name__, url_prefix='/api/diaries')

@collections_bp.route('', methods=['GET'])
@login_required
def get_collections():
    collections = get_all_collections(current_user.id)
    return jsonify({
        "success": True,
        "collections": collections
    })

@diary_chat_bp.route('/<int:diary_id>/chat', methods=['POST'])
@login_required
def create_diary_chat(diary_id):
    chat, error = create_diary_chat_session(current_user.id, diary_id)
    if error:
        return jsonify({"success": False, "error": error}), 404
    
    return jsonify({
        "success": True,
        "chat": {
            "id": chat.id,
            "title": chat.title,
            "diary_id": chat.diary_id,
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat()
        }
    })

@diary_chat_bp.route('/<int:diary_id>/chats', methods=['GET'])
@login_required
def get_diary_chats(diary_id):
    chats = get_diary_chat_sessions(current_user.id, diary_id)
    return jsonify({
        "success": True,
        "chats": [{
            "id": chat.id,
            "title": chat.title,
            "diary_id": chat.diary_id,
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat()
        } for chat in chats]
    })

@diary_chat_bp.route('/<int:diary_id>/chat/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_diary_chat_messages(diary_id, chat_id):
    chat, messages = get_chat_messages(chat_id, current_user.id)
    if not chat:
        return jsonify({"success": False, "error": "Chat session not found"}), 404
    
    if chat.diary_id != diary_id:
        return jsonify({"success": False, "error": "Chat does not belong to this diary"}), 404
    
    return jsonify({
        "success": True,
        "chat": {
            "id": chat.id,
            "title": chat.title,
            "diary_id": chat.diary_id
        },
        "messages": [{
            "id": msg.id,
            "content": msg.content,
            "is_user": msg.is_user,
            "timestamp": msg.timestamp.isoformat(),
            "relevant_entry_ids": msg.relevant_memory_ids.split(",") if msg.relevant_memory_ids else []
        } for msg in messages]
    })



@diary_chat_bp.route('/<int:diary_id>/chat/<int:chat_id>/query', methods=['POST'])
@login_required
def process_diary_query(diary_id, chat_id):
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"success": False, "error": "Query is required"}), 400
    
    # Verify this chat belongs to this diary
    chat = Chat.query.filter_by(
        id=chat_id,
        user_id=current_user.id,
        diary_id=diary_id
    ).first()
    
    if not chat:
        return jsonify({"success": False, "error": "Chat not found for this diary"}), 404
    
    result, error = process_diary_chat_query(chat_id, current_user.id, query)
    if error:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "query": result["query"],
        "response": result["response"],
        "relevant_entries": result["relevant_entries"]
    })

@collections_bp.route('', methods=['POST'])
@login_required
def create_new_collection():
    data = request.json
    name = data.get('name', '')
    description = data.get('description', '')
    
    if not name:
        return jsonify({"success": False, "error": "Collection name is required"}), 400
    
    collection_id, metadata = create_collection(current_user.id, name, description)
    
    return jsonify({
        "success": True,
        "collection": metadata
    })

@collections_bp.route('/<collection_id>', methods=['GET'])
@login_required
def get_collection_details(collection_id):
    collection = get_collection(current_user.id, collection_id)
    if not collection:
        return jsonify({"success": False, "error": "Collection not found"}), 404
    
    return jsonify({
        "success": True,
        "collection": collection
    })

@collections_bp.route('/<collection_id>', methods=['DELETE'])
@login_required
def delete_collection_route(collection_id):
    success = delete_collection(current_user.id, collection_id)
    if not success:
        return jsonify({"success": False, "error": "Collection not found"}), 404
    
    return jsonify({
        "success": True,
        "message": "Collection deleted successfully"
    })


@collections_bp.route('/<collection_id>', methods=['PUT'])
@login_required
def update_collection(collection_id):
    data = request.json
    name = data.get('name')
    description = data.get('description')
    
    # Validate inputs
    if not name:
        return jsonify({"success": False, "error": "Collection name is required"}), 400
    
    # Get the collection
    collection = get_collection(current_user.id, collection_id)
    if not collection:
        return jsonify({"success": False, "error": "Collection not found"}), 404
    
    # Update the collection metadata file
    collection["name"] = name
    collection["description"] = description
    
    # Save updated metadata
    metadata_path = get_collection_metadata_path(current_user.id, collection_id)
    try:
        with open(metadata_path, 'w') as f:
            json.dump(collection, f)
            
        return jsonify({
            "success": True,
            "collection": collection
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

