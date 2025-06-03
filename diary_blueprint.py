# diary_blueprint.py
from flask import Blueprint, request, jsonify, send_from_directory
from flask_login import login_required, current_user
from diary_services import (
    get_all_diaries, get_diary, create_diary, update_diary, delete_diary,
    create_entry, update_entry, delete_entry, get_entry
)

diary_bp = Blueprint('diary', __name__, url_prefix='/api/diaries')

@diary_bp.route('', methods=['GET'])
@login_required
def get_diaries():
    """Get all diaries for the current user"""
    diaries = get_all_diaries(current_user.id)
    return jsonify({
        "success": True,
        "diaries": diaries
    })

@diary_bp.route('', methods=['POST'])
@login_required
def add_diary():
    """Create a new diary"""
    data = request.json
    name = data.get('name', '')
    
    if not name:
        return jsonify({"success": False, "error": "Diary name is required"}), 400
    
    diary, error = create_diary(current_user.id, name)
    if error:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "diary": diary
    })

@diary_bp.route('/<int:diary_id>', methods=['GET'])
@login_required
def get_diary_details(diary_id):
    """Get a specific diary and its entries"""
    diary_data = get_diary(current_user.id, diary_id)
    if not diary_data:
        return jsonify({"success": False, "error": "Diary not found"}), 404
    
    return jsonify({
        "success": True,
        "diary": diary_data
    })

@diary_bp.route('/<int:diary_id>', methods=['PUT'])
@login_required
def update_diary_name(diary_id):
    """Update a diary's name"""
    data = request.json
    name = data.get('name', '')
    
    if not name:
        return jsonify({"success": False, "error": "Diary name is required"}), 400
    
    diary, error = update_diary(current_user.id, diary_id, name)
    if error:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "diary": diary
    })

@diary_bp.route('/<int:diary_id>', methods=['DELETE'])
@login_required
def delete_diary_route(diary_id):
    """Delete a diary"""
    success, error = delete_diary(current_user.id, diary_id)
    if not success:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "message": "Diary deleted successfully"
    })

@diary_bp.route('/<int:diary_id>/entries', methods=['POST'])
@login_required
def add_entry(diary_id):
    """Add a new entry to a diary"""
    if request.is_json:
        data = request.json
        title = data.get('title', '')
        text = data.get('text', '')
        caption = data.get('caption', '')
        
        if not title or not text:
            return jsonify({"success": False, "error": "Title and text are required"}), 400
        
        entry, error = create_entry(current_user.id, diary_id, title, text, caption)
    else:
        # Handle form data with possible file upload
        title = request.form.get('title', '')
        text = request.form.get('text', '')
        caption = request.form.get('caption', '')
        image = request.files.get('image') if 'image' in request.files else None
        
        if not title or not text:
            return jsonify({"success": False, "error": "Title and text are required"}), 400
        
        entry, error = create_entry(current_user.id, diary_id, title, text, caption, image)
    
    if error:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "entry": entry
    })

@diary_bp.route('/<int:diary_id>/entries/<int:entry_id>', methods=['GET'])
@login_required
def get_entry_details(diary_id, entry_id):
    """Get a specific diary entry"""
    entry, error = get_entry(current_user.id, diary_id, entry_id)
    if error:
        return jsonify({"success": False, "error": error}), 404
    
    return jsonify({
        "success": True,
        "entry": entry
    })

@diary_bp.route('/<int:diary_id>/entries/<int:entry_id>', methods=['PUT'])
@login_required
def update_entry_route(diary_id, entry_id):
    """Update a diary entry"""
    if request.is_json:
        data = request.json
        title = data.get('title', '')
        text = data.get('text', '')
        caption = data.get('caption', '')
        
        if not title or not text:
            return jsonify({"success": False, "error": "Title and text are required"}), 400
        
        entry, error = update_entry(current_user.id, diary_id, entry_id, title, text, caption)
    else:
        # Handle form data with possible file upload
        title = request.form.get('title', '')
        text = request.form.get('text', '')
        caption = request.form.get('caption', '')
        image = request.files.get('image') if 'image' in request.files else None
        
        if not title or not text:
            return jsonify({"success": False, "error": "Title and text are required"}), 400
        
        entry, error = update_entry(current_user.id, diary_id, entry_id, title, text, caption, image)
    
    if error:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "entry": entry
    })

@diary_bp.route('/<int:diary_id>/entries/<int:entry_id>', methods=['DELETE'])
@login_required
def delete_entry_route(diary_id, entry_id):
    """Delete a diary entry"""
    success, error = delete_entry(current_user.id, diary_id, entry_id)
    if not success:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "message": "Entry deleted successfully"
    })