from flask import Blueprint, request, jsonify
import json
from flask_login import login_required, current_user
from services import get_collection, process_memory, allowed_file, detect_file_type, delete_memory, get_collection_metadata_path

memory_bp = Blueprint('memory', __name__, url_prefix='/api/collections')

@memory_bp.route('/<collection_id>/memories', methods=['POST'])
@login_required
def add_memory(collection_id):
    collection = get_collection(current_user.id, collection_id)
    if not collection:
        return jsonify({"success": False, "error": "Collection not found"}), 404
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']

    memory_type = request.form.get('type')
    title = request.form.get('title', 'Untitled Memory')
    description = request.form.get('description', '')
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if not memory_type:
        memory_type = detect_file_type(file)
        if not memory_type:
            return jsonify({
                "success": False, 
                "error": "Could not determine file type. Please specify type parameter or use a supported file extension."
            }), 400
    
    if memory_type not in ['audio', 'pdf', 'text']:
        return jsonify({"success": False, "error": "Invalid memory type"}), 400
    
    if not allowed_file(file.filename, memory_type):
        return jsonify({
            "success": False, 
            "error": f"Invalid file type. Allowed types for {memory_type}: {', '.join(ALLOWED_EXTENSIONS[memory_type])}"
        }), 400
    
    memory, error = process_memory(current_user.id, collection_id, file, memory_type, title, description)
    
    if error:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "memory": memory,
        "detected_type": memory_type if not request.form.get('type') else None
    })

@memory_bp.route('/<collection_id>/memories/<memory_id>', methods=['GET'])
@login_required
def get_memory(collection_id, memory_id):
    collection = get_collection(current_user.id, collection_id)
    if not collection:
        return jsonify({"success": False, "error": "Collection not found"}), 404
    
    # Find the specific memory in the collection
    memory = None
    for mem in collection.get("memories", []):
        if mem["id"] == memory_id:
            memory = mem
            break
    
    if not memory:
        return jsonify({"success": False, "error": "Memory not found"}), 404
    
    return jsonify({
        "success": True,
        "memory": memory
    })

@memory_bp.route('/<collection_id>/memories/<memory_id>', methods=['DELETE'])
@login_required
def delete_memory_route(collection_id, memory_id):
    collection = get_collection(current_user.id, collection_id)
    if not collection:
        return jsonify({"success": False, "error": "Collection not found"}), 404
    
    # Add a delete_memory function to your services.py file
    success, error = delete_memory(current_user.id, collection_id, memory_id)
    
    if not success:
        return jsonify({"success": False, "error": error}), 500
    
    return jsonify({
        "success": True,
        "message": "Memory deleted successfully"
    })


@memory_bp.route('/<collection_id>/memories/<memory_id>', methods=['PUT'])
@login_required
def update_memory(collection_id, memory_id):
    data = request.json
    title = data.get('title')
    description = data.get('description')
    
    # Validate inputs
    if not title:
        return jsonify({"success": False, "error": "Memory title is required"}), 400
    
    # Get the collection
    collection = get_collection(current_user.id, collection_id)
    if not collection:
        return jsonify({"success": False, "error": "Collection not found"}), 404
    
    # Find the memory in the collection
    memory_index = None
    for i, mem in enumerate(collection.get("memories", [])):
        if mem["id"] == memory_id:
            memory_index = i
            break
    
    if memory_index is None:
        return jsonify({"success": False, "error": "Memory not found"}), 404
    
    # Update the memory metadata
    collection["memories"][memory_index]["title"] = title
    collection["memories"][memory_index]["description"] = description
    
    # Save updated collection metadata
    metadata_path = get_collection_metadata_path(current_user.id, collection_id)
    try:
        with open(metadata_path, 'w') as f:
            json.dump(collection, f)
            
        return jsonify({
            "success": True,
            "memory": collection["memories"][memory_index]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
