from flask import Blueprint, request, jsonify, send_from_directory, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from models import User
from extensions import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            return jsonify({"success": False, "error": "Username already exists"}), 400
        
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return jsonify({"success": True, "message": "Registration successful", "user_id": new_user.id})
    
    return send_from_directory('static', 'register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return jsonify({"success": False, "error": "Invalid username or password"}), 401
        
        login_user(user)
        return jsonify({"success": True, "message": "Login successful", "user_id": user.id})
    
    # For GET requests, serve the login page
    return send_from_directory('static', 'login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@auth_bp.route('/api/user/current')
@login_required
def current_user_info():
    return jsonify({
        "success": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username
        }
    })