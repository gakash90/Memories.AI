# diary_services.py
from extensions import db
from models import Diary, DiaryEntry
from datetime import datetime
import uuid
import os

def get_all_diaries(user_id):
    """Get all diaries for a specific user with associated entry counts"""
    diaries = Diary.query.filter_by(user_id=user_id).order_by(Diary.updated_at.desc()).all()
    
    result = []
    for diary in diaries:
        entry_count = DiaryEntry.query.filter_by(diary_id=diary.id).count()
        result.append({
            "id": diary.id,
            "name": diary.name,
            "created_at": diary.created_at.isoformat(),
            "updated_at": diary.updated_at.isoformat(),
            "entry_count": entry_count
        })
    
    return result

def get_diary(user_id, diary_id):
    """Get a specific diary and all its entries"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return None
    
    entries = DiaryEntry.query.filter_by(diary_id=diary.id).order_by(DiaryEntry.created_at.desc()).all()
    
    entries_data = [{
        "id": entry.id,
        "title": entry.title,
        "text": entry.text,
        "caption": entry.caption,
        "image_path": entry.image_path,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat()
    } for entry in entries]
    
    return {
        "id": diary.id,
        "name": diary.name,
        "created_at": diary.created_at.isoformat(),
        "updated_at": diary.updated_at.isoformat(),
        "entries": entries_data
    }

def create_diary(user_id, name):
    """Create a new diary for a user"""
    diary = Diary(
        user_id=user_id,
        name=name,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    try:
        db.session.add(diary)
        db.session.commit()
        return {
            "id": diary.id,
            "name": diary.name,
            "created_at": diary.created_at.isoformat(),
            "updated_at": diary.updated_at.isoformat(),
            "entry_count": 0
        }, None
    except Exception as e:
        db.session.rollback()
        return None, str(e)

def update_diary(user_id, diary_id, name):
    """Update a diary's name"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return None, "Diary not found"
    
    diary.name = name
    diary.updated_at = datetime.utcnow()
    
    try:
        db.session.commit()
        return {
            "id": diary.id,
            "name": diary.name,
            "created_at": diary.created_at.isoformat(),
            "updated_at": diary.updated_at.isoformat()
        }, None
    except Exception as e:
        db.session.rollback()
        return None, str(e)

def delete_diary(user_id, diary_id):
    """Delete a diary and all its entries"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return False, "Diary not found"
    
    try:
        db.session.delete(diary)
        db.session.commit()
        return True, None
    except Exception as e:
        db.session.rollback()
        return False, str(e)

def create_entry(user_id, diary_id, title, text, caption=None, image=None):
    """Create a new entry in a diary"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return None, "Diary not found"
    
    image_path = None
    if image:
        # Handle image upload
        try:
            # Create directory for diary images if it doesn't exist
            user_diary_dir = f"static/user_data/{user_id}/diary_images/{diary_id}"
            os.makedirs(user_diary_dir, exist_ok=True)
            
            # Generate unique filename for the image
            filename = f"{uuid.uuid4()}.jpg"
            image_path = f"{user_diary_dir}/{filename}"
            
            # Save the image
            image.save(image_path)
        except Exception as e:
            return None, f"Error saving image: {str(e)}"
    
    entry = DiaryEntry(
        diary_id=diary_id,
        title=title,
        text=text,
        caption=caption,
        image_path=image_path,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    try:
        db.session.add(entry)
        # Update the diary's updated_at timestamp
        diary.updated_at = datetime.utcnow()
        db.session.commit()
        
        return {
            "id": entry.id,
            "title": entry.title,
            "text": entry.text,
            "caption": entry.caption,
            "image_path": entry.image_path,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat()
        }, None
    except Exception as e:
        db.session.rollback()
        return None, str(e)

def update_entry(user_id, diary_id, entry_id, title, text, caption=None, image=None):
    """Update a diary entry"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return None, "Diary not found"
    
    entry = DiaryEntry.query.filter_by(id=entry_id, diary_id=diary_id).first()
    if not entry:
        return None, "Entry not found"
    
    image_path = entry.image_path
    if image:
        # Handle image upload
        try:
            # Create directory for diary images if it doesn't exist
            user_diary_dir = f"static/user_data/{user_id}/diary_images/{diary_id}"
            os.makedirs(user_diary_dir, exist_ok=True)
            
            # Remove old image if it exists
            if entry.image_path and os.path.exists(entry.image_path):
                os.remove(entry.image_path)
            
            # Generate unique filename for the image
            filename = f"{uuid.uuid4()}.jpg"
            image_path = f"{user_diary_dir}/{filename}"
            
            # Save the image
            image.save(image_path)
        except Exception as e:
            return None, f"Error saving image: {str(e)}"
    
    entry.title = title
    entry.text = text
    entry.caption = caption
    entry.image_path = image_path
    entry.updated_at = datetime.utcnow()
    
    try:
        # Update the diary's updated_at timestamp
        diary.updated_at = datetime.utcnow()
        db.session.commit()
        
        return {
            "id": entry.id,
            "title": entry.title,
            "text": entry.text,
            "caption": entry.caption,
            "image_path": entry.image_path,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat()
        }, None
    except Exception as e:
        db.session.rollback()
        return None, str(e)

def delete_entry(user_id, diary_id, entry_id):
    """Delete a diary entry"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return False, "Diary not found"
    
    entry = DiaryEntry.query.filter_by(id=entry_id, diary_id=diary_id).first()
    if not entry:
        return False, "Entry not found"
    
    # Delete image file if it exists
    if entry.image_path and os.path.exists(entry.image_path):
        try:
            os.remove(entry.image_path)
        except Exception as e:
            print(f"Warning: Could not delete image file: {str(e)}")
    
    try:
        db.session.delete(entry)
        # Update the diary's updated_at timestamp
        diary.updated_at = datetime.utcnow()
        db.session.commit()
        return True, None
    except Exception as e:
        db.session.rollback()
        return False, str(e)

def get_entry(user_id, diary_id, entry_id):
    """Get a specific diary entry"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return None, "Diary not found"
    
    entry = DiaryEntry.query.filter_by(id=entry_id, diary_id=diary_id).first()
    if not entry:
        return None, "Entry not found"
    
    return {
        "id": entry.id,
        "diary_id": entry.diary_id,
        "title": entry.title,
        "text": entry.text,
        "caption": entry.caption,
        "image_path": entry.image_path,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat()
    }, None


def get_diary_with_entries(user_id, diary_id):
    """Get a specific diary and all its entries"""
    diary = Diary.query.filter_by(id=diary_id, user_id=user_id).first()
    if not diary:
        return None
    
    entries = DiaryEntry.query.filter_by(diary_id=diary.id).order_by(DiaryEntry.created_at.desc()).all()
    
    entries_data = [{
        "id": entry.id,
        "title": entry.title,
        "text": entry.text,
        "caption": entry.caption,
        "image_path": entry.image_path,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat()
    } for entry in entries]
    
    return {
        "id": diary.id,
        "name": diary.name,
        "created_at": diary.created_at.isoformat(),
        "updated_at": diary.updated_at.isoformat(),
        "entries": entries_data
    }