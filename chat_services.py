from models import Chat, ChatMessage
from extensions import db
from services import (
    query_collection, get_collection, generate_response, 
    get_collection_documents_path, query_specific_memory,
    query_across_collections, global_search, search_suggestions
)
from diary_services import get_diary, get_diary_with_entries
from datetime import datetime
import ollama
import numpy as np
import os
from knowledge_graph import ConversationContext
import json
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

conversation_context = ConversationContext()

# Enhanced chat session management (keeping original function names)
def create_memory_chat_session(user_id, collection_id, memory_id):
    """Enhanced memory chat session creation with validation"""
    collection = get_collection(user_id, collection_id)
    if not collection:
        return None, "Collection not found"
    
    # Find the memory in the collection
    memory = None
    for mem in collection.get("memories", []):
        if mem["id"] == memory_id:
            memory = mem
            break
    
    if not memory:
        return None, "Memory not found"
    
    # Create enhanced chat title
    memory_type = memory.get('type', 'document').title()
    original_filename = memory.get('original_filename', 'Unknown')
    title = f"Chat: {memory['title']} ({memory_type} - {original_filename})"
    
    chat = Chat(
        user_id=user_id,
        collection_id=collection_id,
        memory_id=memory_id,
        title=title,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    try:
        db.session.add(chat)
        db.session.commit()
        logger.info(f"Created memory chat session {chat.id} for memory {memory_id}")
        return chat, None
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating memory chat session: {e}")
        return None, str(e)

def create_chat_session(user_id, collection_id):
    """Enhanced collection chat session creation"""
    collection = get_collection(user_id, collection_id)
    if not collection:
        return None, "Collection not found"
    
    # Create descriptive title with statistics
    stats = collection.get('stats', {})
    memory_count = stats.get('total_memories', len(collection.get('memories', [])))
    title = f"Chat: {collection['name']} ({memory_count} documents)"
    
    chat = Chat(
        user_id=user_id,
        collection_id=collection_id,
        title=title,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    try:
        db.session.add(chat)
        db.session.commit()
        logger.info(f"Created collection chat session {chat.id} for collection {collection_id}")
        return chat, None
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating collection chat session: {e}")
        return None, str(e)

def get_chat_sessions(user_id, collection_id=None, memory_id=None):
    """Enhanced chat session retrieval with better filtering"""
    query = Chat.query.filter_by(user_id=user_id)
    
    if collection_id:
        query = query.filter_by(collection_id=collection_id)
    
    # Enhanced filtering logic
    if memory_id is None and collection_id is not None:
        # Collection-level chats only
        query = query.filter(Chat.memory_id.is_(None))
    elif memory_id:
        # Specific memory chats only
        query = query.filter_by(memory_id=memory_id)
    
    # Order by most recent activity
    chat_sessions = query.order_by(Chat.updated_at.desc()).all()
    
    # Add session metadata
    for chat in chat_sessions:
        # Count messages in this chat
        message_count = ChatMessage.query.filter_by(chat_id=chat.id).count()
        chat.message_count = message_count
        
        # Get last message timestamp
        last_message = ChatMessage.query.filter_by(chat_id=chat.id).order_by(ChatMessage.timestamp.desc()).first()
        chat.last_activity = last_message.timestamp if last_message else chat.created_at
    
    return chat_sessions

def get_chat_messages(chat_id, user_id):
    """Enhanced chat message retrieval with pagination support"""
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return None, "Chat session not found"
    
    # Get messages with enhanced metadata
    messages = ChatMessage.query.filter_by(chat_id=chat.id).order_by(ChatMessage.timestamp).all()
    
    # Add metadata to messages
    for message in messages:
        # Parse relevant memory IDs if present
        if message.relevant_memory_ids:
            try:
                if isinstance(message.relevant_memory_ids, str):
                    memory_ids = [id.strip() for id in message.relevant_memory_ids.split(',') if id.strip()]
                else:
                    memory_ids = [str(message.relevant_memory_ids)]
                message.parsed_memory_ids = memory_ids
            except:
                message.parsed_memory_ids = []
        else:
            message.parsed_memory_ids = []
    
    return chat, messages

def add_message_to_chat(chat_id, user_id, content, is_user=True, relevant_memory_ids=None):
    """Enhanced message addition with better metadata handling"""
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return None, "Chat session not found"
    
    # Enhanced memory ID handling
    memory_ids_str = None
    if relevant_memory_ids:
        if isinstance(relevant_memory_ids, list):
            memory_ids_str = ",".join(str(id) for id in relevant_memory_ids if id)
        else:
            memory_ids_str = str(relevant_memory_ids)
    
    # Create message with enhanced metadata
    message = ChatMessage(
        chat_id=chat.id,
        content=content,
        is_user=is_user,
        timestamp=datetime.utcnow(),
        relevant_memory_ids=memory_ids_str
    )
    
    # Update chat timestamp
    chat.updated_at = datetime.utcnow()
    
    try:
        db.session.add(message)
        db.session.commit()
        
        logger.info(f"Added {'user' if is_user else 'AI'} message to chat {chat.id}")
        return message, None
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding message to chat: {e}")
        return None, str(e)

def process_chat_query(chat_id, user_id, query_text, search_type="hybrid"):
    """Enhanced chat query processing with multiple search strategies"""
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return None, "Chat session not found"
    
    try:
        # Initialize conversation context for this chat
        conversation_context.start_session(chat_id)
        
        # Analyze the query in context of conversation history
        context_analysis = conversation_context.analyze_question_context(query_text)
        
        # Use the expanded question for search if it's a follow-up
        search_query = context_analysis.get('expanded_question', query_text)
        
        # Add the original user message to the database
        user_message, error = add_message_to_chat(chat_id, user_id, query_text, is_user=True)
        if error:
            return None, error
        
        # Add to conversation graph
        conversation_context.add_message(
            user_message.id, 
            query_text, 
            True, 
            user_message.timestamp.isoformat()
        )
        
        # Enhanced search with multiple strategies
        relevant_memories = []
        search_metadata = {}
        
        # Primary search in the collection
        if chat.collection_id:
            memories, error = query_collection(
                user_id, 
                chat.collection_id, 
                search_query, 
                top_k=20,  # Increased for better coverage
                search_type=search_type
            )
            
            if not error and memories:
                relevant_memories.extend(memories)
                search_metadata["primary_search"] = {
                    "collection_id": chat.collection_id,
                    "results_count": len(memories),
                    "search_type": search_type
                }
        
        # If not enough results, try cross-collection search
        if len(relevant_memories) < 5:
            cross_memories, error = query_across_collections(
                user_id, 
                search_query, 
                collection_ids=None,  # Search all collections
                top_k=15,
                search_type=search_type
            )
            
            if not error and cross_memories:
                # Add non-duplicate results
                existing_memory_ids = {m["metadata"]["id"] for m in relevant_memories}
                new_memories = [m for m in cross_memories 
                              if m["metadata"]["id"] not in existing_memory_ids]
                relevant_memories.extend(new_memories[:10])
                
                search_metadata["cross_collection_search"] = {
                    "results_count": len(new_memories),
                    "total_collections_searched": len(set(m.get("source_collection_id") for m in cross_memories))
                }
        
        # If still not enough results, try global search
        if len(relevant_memories) < 3:
            global_results, error = global_search(
                user_id,
                search_query,
                search_type=search_type,
                limit=10
            )
            
            if not error and global_results:
                # Convert global results to memory format
                for result in global_results:
                    if result["document_id"] not in {m["metadata"]["id"] for m in relevant_memories}:
                        relevant_memories.append({
                            "metadata": {
                                "id": result["document_id"],
                                "title": result["title"],
                                "type": result["memory_type"]
                            },
                            "content": result["content"],
                            "score": result["score"],
                            "search_type": "global"
                        })
                
                search_metadata["global_search"] = {
                    "results_count": len(global_results)
                }
        
        # Generate response with conversation history
        history = conversation_context.get_conversation_history(limit=5)
        history_text = "\n".join([
            f"{'User' if msg['is_user'] else 'Assistant'}: {msg['content']}"
            for msg in history[:-1]  # Exclude the current question
        ])
        
        # Enhanced response generation
        response_text = generate_response(
            search_query, 
            relevant_memories[:10],  # Limit to top 10 for context
            history_text if history_text.strip() else None
        )
        
        # Add search insights to response if helpful
        if search_metadata and len(relevant_memories) > 5:
            insights = []
            if "cross_collection_search" in search_metadata:
                insights.append(f"Found additional relevant information across {search_metadata['cross_collection_search']['total_collections_searched']} collections")
            if "global_search" in search_metadata:
                insights.append("Expanded search to your entire knowledge base")
            
            if insights:
                response_text += f"\n\n*Search insights: {'; '.join(insights)}*"
        
        memory_ids = [memory['metadata']['id'] for memory in relevant_memories[:5]]
        
        # Add AI response to database
        ai_message, error = add_message_to_chat(
            chat_id, 
            user_id, 
            response_text, 
            is_user=False,
            relevant_memory_ids=memory_ids
        )
        
        if error:
            return None, error
        
        # Add to conversation graph with enhanced metadata
        conversation_context.add_message(
            ai_message.id, 
            response_text, 
            False, 
            ai_message.timestamp.isoformat(),
            related_entities=[(mem_id, 'memory') for mem_id in memory_ids]
        )
        
        return {
            "query": query_text,
            "expanded_query": search_query if search_query != query_text else None,
            "response": response_text,
            "relevant_memories": [m["metadata"] for m in relevant_memories[:10]],
            "search_metadata": search_metadata,
            "total_results_found": len(relevant_memories)
        }, None
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        return None, str(e)

def process_memory_chat_query(chat_id, user_id, query_text):
    """Enhanced memory-specific chat query processing"""
    logger.info(f"Processing memory chat query for chat {chat_id}")
    
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return None, "Chat session not found"
    
    if not chat.memory_id:
        return None, "This is not a memory-specific chat"
    
    try:
        # Add user message
        user_message, error = add_message_to_chat(chat_id, user_id, query_text, is_user=True)
        if error:
            return None, error
        
        # Enhanced memory querying with chunk-level search
        memory_result, error = query_specific_memory(
            user_id, 
            chat.collection_id, 
            chat.memory_id, 
            query_text
        )
        
        if error:
            return None, error
        
        # Get enhanced conversation history
        previous_messages = ChatMessage.query.filter_by(chat_id=chat_id)\
            .order_by(ChatMessage.timestamp.desc()).limit(10).all()
        
        history_text = "\n".join([
            f"{'User' if msg.is_user else 'Assistant'}: {msg.content}"
            for msg in reversed(previous_messages[1:])  # Exclude current, reverse to chronological
        ])
        
        # Enhanced response generation for specific memory
        if memory_result:
            memory_data = memory_result[0]
            
            # Add memory context to the prompt
            memory_context = f"""
This is a focused conversation about a specific document:
- Title: {memory_data['metadata']['title']}
- Type: {memory_data['metadata']['type']}
- File: {memory_data['metadata'].get('original_filename', 'Unknown')}
- Chunks used: {memory_data.get('chunks_used', 1)}
"""
            
            # Generate contextual response
            enhanced_history = f"{memory_context}\n{history_text}" if history_text else memory_context
            response_text = generate_response(query_text, memory_result, enhanced_history)
            
            # Add memory-specific insights
            if memory_data.get('chunks_used', 0) > 1:
                response_text += f"\n\n*Note: Response based on {memory_data['chunks_used']} most relevant sections of this document.*"
        else:
            response_text = "I couldn't find relevant information in this specific document to answer your question."
        
        # Add AI response
        ai_message, error = add_message_to_chat(
            chat_id, 
            user_id, 
            response_text, 
            is_user=False,
            relevant_memory_ids=chat.memory_id
        )
        
        if error:
            return None, error
        
        return {
            "query": query_text,
            "response": response_text,
            "relevant_memories": [memory_result[0]["metadata"]] if memory_result else [],
            "memory_specific": True,
            "memory_id": chat.memory_id
        }, None
        
    except Exception as e:
        logger.error(f"Error processing memory chat query: {e}")
        return None, str(e)

def create_diary_chat_session(user_id, diary_id):
    """Enhanced diary chat session creation"""
    diary = get_diary(user_id, diary_id)
    if not diary:
        return None, "Diary not found"
    
    # Create descriptive title
    title = f"Diary Chat: {diary['name']}"
    
    chat = Chat(
        user_id=user_id,
        diary_id=diary_id,
        title=title,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    try:
        db.session.add(chat)
        db.session.commit()
        logger.info(f"Created diary chat session {chat.id} for diary {diary_id}")
        return chat, None
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating diary chat session: {e}")
        return None, str(e)

def get_diary_chat_sessions(user_id, diary_id):
    """Enhanced diary chat session retrieval"""
    # Validate diary exists
    diary = get_diary(user_id, diary_id)
    if not diary:
        return []
    
    chat_sessions = Chat.query.filter_by(
        user_id=user_id,
        diary_id=diary_id
    ).order_by(Chat.updated_at.desc()).all()
    
    # Add session metadata
    for chat in chat_sessions:
        message_count = ChatMessage.query.filter_by(chat_id=chat.id).count()
        chat.message_count = message_count
        
        last_message = ChatMessage.query.filter_by(chat_id=chat.id)\
            .order_by(ChatMessage.timestamp.desc()).first()
        chat.last_activity = last_message.timestamp if last_message else chat.created_at
    
    return chat_sessions

def process_diary_chat_query(chat_id, user_id, query_text):
    """Enhanced diary chat query processing"""
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return None, "Chat session not found"
    
    if not chat.diary_id:
        return None, "This is not a diary chat"
    
    try:
        # Add user message
        user_message, error = add_message_to_chat(chat_id, user_id, query_text, is_user=True)
        if error:
            return None, error
        
        # Get diary data with entries
        diary_data = get_diary_with_entries(user_id, chat.diary_id)
        if not diary_data:
            return None, "Diary not found"
        
        # Enhanced entry processing
        entries_text = ""
        relevant_entry_ids = []
        entry_count = 0
        
        # Sort entries by relevance (simple keyword matching + date)
        query_tokens = set(query_text.lower().split())
        scored_entries = []
        
        for entry in diary_data.get("entries", []):
            entry_text = f"{entry['title']} {entry['text']}"
            if entry.get('caption'):
                entry_text += f" {entry['caption']}"
            
            # Simple relevance scoring
            entry_tokens = set(entry_text.lower().split())
            overlap_score = len(query_tokens & entry_tokens) / len(query_tokens) if query_tokens else 0
            
            scored_entries.append({
                "entry": entry,
                "score": overlap_score,
                "text": entry_text
            })
        
        # Sort by relevance, then by date
        scored_entries.sort(key=lambda x: (x["score"], x["entry"]["created_at"]), reverse=True)
        
        # Include top relevant entries or recent ones if no good matches
        max_entries = 10
        selected_entries = scored_entries[:max_entries] if scored_entries[0]["score"] > 0 else scored_entries[-5:]
        
        for scored_entry in selected_entries:
            entry = scored_entry["entry"]
            entry_id = entry["id"]
            
            entry_header = f"Entry from {entry['created_at']}"
            if entry.get('title'):
                entry_header += f", Title: {entry['title']}"
            
            entry_content = entry['text']
            if entry.get('caption'):
                entry_content += f"\nCaption: {entry['caption']}"
            
            entries_text += f"\n--- ENTRY {entry_id} ({entry_header}) ---\n{entry_content}\n"
            relevant_entry_ids.append(entry_id)
            entry_count += 1
        
        # Get conversation history
        previous_messages = ChatMessage.query.filter_by(chat_id=chat_id)\
            .order_by(ChatMessage.timestamp.desc()).limit(8).all()
        
        history_text = "\n".join([
            f"{'User' if msg.is_user else 'Assistant'}: {msg.content}"
            for msg in reversed(previous_messages[1:])
        ])
        
        # Enhanced response generation
        response_text = ""
        if entries_text:
            diary_context = f"""
This is a conversation about the user's personal diary "{diary_data['name']}".
The diary contains {len(diary_data.get('entries', []))} total entries.
Showing {entry_count} most relevant entries for this query.
"""
            
            if history_text:
                prompt = f"""
You are an AI assistant helping a user explore and understand their personal diary.

{diary_context}

Previous conversation:
{history_text}

Based on the conversation history and the following diary entries, provide a thoughtful, 
personal response to the user's question. Consider patterns, emotions, and insights 
that might be helpful.

Relevant diary entries:
{entries_text}

User question: {query_text}

Your response should be empathetic, insightful, and reference specific entries when relevant.
Your response:
"""
            else:
                prompt = f"""
You are an AI assistant helping a user explore and understand their personal diary.

{diary_context}

Based on the following diary entries, provide a thoughtful, personal response to the 
user's question. Look for patterns, emotions, and insights that might be helpful.

Relevant diary entries:
{entries_text}

User question: {query_text}

Your response should be empathetic, insightful, and reference specific entries when relevant.
Your response:
"""
            
            try:
                output = ollama.generate(
                    model="llama3",
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                )
                response_text = output['response']
                
                # Add entry summary if many entries were searched
                if entry_count > 3:
                    response_text += f"\n\n*Based on analysis of {entry_count} diary entries from your collection.*"
                
            except Exception as e:
                logger.error(f"Error generating diary response: {e}")
                response_text = f"I had trouble processing your question about your diary. Technical error: {str(e)}"
        else:
            response_text = "I don't see any entries in your diary yet. Add some entries and then we can chat about them!"
        
        # Store the AI message
        ai_message, error = add_message_to_chat(
            chat_id, 
            user_id, 
            response_text, 
            is_user=False,
            relevant_memory_ids=",".join(map(str, relevant_entry_ids))
        )
        
        if error:
            return None, error
        
        return {
            "query": query_text,
            "response": response_text,
            "relevant_entries": [scored_entry["entry"] for scored_entry in selected_entries],
            "total_entries_analyzed": len(diary_data.get("entries", [])),
            "entries_used": entry_count
        }, None
        
    except Exception as e:
        logger.error(f"Error processing diary chat query: {e}")
        return None, str(e)

def delete_chat_session(chat_id, user_id):
    """Enhanced chat session deletion with cleanup"""
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return False, "Chat session not found"
    
    try:
        # Clean up conversation context if exists
        conversation_context.clear_session(chat_id)
        
        # Delete the chat (messages will be cascade deleted)
        db.session.delete(chat)
        db.session.commit()
        
        logger.info(f"Deleted chat session {chat_id}")
        return True, None
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting chat session: {e}")
        return False, str(e)

# Additional enhanced features
def get_chat_suggestions(user_id, chat_id, limit=5):
    """Get suggested questions based on chat context and available memories"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return []
        
        suggestions = []
        
        # Get recent messages for context
        recent_messages = ChatMessage.query.filter_by(chat_id=chat_id)\
            .order_by(ChatMessage.timestamp.desc()).limit(3).all()
        
        if recent_messages:
            last_message = recent_messages[0].content
            
            # Get search suggestions based on last message
            search_suggestions_list = search_suggestions(user_id, last_message, limit)
            suggestions.extend([s["text"] for s in search_suggestions_list])
        
        # Add contextual suggestions based on chat type
        if chat.memory_id:
            # Memory-specific suggestions
            suggestions.extend([
                "Can you summarize the key points?",
                "What are the main themes in this document?",
                "Find specific details about...",
                "How does this relate to other documents?"
            ])
        elif chat.collection_id:
            # Collection-wide suggestions
            suggestions.extend([
                "What are the common themes across all documents?",
                "Find documents related to...",
                "Compare information between documents",
                "Show me the most recent additions"
            ])
        elif chat.diary_id:
            # Diary-specific suggestions
            suggestions.extend([
                "What patterns do you see in my entries?",
                "How has my mood changed over time?",
                "Find entries about specific topics",
                "What insights can you share?"
            ])
        
        return suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Error getting chat suggestions: {e}")
        return []

def get_chat_analytics(user_id, chat_id):
    """Get analytics for a specific chat session"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return None
        
        # Basic message statistics
        total_messages = ChatMessage.query.filter_by(chat_id=chat_id).count()
        user_messages = ChatMessage.query.filter_by(chat_id=chat_id, is_user=True).count()
        ai_messages = total_messages - user_messages
        
        # Get message history
        messages = ChatMessage.query.filter_by(chat_id=chat_id)\
            .order_by(ChatMessage.timestamp).all()
        
        # Calculate session duration
        if messages:
            session_start = messages[0].timestamp
            session_end = messages[-1].timestamp
            duration_minutes = (session_end - session_start).total_seconds() / 60
        else:
            duration_minutes = 0
        
        # Analyze memory usage
        unique_memories_used = set()
        for message in messages:
            if message.relevant_memory_ids:
                memory_ids = message.relevant_memory_ids.split(',')
                unique_memories_used.update(memory_ids)
        
        analytics = {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "session_duration_minutes": round(duration_minutes, 2),
            "unique_memories_referenced": len(unique_memories_used),
            "chat_type": "memory" if chat.memory_id else "collection" if chat.collection_id else "diary",
            "created_at": chat.created_at.isoformat(),
            "last_updated": chat.updated_at.isoformat()
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting chat analytics: {e}")
        return None