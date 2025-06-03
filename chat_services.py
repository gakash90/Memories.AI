# Core Flask/SQLAlchemy imports
from models import Chat, ChatMessage
from extensions import db

# Enhanced services imports
from services import (
    query_collection, get_collection, generate_response, 
    get_collection_documents_path, query_specific_memory,
    query_across_collections, global_search, search_suggestions,
    handle_errors, logger
)

# Other service imports
from diary_services import get_diary, get_diary_with_entries
from knowledge_graph import ConversationContext

# Standard library imports
from datetime import datetime, timedelta
import ollama
import numpy as np
import os
import json
import logging
import threading
import time
import hashlib
import sys
import traceback
from typing import List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


# Enhanced logging for chat services
chat_logger = logging.getLogger(__name__ + '.chat')

# Enhanced conversation context with thread safety
conversation_context = ConversationContext()

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now():
    return datetime.now(IST).replace(tzinfo=None)

# Chat session analytics storage
class ChatAnalytics:
    """Thread-safe chat analytics manager"""
    
    def __init__(self):
        self._analytics = {}  # chat_id -> analytics data
        self._lock = threading.Lock()
    
    def track_query(self, chat_id, query_text, response_time, results_count, search_type):
        """Track query analytics"""
        with self._lock:
            if chat_id not in self._analytics:
                self._analytics[chat_id] = {
                    "queries": [],
                    "total_queries": 0,
                    "avg_response_time": 0,
                    "search_types_used": {},
                    "common_topics": []
                }
            
            analytics = self._analytics[chat_id]
            analytics["queries"].append({
                "query": query_text[:100],  # Truncate for privacy
                "response_time": response_time,
                "results_count": results_count,
                "search_type": search_type,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update aggregated metrics
            analytics["total_queries"] += 1
            total_time = sum(q["response_time"] for q in analytics["queries"])
            analytics["avg_response_time"] = total_time / len(analytics["queries"])
            
            # Track search type usage
            analytics["search_types_used"][search_type] = analytics["search_types_used"].get(search_type, 0) + 1
            
            # Keep only last 50 queries for memory management
            if len(analytics["queries"]) > 50:
                analytics["queries"] = analytics["queries"][-50:]
    
    def get_analytics(self, chat_id):
        """Get analytics for a chat session"""
        with self._lock:
            return self._analytics.get(chat_id, {})
    
    def clear_analytics(self, chat_id):
        """Clear analytics for a chat session"""
        with self._lock:
            if chat_id in self._analytics:
                del self._analytics[chat_id]

# Global analytics instance
chat_analytics = ChatAnalytics()

# Enhanced error handling for chat operations
@contextmanager
def chat_error_handler(operation_name, chat_id=None):
    """Context manager for consistent chat error handling"""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error in {operation_name}"
        if chat_id:
            error_msg += f" (chat_id: {chat_id})"
        error_msg += f" after {duration:.2f}s: {e}"
        
        chat_logger.error(error_msg, exc_info=True)
        raise e

# Enhanced chat session management with better validation
@handle_errors(default_return=(None, "Error creating memory chat session"))
def create_memory_chat_session(user_id, collection_id, memory_id):
    """Enhanced memory chat session creation with comprehensive validation"""
    with chat_error_handler("create_memory_chat_session"):
        # Validate inputs
        if not all([user_id, collection_id, memory_id]):
            return None, "Missing required parameters"
        

        existing_empty_chat = get_empty_chat_session(user_id, collection_id=collection_id, memory_id=memory_id)
        if existing_empty_chat:
            chat_logger.info(f"Reusing empty memory chat session {existing_empty_chat.id}")
            return existing_empty_chat, None
        
        collection = get_collection(user_id, collection_id)
        if not collection:
            return None, "Collection not found"
        
        # Find and validate the memory
        memory = None
        for mem in collection.get("memories", []):
            if mem["id"] == memory_id:
                memory = mem
                break
        
        if not memory:
            return None, "Memory not found in collection"
        
        # Validate memory status
        if memory.get("status") != "completed":
            return None, "Memory is not ready for chat (processing incomplete)"
        
        # Create enhanced chat title with metadata
        memory_type = memory.get('type', 'document').title()
        original_filename = memory.get('original_filename', 'Unknown')
        file_size = memory.get('file_size_mb', 0)
        chunk_count = memory.get('chunk_count', 0)
        
        title = f"Chat: {memory['title']} ({memory_type})"
        if file_size > 0:
            title += f" - {file_size:.1f}MB"
        if chunk_count > 1:
            title += f" - {chunk_count} sections"
        
        chat = Chat(
            user_id=user_id,
            collection_id=collection_id,
            memory_id=memory_id,
            title=title,
            created_at=ist_now(),
            updated_at=ist_now()
        )
        
        try:
            db.session.add(chat)
            db.session.commit()
            
            # Initialize analytics
            chat_analytics.track_query(chat.id, "", 0, 0, "session_created")
            
            chat_logger.info(f"Created memory chat session {chat.id} for memory {memory_id}")
            return chat, None
            
        except Exception as e:
            db.session.rollback()
            raise e
        
@handle_errors(default_return=None)
def get_empty_chat_session(user_id, collection_id=None, memory_id=None, diary_id=None):
    """Find existing empty chat session for reuse"""
    with chat_error_handler("get_empty_chat_session"):
        if not user_id:
            return None
        
        # Build query based on chat type
        query = Chat.query.filter_by(user_id=user_id)
        
        if collection_id and not memory_id:
            # Collection chat
            query = query.filter_by(collection_id=collection_id, memory_id=None)
        elif memory_id:
            # Memory chat
            query = query.filter_by(collection_id=collection_id, memory_id=memory_id)
        elif diary_id:
            # Diary chat
            query = query.filter_by(diary_id=diary_id)
        else:
            return None
        
        # Get recent chats (last 5)
        recent_chats = query.order_by(Chat.updated_at.desc()).limit(5).all()
        
        # Check if any recent chat is empty (no messages)
        for chat in recent_chats:
            message_count = ChatMessage.query.filter_by(chat_id=chat.id).count()
            if message_count == 0:
                return chat
        
        return None

@handle_errors(default_return=(None, "Error creating collection chat session"))
def create_chat_session(user_id, collection_id):
    """Enhanced collection chat session creation with validation"""
    with chat_error_handler("create_chat_session"):
        if not all([user_id, collection_id]):
            return None, "Missing required parameters"
        

        existing_empty_chat = get_empty_chat_session(user_id, collection_id=collection_id)
        if existing_empty_chat:
            chat_logger.info(f"Reusing empty chat session {existing_empty_chat.id}")
            return existing_empty_chat, None
        
        collection = get_collection(user_id, collection_id)
        if not collection:
            return None, "Collection not found"
        
        # Validate collection has searchable content
        stats = collection.get('stats', {})
        total_memories = stats.get('total_memories', len(collection.get('memories', [])))
        
        if total_memories == 0:
            return None, "Collection is empty - add some documents first"
        
        # Create descriptive title with enhanced statistics
        memory_count = total_memories
        memory_types = stats.get('memory_types', {})
        type_summary = ", ".join([f"{count} {type_}" for type_, count in memory_types.items()])
        
        title = f"Chat: {collection['name']} ({memory_count} documents"
        if type_summary:
            title += f" - {type_summary}"
        title += ")"
        
        chat = Chat(
            user_id=user_id,
            collection_id=collection_id,
            title=title,
            created_at=ist_now(),
            updated_at=ist_now()
        )
        
        try:
            db.session.add(chat)
            db.session.commit()
            
            # Initialize analytics
            chat_analytics.track_query(chat.id, "", 0, 0, "session_created")
            
            chat_logger.info(f"Created collection chat session {chat.id} for collection {collection_id}")
            return chat, None
            
        except Exception as e:
            db.session.rollback()
            raise e

@handle_errors(default_return=[])
def get_chat_sessions(user_id, collection_id=None, memory_id=None):
    """Enhanced chat session retrieval with better filtering and metadata"""
    with chat_error_handler("get_chat_sessions"):
        if not user_id:
            return []
        
        query = Chat.query.filter_by(user_id=user_id)
        
        if collection_id:
            query = query.filter_by(collection_id=collection_id)
        
        # Enhanced filtering logic with validation
        if memory_id is None and collection_id is not None:
            # Collection-level chats only
            query = query.filter(Chat.memory_id.is_(None))
        elif memory_id:
            # Specific memory chats only
            query = query.filter_by(memory_id=memory_id)
        
        # Order by most recent activity with pagination consideration
        chat_sessions = query.order_by(Chat.updated_at.desc()).limit(100).all()
        
        # Add enhanced session metadata
        for chat in chat_sessions:
            try:
                # Count messages in this chat
                message_count = ChatMessage.query.filter_by(chat_id=chat.id).count()
                chat.message_count = message_count
                
                # Get last message timestamp with validation
                last_message = ChatMessage.query.filter_by(chat_id=chat.id)\
                    .order_by(ChatMessage.timestamp.desc()).first()
                chat.last_activity = last_message.timestamp if last_message else chat.created_at
                
                # Add analytics summary
                analytics = chat_analytics.get_analytics(chat.id)
                chat.analytics_summary = {
                    "total_queries": analytics.get("total_queries", 0),
                    "avg_response_time": round(analytics.get("avg_response_time", 0), 2),
                    "most_used_search": max(analytics.get("search_types_used", {"hybrid": 1}).items(), 
                                          key=lambda x: x[1])[0] if analytics.get("search_types_used") else "hybrid"
                }
                
            except Exception as e:
                chat_logger.warning(f"Error adding metadata to chat {chat.id}: {e}")
                # Set default values
                chat.message_count = 0
                chat.last_activity = chat.created_at
                chat.analytics_summary = {"total_queries": 0, "avg_response_time": 0, "most_used_search": "hybrid"}
        
        return chat_sessions

@handle_errors(default_return=(None, "Error retrieving chat messages"))
def get_chat_messages(chat_id, user_id):
    """Enhanced chat message retrieval with validation and pagination support"""
    with chat_error_handler("get_chat_messages", chat_id):
        if not all([chat_id, user_id]):
            return None, "Missing required parameters"
        
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return None, "Chat session not found or access denied"
        
        # Get messages with enhanced error handling
        try:
            messages = ChatMessage.query.filter_by(chat_id=chat.id)\
                .order_by(ChatMessage.timestamp).all()
            
            # Enhanced message processing with validation
            processed_messages = []
            for message in messages:
                try:
                    # Parse relevant memory IDs with error handling
                    if message.relevant_memory_ids:
                        try:
                            if isinstance(message.relevant_memory_ids, str):
                                memory_ids = [id.strip() for id in message.relevant_memory_ids.split(',') if id.strip()]
                            else:
                                memory_ids = [str(message.relevant_memory_ids)]
                            message.parsed_memory_ids = memory_ids
                        except Exception as e:
                            chat_logger.warning(f"Error parsing memory IDs for message {message.id}: {e}")
                            message.parsed_memory_ids = []
                    else:
                        message.parsed_memory_ids = []
                    
                    # Validate message content
                    if not message.content:
                        chat_logger.warning(f"Empty content for message {message.id}")
                        continue
                    
                    processed_messages.append(message)
                    
                except Exception as e:
                    chat_logger.warning(f"Error processing message {message.id}: {e}")
                    continue
            
            return chat, processed_messages
            
        except Exception as e:
            chat_logger.error(f"Error retrieving messages for chat {chat_id}: {e}")
            return chat, []

@handle_errors(default_return=(None, "Error adding message to chat"))
def add_message_to_chat(chat_id, user_id, content, is_user=True, relevant_memory_ids=None):
    """Enhanced message addition with validation and metadata handling"""
    with chat_error_handler("add_message_to_chat", chat_id):
        if not all([chat_id, user_id, content]):
            return None, "Missing required parameters"
        
        if not content.strip():
            return None, "Message content cannot be empty"
        
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return None, "Chat session not found or access denied"
        
        # Enhanced memory ID handling with validation
        memory_ids_str = None
        if relevant_memory_ids:
            try:
                if isinstance(relevant_memory_ids, list):
                    # Validate each memory ID
                    valid_ids = [str(id).strip() for id in relevant_memory_ids if str(id).strip()]
                    memory_ids_str = ",".join(valid_ids) if valid_ids else None
                else:
                    memory_ids_str = str(relevant_memory_ids).strip() if str(relevant_memory_ids).strip() else None
            except Exception as e:
                chat_logger.warning(f"Error processing relevant_memory_ids: {e}")
                memory_ids_str = None
        
        # Create message with enhanced metadata
        message = ChatMessage(
            chat_id=chat.id,
            content=content.strip(),
            is_user=is_user,
            timestamp=ist_now(),
            relevant_memory_ids=memory_ids_str
        )
        
        # Update chat timestamp
        chat.updated_at = ist_now()
        
        try:
            db.session.add(message)
            db.session.commit()
            
            # Log message for analytics
            message_type = "user" if is_user else "assistant"
            chat_logger.info(f"Added {message_type} message to chat {chat.id} (length: {len(content)})")
            
            return message, None
            
        except Exception as e:
            db.session.rollback()
            raise e

# CRITICAL FIX: Enhanced chat query processing with proper error handling and search orchestration
@handle_errors(default_return=(None, "Error processing chat query"))
def process_chat_query(chat_id, user_id, query_text, search_type="hybrid"):
    """Enhanced chat query processing with comprehensive error handling and search optimization"""
    start_time = time.time()
    
    with chat_error_handler("process_chat_query", chat_id):
        # Input validation
        if not all([chat_id, user_id, query_text]):
            return None, "Missing required parameters"
        
        if not query_text.strip():
            return None, "Query cannot be empty"
        
        if len(query_text.strip()) > 1000:
            return None, "Query too long (max 1000 characters)"
        
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return None, "Chat session not found or access denied"
        
        query_text = query_text.strip()
        
        try:
            # Step 1: Initialize conversation context
            conversation_context.start_session(chat_id)
            
            # Step 2: Analyze query in conversation context
            context_analysis = conversation_context.analyze_question_context(query_text)
            search_query = context_analysis.get('expanded_question', query_text)
            
            # Step 3: Add user message to database first
            user_message, error = add_message_to_chat(chat_id, user_id, query_text, is_user=True)
            
            if error:
                return None, error
            
            # Step 4: Add to conversation graph
            conversation_context.add_message(
                user_message.id, 
                query_text, 
                True, 
                user_message.timestamp.isoformat()
            )
            
            # Step 5: Enhanced search orchestration with parallel processing and fallbacks
            relevant_memories = []
            search_metadata = {
                "search_strategies_used": [],
                "total_search_time": 0,
                "search_results_count": 0
            }
            
            search_start_time = time.time()
            
            # Primary search strategy - Collection-specific search
            if chat.collection_id:
                try:
                    primary_memories, error = query_collection(
                        user_id, 
                        chat.collection_id, 
                        search_query, 
                        top_k=15,  # Increased for better coverage
                        search_type=search_type
                    )
                    
                    if not error and primary_memories:
                        relevant_memories.extend(primary_memories)
                        search_metadata["search_strategies_used"].append({
                            "strategy": "primary_collection",
                            "collection_id": chat.collection_id,
                            "results_count": len(primary_memories),
                            "search_type": search_type
                        })
                    
                except Exception as e:
                    chat_logger.warning(f"Primary collection search failed: {e}")
            
            # Secondary search strategy - Cross-collection search if needed
            if len(relevant_memories) < 5:
                try:
                    cross_memories, error = query_across_collections(
                        user_id, 
                        search_query, 
                        collection_ids=None,  # Search all collections
                        top_k=10,
                        search_type=search_type
                    )
                    
                    if not error and cross_memories:
                        # Filter out duplicates and add unique results
                        existing_memory_ids = {m["metadata"]["id"] for m in relevant_memories}
                        new_memories = [m for m in cross_memories 
                                      if m["metadata"]["id"] not in existing_memory_ids]
                        
                        relevant_memories.extend(new_memories[:8])  # Limit additional results
                        
                        search_metadata["search_strategies_used"].append({
                            "strategy": "cross_collection",
                            "results_count": len(new_memories),
                            "total_collections_searched": len(set(m.get("source_collection_id") for m in cross_memories))
                        })
                
                except Exception as e:
                    chat_logger.warning(f"Cross-collection search failed: {e}")
            
            # Tertiary search strategy - Global search as last resort
            if len(relevant_memories) < 3:
                try:
                    global_results, error = global_search(
                        user_id,
                        search_query,
                        search_type=search_type,
                        limit=8
                    )
                    
                    if not error and global_results:
                        # Convert global results to memory format
                        existing_memory_ids = {m["metadata"]["id"] for m in relevant_memories}
                        
                        for result in global_results:
                            if result["document_id"] not in existing_memory_ids:
                                relevant_memories.append({
                                    "metadata": {
                                        "id": result["document_id"],
                                        "title": result["title"],
                                        "type": result["memory_type"],
                                        "collection_id": result["collection_id"]
                                    },
                                    "content": result["content"],
                                    "score": result["score"],
                                    "search_type": "global"
                                })
                        
                        search_metadata["search_strategies_used"].append({
                            "strategy": "global_search",
                            "results_count": len(global_results)
                        })
                
                except Exception as e:
                    chat_logger.warning(f"Global search failed: {e}")
            
            search_metadata["total_search_time"] = time.time() - search_start_time
            search_metadata["search_results_count"] = len(relevant_memories)
            
            # Step 6: Enhanced conversation history retrieval
            history_text = ""
            try:
                history = conversation_context.get_conversation_history(limit=5)
                if history:
                    history_messages = []
                    for msg in history[:-1]:  # Exclude current question
                        role = "User" if msg['is_user'] else "Assistant"
                        content = msg['content'][:200]  # Truncate long messages
                        history_messages.append(f"{role}: {content}")
                    history_text = "\n".join(history_messages)
            except Exception as e:
                chat_logger.warning(f"Error retrieving conversation history: {e}")
                history_text = ""
            
            # Step 7: Enhanced response generation with retry logic
            if not relevant_memories:
                response_text = "I don't have any relevant information to answer your question. Try rephrasing your query or check if you have uploaded related documents to your collections."
            else:
                try:
                    response_text = generate_response(
                        search_query, 
                        relevant_memories[:8],  # Limit to top 8 for context
                        history_text if history_text.strip() else None
                    )
                    
                    # Validate response
                    if not response_text or not response_text.strip():
                        response_text = "I found relevant information but had trouble generating a response. Please try rephrasing your question."
                    
                except Exception as e:
                    chat_logger.error(f"Error generating response: {e}")
                    response_text = f"I found relevant information but encountered an error while generating the response. Please try again."
            
            # Step 8: Add search insights to response
            if search_metadata["search_strategies_used"] and len(relevant_memories) > 3:
                insights = []
                for strategy in search_metadata["search_strategies_used"]:
                    if strategy["strategy"] == "cross_collection":
                        insights.append(f"Searched across {strategy['total_collections_searched']} collections")
                    elif strategy["strategy"] == "global_search":
                        insights.append("Expanded search to your entire knowledge base")
                
                if insights:
                    response_text += f"\n\n*Search insights: {'; '.join(insights)}*"
            
            # Step 9: Prepare memory IDs for database storage
            memory_ids = []
            if relevant_memories:
                memory_ids = [memory['metadata']['id'] for memory in relevant_memories[:5]]
            
            # Step 10: Add AI response to database
            ai_message, error = add_message_to_chat(
                chat_id, 
                user_id, 
                response_text, 
                is_user=False,
                relevant_memory_ids=memory_ids
            )
            
            if error:
                return None, error
            
            # Step 11: Update conversation graph with enhanced metadata
            try:
                conversation_context.add_message(
                    ai_message.id, 
                    response_text, 
                    False, 
                    ai_message.timestamp.isoformat(),
                    related_entities=[(mem_id, 'memory') for mem_id in memory_ids]
                )
            except Exception as e:
                chat_logger.warning(f"Error updating conversation graph: {e}")
            
            # Step 12: Track analytics
            total_time = time.time() - start_time
            chat_analytics.track_query(
                chat_id, 
                query_text, 
                total_time, 
                len(relevant_memories), 
                search_type
            )
            
            # Step 13: Build comprehensive response
            response_data = {
                "query": query_text,
                "expanded_query": search_query if search_query != query_text else None,
                "response": response_text,
                "relevant_memories": [m["metadata"] for m in relevant_memories[:8]],
                "search_metadata": search_metadata,
                "total_results_found": len(relevant_memories),
                "response_time_seconds": round(total_time, 2),
                "conversation_context_used": bool(history_text.strip())
            }
            
            chat_logger.info(f"Successfully processed query for chat {chat_id} in {total_time:.2f}s")
            return response_data, None
            
        except Exception as e:
            chat_logger.error(f"Error in chat query processing: {e}")
            return None, str(e)

@handle_errors(default_return=(None, "Error processing memory chat query"))
def process_memory_chat_query(chat_id, user_id, query_text):
    """Enhanced memory-specific chat query processing with comprehensive error handling"""
    start_time = time.time()
    
    with chat_error_handler("process_memory_chat_query", chat_id):
        # Input validation
        if not all([chat_id, user_id, query_text]):
            return None, "Missing required parameters"
        
        if not query_text.strip():
            return None, "Query cannot be empty"
        
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return None, "Chat session not found or access denied"
        
        if not chat.memory_id:
            return None, "This is not a memory-specific chat"
        
        query_text = query_text.strip()
        
        try:
            # Step 1: Add user message
            user_message, error = add_message_to_chat(chat_id, user_id, query_text, is_user=True)
            if error:
                return None, error
            
            # Step 2: Enhanced memory querying with validation
            memory_result, error = query_specific_memory(
                user_id, 
                chat.collection_id, 
                chat.memory_id, 
                query_text
            )
            
            if error:
                return None, error
            
            if not memory_result:
                return None, "Could not retrieve memory content"
            
            # Step 3: Get enhanced conversation history
            history_text = ""
            try:
                previous_messages = ChatMessage.query.filter_by(chat_id=chat_id)\
                    .order_by(ChatMessage.timestamp.desc()).limit(8).all()
                
                if previous_messages:
                    history_messages = []
                    for msg in reversed(previous_messages[1:]):  # Exclude current, reverse to chronological
                        role = "User" if msg.is_user else "Assistant"
                        content = msg.content[:150]  # Truncate for context
                        history_messages.append(f"{role}: {content}")
                    history_text = "\n".join(history_messages)
            except Exception as e:
                chat_logger.warning(f"Error retrieving memory chat history: {e}")
            
            # Step 4: Enhanced response generation for specific memory
            response_text = ""
            if memory_result and len(memory_result) > 0:
                memory_data = memory_result[0]
                
                # Build enhanced memory context
                memory_context = f"""
This is a focused conversation about a specific document:
- Title: {memory_data['metadata']['title']}
- Type: {memory_data['metadata']['type']}
- File: {memory_data['metadata'].get('original_filename', 'Unknown')}
"""
                
                # Add chunk information if available
                if 'chunks_used' in memory_data:
                    memory_context += f"- Sections analyzed: {memory_data['chunks_used']}\n"
                
                if 'chunk_details' in memory_data:
                    section_info = ", ".join([f"Section {d['index']+1} (relevance: {d['score']:.2f})" 
                                            for d in memory_data['chunk_details']])
                    memory_context += f"- Relevant sections: {section_info}\n"
                
                # Generate contextual response
                enhanced_history = f"{memory_context}\n{history_text}" if history_text else memory_context
                
                try:
                    response_text = generate_response(query_text, memory_result, enhanced_history)
                    
                    # Validate response
                    if not response_text or not response_text.strip():
                        response_text = "I found the document but had trouble generating a response. Please try rephrasing your question."
                    
                except Exception as e:
                    chat_logger.error(f"Error generating memory response: {e}")
                    response_text = "I found the document but encountered an error while generating the response."
                
                # Add memory-specific insights
                if memory_data.get('chunks_used', 0) > 1:
                    response_text += f"\n\n*Response based on {memory_data['chunks_used']} most relevant sections of this document.*"
                elif memory_data.get('fallback_method'):
                    response_text += "\n\n*Response based on full document content.*"
            else:
                response_text = "I couldn't find relevant information in this specific document to answer your question. Try asking about different aspects of the document."
            
            # Step 5: Add AI response to database
            ai_message, error = add_message_to_chat(
                chat_id, 
                user_id, 
                response_text, 
                is_user=False,
                relevant_memory_ids=chat.memory_id
            )
            
            if error:
                return None, error
            
            # Step 6: Track analytics
            total_time = time.time() - start_time
            chat_analytics.track_query(
                chat_id, 
                query_text, 
                total_time, 
                len(memory_result) if memory_result else 0, 
                "memory_specific"
            )
            
            # Step 7: Build response
            response_data = {
                "query": query_text,
                "response": response_text,
                "relevant_memories": [memory_result[0]["metadata"]] if memory_result else [],
                "memory_specific": True,
                "memory_id": chat.memory_id,
                "response_time_seconds": round(total_time, 2),
                "sections_used": memory_result[0].get('chunks_used', 1) if memory_result else 0
            }
            
            chat_logger.info(f"Successfully processed memory query for chat {chat_id} in {total_time:.2f}s")
            return response_data, None
        
        except Exception as e:
            chat_logger.error(f"Error processing memory chat query: {e}")
            return None, str(e)

@handle_errors(default_return=(None, "Error creating diary chat session"))
def create_diary_chat_session(user_id, diary_id):
    """Enhanced diary chat session creation with validation"""
    with chat_error_handler("create_diary_chat_session"):
        if not all([user_id, diary_id]):
            return None, "Missing required parameters"
        

        existing_empty_chat = get_empty_chat_session(user_id, diary_id=diary_id)
        if existing_empty_chat:
            chat_logger.info(f"Reusing empty diary chat session {existing_empty_chat.id}")
            return existing_empty_chat, None
        
        diary = get_diary(user_id, diary_id)
        if not diary:
            return None, "Diary not found or access denied"
        
        # Validate diary has entries
        diary_with_entries = get_diary_with_entries(user_id, diary_id)
        entry_count = len(diary_with_entries.get("entries", [])) if diary_with_entries else 0
        
        # Create descriptive title with entry count
        title = f"Diary Chat: {diary['name']}"
        if entry_count > 0:
            title += f" ({entry_count} entries)"
        else:
            title += " (empty)"
        
        chat = Chat(
            user_id=user_id,
            diary_id=diary_id,
            title=title,
            created_at=ist_now(),
            updated_at=ist_now()
        )
        
        try:
            db.session.add(chat)
            db.session.commit()
            
            # Initialize analytics
            chat_analytics.track_query(chat.id, "", 0, entry_count, "diary_session_created")
            
            chat_logger.info(f"Created diary chat session {chat.id} for diary {diary_id}")
            return chat, None
            
        except Exception as e:
            db.session.rollback()
            raise e

@handle_errors(default_return=[])
def get_diary_chat_sessions(user_id, diary_id):
    """Enhanced diary chat session retrieval with validation"""
    with chat_error_handler("get_diary_chat_sessions"):
        if not all([user_id, diary_id]):
            return []
        
        # Validate diary exists
        diary = get_diary(user_id, diary_id)
        if not diary:
            return []
        
        chat_sessions = Chat.query.filter_by(
            user_id=user_id,
            diary_id=diary_id
        ).order_by(Chat.updated_at.desc()).all()
        
        # Add enhanced session metadata
        for chat in chat_sessions:
            try:
                message_count = ChatMessage.query.filter_by(chat_id=chat.id).count()
                chat.message_count = message_count
                
                last_message = ChatMessage.query.filter_by(chat_id=chat.id)\
                    .order_by(ChatMessage.timestamp.desc()).first()
                chat.last_activity = last_message.timestamp if last_message else chat.created_at
                
                # Add analytics
                analytics = chat_analytics.get_analytics(chat.id)
                chat.analytics_summary = {
                    "total_queries": analytics.get("total_queries", 0),
                    "avg_response_time": round(analytics.get("avg_response_time", 0), 2)
                }
                
            except Exception as e:
                chat_logger.warning(f"Error adding metadata to diary chat {chat.id}: {e}")
                chat.message_count = 0
                chat.last_activity = chat.created_at
                chat.analytics_summary = {"total_queries": 0, "avg_response_time": 0}
        
        return chat_sessions

@handle_errors(default_return=(None, "Error processing diary chat query"))
def process_diary_chat_query(chat_id, user_id, query_text):
    """Enhanced diary chat query processing with improved entry analysis"""
    start_time = time.time()
    
    with chat_error_handler("process_diary_chat_query", chat_id):
        # Input validation
        if not all([chat_id, user_id, query_text]):
            return None, "Missing required parameters"
        
        if not query_text.strip():
            return None, "Query cannot be empty"
        
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return None, "Chat session not found or access denied"
        
        if not chat.diary_id:
            return None, "This is not a diary chat"
        
        query_text = query_text.strip()
        
        try:
            # Step 1: Add user message
            user_message, error = add_message_to_chat(chat_id, user_id, query_text, is_user=True)
            if error:
                return None, error
            
            # Step 2: Get diary data with comprehensive error handling
            diary_data = get_diary_with_entries(user_id, chat.diary_id)
            if not diary_data:
                return None, "Diary not found or access denied"
            
            entries = diary_data.get("entries", [])
            if not entries:
                response_text = "Your diary doesn't have any entries yet. Add some entries and then we can chat about them!"
            else:
                # Step 3: Enhanced entry processing with better relevance scoring
                query_tokens = set(query_text.lower().split())
                scored_entries = []
                
                for entry in entries:
                    # Build comprehensive entry text for analysis
                    entry_text_parts = []
                    if entry.get('title'):
                        entry_text_parts.append(entry['title'])
                    if entry.get('text'):
                        entry_text_parts.append(entry['text'])
                    if entry.get('caption'):
                        entry_text_parts.append(entry['caption'])
                    
                    entry_text = " ".join(entry_text_parts)
                    
                    if not entry_text.strip():
                        continue
                    
                    # Enhanced relevance scoring
                    entry_tokens = set(entry_text.lower().split())
                    
                    # Keyword overlap score
                    keyword_overlap = len(query_tokens & entry_tokens) / len(query_tokens) if query_tokens else 0
                    
                    # Date relevance (recent entries get slight boost)
                    try:
                        entry_date = datetime.fromisoformat(entry['created_at'])
                        days_ago = (datetime.now() - entry_date).days
                        date_boost = max(0, 1 - (days_ago / 365))  # Boost diminishes over a year
                    except:
                        date_boost = 0
                    
                    # Content length consideration (longer entries get slight boost)
                    length_boost = min(0.2, len(entry_text) / 1000)  # Max 0.2 boost
                    
                    # Final score
                    final_score = keyword_overlap + (date_boost * 0.1) + (length_boost * 0.1)
                    
                    scored_entries.append({
                        "entry": entry,
                        "score": final_score,
                        "text": entry_text,
                        "keyword_matches": len(query_tokens & entry_tokens)
                    })
                
                # Sort by relevance and select top entries
                scored_entries.sort(key=lambda x: (x["score"], x["keyword_matches"]), reverse=True)
                
                # Select entries based on relevance threshold
                relevant_entries = []
                min_score = 0.1  # Minimum relevance threshold
                
                for scored_entry in scored_entries:
                    if scored_entry["score"] >= min_score or len(relevant_entries) < 3:
                        relevant_entries.append(scored_entry)
                    if len(relevant_entries) >= 10:  # Maximum entries to include
                        break
                
                # If no relevant entries found, use most recent ones
                if not relevant_entries:
                    relevant_entries = scored_entries[-5:]  # Last 5 entries
                
                # Step 4: Build context for response generation
                entries_text = ""
                relevant_entry_ids = []
                
                for i, scored_entry in enumerate(relevant_entries):
                    entry = scored_entry["entry"]
                    entry_id = entry["id"]
                    
                    # Format entry with metadata
                    entry_header = f"Entry {i+1} from {entry['created_at']}"
                    if entry.get('title'):
                        entry_header += f" - {entry['title']}"
                    if scored_entry["score"] > 0:
                        entry_header += f" (relevance: {scored_entry['score']:.2f})"
                    
                    entry_content = entry.get('text', '')
                    if entry.get('caption'):
                        entry_content += f"\nCaption: {entry['caption']}"
                    
                    entries_text += f"\n--- {entry_header} ---\n{entry_content}\n"
                    relevant_entry_ids.append(entry_id)
                
                # Step 5: Get conversation history
                history_text = ""
                try:
                    previous_messages = ChatMessage.query.filter_by(chat_id=chat_id)\
                        .order_by(ChatMessage.timestamp.desc()).limit(6).all()
                    
                    if previous_messages:
                        history_messages = []
                        for msg in reversed(previous_messages[1:]):
                            role = "User" if msg.is_user else "Assistant"
                            content = msg.content[:100]  # Truncate for context
                            history_messages.append(f"{role}: {content}")
                        history_text = "\n".join(history_messages)
                except Exception as e:
                    chat_logger.warning(f"Error retrieving diary chat history: {e}")
                
                # Step 6: Enhanced response generation
                if entries_text:
                    diary_context = f"""
This is a conversation about the user's personal diary "{diary_data['name']}".
The diary contains {len(entries)} total entries.
Analyzing {len(relevant_entries)} most relevant entries for this query.
"""
                    
                    # Build comprehensive prompt
                    if history_text:
                        prompt = f"""
You are an AI assistant helping a user explore and understand their personal diary.

{diary_context}

Previous conversation:
{history_text}

Based on the conversation history and the following diary entries, provide a thoughtful, 
personal response to the user's question. Consider patterns, emotions, and insights 
that might be helpful. Be empathetic and supportive.

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
Be empathetic and supportive.

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
                                "top_p": 0.9,
                                "max_tokens": 800
                            }
                        )
                        response_text = output.get('response', '').strip()
                        
                        if not response_text:
                            response_text = "I found relevant entries but had trouble generating a response. Please try rephrasing your question."
                        
                        # Add analysis summary
                        if len(relevant_entries) > 2:
                            response_text += f"\n\n*Analysis based on {len(relevant_entries)} diary entries from your collection.*"
                        
                    except Exception as e:
                        chat_logger.error(f"Error generating diary response: {e}")
                        response_text = f"I found relevant entries but encountered an error while generating the response. Please try again."
                
                else:
                    response_text = "I couldn't find any relevant entries in your diary for this question. Try asking about different topics or add more entries to your diary."
            
            # Step 7: Store the AI message
            ai_message, error = add_message_to_chat(
                chat_id, 
                user_id, 
                response_text, 
                is_user=False,
                relevant_memory_ids=",".join(map(str, relevant_entry_ids)) if 'relevant_entry_ids' in locals() else None
            )
            
            if error:
                return None, error
            
            # Step 8: Track analytics
            total_time = time.time() - start_time
            entries_analyzed = len(relevant_entries) if 'relevant_entries' in locals() else 0
            chat_analytics.track_query(
                chat_id, 
                query_text, 
                total_time, 
                entries_analyzed, 
                "diary_analysis"
            )
            
            # Step 9: Build response
            response_data = {
                "query": query_text,
                "response": response_text,
                "relevant_entries": [entry["entry"] for entry in relevant_entries] if 'relevant_entries' in locals() else [],
                "total_entries_analyzed": len(entries),
                "entries_used": entries_analyzed,
                "response_time_seconds": round(total_time, 2)
            }
            
            chat_logger.info(f"Successfully processed diary query for chat {chat_id} in {total_time:.2f}s")
            return response_data, None
            
        except Exception as e:
            chat_logger.error(f"Error processing diary chat query: {e}")
            return None, str(e)

@handle_errors(default_return=(False, "Error deleting chat session"))
def delete_chat_session(chat_id, user_id):
    """Enhanced chat session deletion with comprehensive cleanup"""
    with chat_error_handler("delete_chat_session", chat_id):
        if not all([chat_id, user_id]):
            return False, "Missing required parameters"
        
        chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
        if not chat:
            return False, "Chat session not found or access denied"
        
        try:
            # Step 1: Clean up conversation context
            try:
                conversation_context.clear_session(chat_id)
            except Exception as e:
                chat_logger.warning(f"Error clearing conversation context: {e}")
            
            # Step 2: Clear analytics
            try:
                chat_analytics.clear_analytics(chat_id)
            except Exception as e:
                chat_logger.warning(f"Error clearing analytics: {e}")
            
            # Step 3: Delete the chat (messages will be cascade deleted)
            db.session.delete(chat)
            db.session.commit()
            
            chat_logger.info(f"Successfully deleted chat session {chat_id}")
            return True, None
            
        except Exception as e:
            db.session.rollback()
            raise e

# Enhanced utility functions for chat management
@handle_errors(default_return=[])
def get_chat_suggestions(user_id, chat_id, limit=5):
    """Enhanced chat suggestions with context awareness and better relevance"""
    with chat_error_handler("get_chat_suggestions"):
        if not all([user_id, chat_id]):
            return []
        
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
                try:
                    search_suggestions_list = search_suggestions(user_id, last_message, limit * 2)
                    suggestions.extend([s["text"] for s in search_suggestions_list[:limit]])
                except Exception as e:
                    chat_logger.warning(f"Error getting search suggestions: {e}")
            
            # Add contextual suggestions based on chat type
            if chat.memory_id:
                # Memory-specific suggestions
                memory_suggestions = [
                    "Can you summarize the key points?",
                    "What are the main themes in this document?",
                    "Find specific details about...",
                    "How does this relate to other documents?",
                    "What are the most important takeaways?"
                ]
                suggestions.extend(memory_suggestions)
            elif chat.collection_id:
                # Collection-wide suggestions
                collection_suggestions = [
                    "What are the common themes across all documents?",
                    "Find documents related to...",
                    "Compare information between documents",
                    "Show me the most recent additions",
                    "What topics do I have the most information about?"
                ]
                suggestions.extend(collection_suggestions)
            elif chat.diary_id:
                # Diary-specific suggestions
                diary_suggestions = [
                    "What patterns do you see in my entries?",
                    "How has my mood changed over time?",
                    "Find entries about specific topics",
                    "What insights can you share?",
                    "What were my main concerns this month?",
                    "Show me my most positive entries",
                    "What goals have I mentioned?",
                    "How do I handle stress based on my entries?"
                ]
                suggestions.extend(diary_suggestions)
            
            # Remove duplicates and return limited results
            unique_suggestions = []
            seen = set()
            for suggestion in suggestions:
                if suggestion.lower() not in seen and len(unique_suggestions) < limit:
                    unique_suggestions.append(suggestion)
                    seen.add(suggestion.lower())
            
            return unique_suggestions
            
        except Exception as e:
            chat_logger.warning(f"Error getting chat suggestions: {e}")
            return []

@handle_errors(default_return=None)
def get_chat_analytics(user_id, chat_id):
    """Enhanced chat analytics with comprehensive metrics and insights"""
    with chat_error_handler("get_chat_analytics"):
        if not all([user_id, chat_id]):
            return None
        
        try:
            chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
            if not chat:
                return None
            
            # Basic message statistics
            total_messages = ChatMessage.query.filter_by(chat_id=chat_id).count()
            user_messages = ChatMessage.query.filter_by(chat_id=chat_id, is_user=True).count()
            ai_messages = total_messages - user_messages
            
            # Get message history for analysis
            messages = ChatMessage.query.filter_by(chat_id=chat_id)\
                .order_by(ChatMessage.timestamp).all()
            
            # Calculate session metrics
            session_metrics = {
                "duration_minutes": 0,
                "message_frequency": 0,
                "avg_message_length": 0
            }
            
            if messages:
                session_start = messages[0].timestamp
                session_end = messages[-1].timestamp
                session_metrics["duration_minutes"] = round((session_end - session_start).total_seconds() / 60, 2)
                
                if session_metrics["duration_minutes"] > 0:
                    session_metrics["message_frequency"] = round(total_messages / session_metrics["duration_minutes"], 2)
                
                # Average message length
                total_chars = sum(len(msg.content) for msg in messages)
                session_metrics["avg_message_length"] = round(total_chars / total_messages, 1) if total_messages > 0 else 0
            
            # Analyze memory usage
            unique_memories_used = set()
            memory_references = 0
            
            for message in messages:
                if message.relevant_memory_ids:
                    try:
                        memory_ids = message.relevant_memory_ids.split(',')
                        unique_memories_used.update(memory_ids)
                        memory_references += len(memory_ids)
                    except:
                        pass
            
            # Get stored analytics
            stored_analytics = chat_analytics.get_analytics(chat_id)
            
            # Query pattern analysis
            query_patterns = {
                "question_words": 0,
                "search_terms": 0,
                "avg_query_length": 0
            }
            
            user_queries = [msg.content for msg in messages if msg.is_user]
            if user_queries:
                question_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
                query_patterns["question_words"] = sum(
                    1 for query in user_queries 
                    if any(word in query.lower() for word in question_indicators)
                )
                
                total_query_length = sum(len(query) for query in user_queries)
                query_patterns["avg_query_length"] = round(total_query_length / len(user_queries), 1)
            
            # Build comprehensive analytics
            analytics = {
                "basic_metrics": {
                    "total_messages": total_messages,
                    "user_messages": user_messages,
                    "ai_messages": ai_messages,
                    "message_ratio": round(ai_messages / user_messages, 2) if user_messages > 0 else 0
                },
                "session_metrics": session_metrics,
                "content_metrics": {
                    "unique_memories_referenced": len(unique_memories_used),
                    "total_memory_references": memory_references,
                    "memory_reuse_rate": round(memory_references / len(unique_memories_used), 2) if unique_memories_used else 0
                },
                "query_patterns": query_patterns,
                "performance_metrics": {
                    "total_queries": stored_analytics.get("total_queries", 0),
                    "avg_response_time": stored_analytics.get("avg_response_time", 0),
                    "search_types_used": stored_analytics.get("search_types_used", {})
                },
                "chat_metadata": {
                    "chat_type": "memory" if chat.memory_id else "collection" if chat.collection_id else "diary",
                    "created_at": chat.created_at.isoformat(),
                    "last_updated": chat.updated_at.isoformat(),
                    "title": chat.title
                },
                "insights": []
            }
            
            # Generate insights
            insights = []
            
            if analytics["session_metrics"]["duration_minutes"] > 60:
                insights.append("Long conversation session - indicates deep engagement")
            
            if analytics["content_metrics"]["memory_reuse_rate"] > 2:
                insights.append("High memory reuse - user is exploring topics in depth")
            
            if analytics["query_patterns"]["question_words"] / user_messages > 0.8 if user_messages > 0 else False:
                insights.append("Primarily inquiry-based conversation")
            
            if analytics["performance_metrics"]["avg_response_time"] < 3:
                insights.append("Fast response times - good system performance")
            
            analytics["insights"] = insights
            
            return analytics
            
        except Exception as e:
            chat_logger.error(f"Error getting chat analytics: {e}")
            return None

# Enhanced batch operations for chat management
@handle_errors(default_return=(0, "Error in batch operation"))
def cleanup_inactive_chats(user_id, days_inactive=30):
    """Clean up inactive chat sessions to improve performance"""
    with chat_error_handler("cleanup_inactive_chats"):
        if not user_id:
            return 0, "User ID required"
        
        try:
            cutoff_date = ist_now() - timedelta(days=days_inactive)
            
            # Find inactive chats
            inactive_chats = Chat.query.filter(
                Chat.user_id == user_id,
                Chat.updated_at < cutoff_date
            ).all()
            
            cleaned_count = 0
            
            for chat in inactive_chats:
                try:
                    # Check if chat has any messages
                    message_count = ChatMessage.query.filter_by(chat_id=chat.id).count()
                    
                    # Only delete chats with no messages or very few messages
                    if message_count <= 2:
                        success, error = delete_chat_session(chat.id, user_id)
                        if success:
                            cleaned_count += 1
                        else:
                            chat_logger.warning(f"Failed to delete inactive chat {chat.id}: {error}")
                            
                except Exception as e:
                    chat_logger.warning(f"Error processing inactive chat {chat.id}: {e}")
                    continue
            
            chat_logger.info(f"Cleaned up {cleaned_count} inactive chat sessions for user {user_id}")
            return cleaned_count, None
            
        except Exception as e:
            chat_logger.error(f"Error in cleanup_inactive_chats: {e}")
            return 0, str(e)

@handle_errors(default_return={})
def get_user_chat_summary(user_id):
    """Get comprehensive chat summary for a user"""
    with chat_error_handler("get_user_chat_summary"):
        if not user_id:
            return {}
        
        try:
            # Get all user chats
            all_chats = Chat.query.filter_by(user_id=user_id).all()
            
            if not all_chats:
                return {"total_chats": 0, "message": "No chat sessions found"}
            
            # Basic statistics
            total_chats = len(all_chats)
            active_chats = sum(1 for chat in all_chats 
                             if (ist_now() - chat.updated_at).days <= 7)
            
            # Chat type distribution
            chat_types = {"memory": 0, "collection": 0, "diary": 0}
            for chat in all_chats:
                if chat.memory_id:
                    chat_types["memory"] += 1
                elif chat.collection_id:
                    chat_types["collection"] += 1
                elif chat.diary_id:
                    chat_types["diary"] += 1
            
            # Message statistics  
            total_messages = sum(
                ChatMessage.query.filter_by(chat_id=chat.id).count() 
                for chat in all_chats
            )
            
            # Most active chat
            most_active_chat = None
            max_messages = 0
            
            for chat in all_chats:
                message_count = ChatMessage.query.filter_by(chat_id=chat.id).count()
                if message_count > max_messages:
                    max_messages = message_count
                    most_active_chat = {
                        "id": chat.id,
                        "title": chat.title,
                        "message_count": message_count,
                        "last_activity": chat.updated_at.isoformat()
                    }
            
            # Recent activity
            recent_chats = sorted(all_chats, key=lambda x: x.updated_at, reverse=True)[:5]
            recent_activity = [
                {
                    "id": chat.id,
                    "title": chat.title,
                    "last_activity": chat.updated_at.isoformat(),
                    "type": "memory" if chat.memory_id else "collection" if chat.collection_id else "diary"
                }
                for chat in recent_chats
            ]
            
            summary = {
                "total_chats": total_chats,
                "active_chats_last_week": active_chats,
                "total_messages": total_messages,
                "avg_messages_per_chat": round(total_messages / total_chats, 1) if total_chats > 0 else 0,
                "chat_type_distribution": chat_types,
                "most_active_chat": most_active_chat,
                "recent_activity": recent_activity,
                "generated_at": ist_now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            chat_logger.error(f"Error getting user chat summary: {e}")
            return {"error": str(e)}

# Enhanced search and discovery functions
@handle_errors(default_return=[])
def search_chat_history(user_id, search_query, limit=20):
    """Search across all user's chat messages for specific content"""
    with chat_error_handler("search_chat_history"):
        if not all([user_id, search_query]):
            return []
        
        if len(search_query.strip()) < 3:
            return []
        
        try:
            # Get user's chats
            user_chats = Chat.query.filter_by(user_id=user_id).all()
            chat_ids = [chat.id for chat in user_chats]
            
            if not chat_ids:
                return []
            
            # Search messages
            search_pattern = f"%{search_query.strip()}%"
            
            matching_messages = ChatMessage.query.filter(
                ChatMessage.chat_id.in_(chat_ids),
                ChatMessage.content.ilike(search_pattern)
            ).order_by(ChatMessage.timestamp.desc()).limit(limit).all()
            
            # Build results with context
            results = []
            for message in matching_messages:
                # Get chat info
                chat = next((c for c in user_chats if c.id == message.chat_id), None)
                if not chat:
                    continue
                
                # Get surrounding context (previous and next message)
                context_messages = ChatMessage.query.filter_by(chat_id=message.chat_id)\
                    .order_by(ChatMessage.timestamp)\
                    .filter(
                        ChatMessage.timestamp >= message.timestamp - timedelta(minutes=5),
                        ChatMessage.timestamp <= message.timestamp + timedelta(minutes=5)
                    ).all()
                
                results.append({
                    "message_id": message.id,
                    "chat_id": message.chat_id,
                    "chat_title": chat.title,
                    "content": message.content,
                    "is_user": message.is_user,
                    "timestamp": message.timestamp.isoformat(),
                    "context_messages": [
                        {
                            "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                            "is_user": msg.is_user,
                            "timestamp": msg.timestamp.isoformat()
                        }
                        for msg in context_messages if msg.id != message.id
                    ]
                })
            
            chat_logger.info(f"Found {len(results)} chat history matches for query '{search_query}'")
            return results
            
        except Exception as e:
            chat_logger.error(f"Error searching chat history: {e}")
            return []

# System health and monitoring functions
def get_chat_system_health():
    """Get comprehensive chat system health metrics"""
    try:
        health_data = {
            "status": "healthy",
            "checks": {},
            "statistics": {},
            "issues": [],
            "timestamp": ist_now().isoformat()
        }
        
        # Database connectivity check
        try:
            total_chats = Chat.query.count()
            total_messages = ChatMessage.query.count()
            health_data["checks"]["database"] = "ok"
            health_data["statistics"]["total_chats"] = total_chats
            health_data["statistics"]["total_messages"] = total_messages
        except Exception as e:
            health_data["checks"]["database"] = "error"
            health_data["issues"].append(f"Database connectivity: {str(e)}")
            health_data["status"] = "degraded"
        
        # Conversation context check
        try:
            # Test conversation context functionality
            test_session_id = "health_check_session"
            conversation_context.start_session(test_session_id)
            conversation_context.clear_session(test_session_id)
            health_data["checks"]["conversation_context"] = "ok"
        except Exception as e:
            health_data["checks"]["conversation_context"] = "error"
            health_data["issues"].append(f"Conversation context: {str(e)}")
            health_data["status"] = "degraded"
        
        # Analytics system check
        try:
            # Test analytics functionality
            test_analytics = chat_analytics.get_analytics("test_chat")
            health_data["checks"]["analytics"] = "ok"
        except Exception as e:
            health_data["checks"]["analytics"] = "error"
            health_data["issues"].append(f"Analytics system: {str(e)}")
            health_data["status"] = "degraded"
        
        # Enhanced services integration check
        try:
            # Test search suggestions
            suggestions = search_suggestions(1, "test", 1)
            health_data["checks"]["enhanced_services"] = "ok"
        except Exception as e:
            health_data["checks"]["enhanced_services"] = "error"
            health_data["issues"].append(f"Enhanced services integration: {str(e)}")
            health_data["status"] = "degraded"
        
        # Performance metrics
        try:
            # Recent chat activity
            recent_chats = Chat.query.filter(
                Chat.updated_at >= datetime.utcnow() - timedelta(hours=24)
            ).count()
            
            health_data["statistics"]["recent_chat_activity"] = recent_chats
            health_data["statistics"]["avg_response_time"] = "calculated_from_analytics"
            
        except Exception as e:
            health_data["issues"].append(f"Performance metrics: {str(e)}")
        
        # Set overall status
        if len(health_data["issues"]) == 0:
            health_data["status"] = "healthy"
        elif len(health_data["issues"]) <= 2:
            health_data["status"] = "degraded"
        else:
            health_data["status"] = "unhealthy"
        
        return health_data
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Export all enhanced functions with backward compatibility
__all__ = [
    # Core chat session functions
    'create_memory_chat_session',
    'create_chat_session', 
    'get_chat_sessions',
    'get_chat_messages',
    'add_message_to_chat',
    'delete_chat_session',
    
    # Query processing functions
    'process_chat_query',
    'process_memory_chat_query',
    
    # Diary chat functions
    'create_diary_chat_session',
    'get_diary_chat_sessions', 
    'process_diary_chat_query',
    
    # Utility functions
    'get_chat_suggestions',
    'get_chat_analytics',
    
    # Enhanced functions
    'cleanup_inactive_chats',
    'get_user_chat_summary',
    'search_chat_history',
    'get_chat_system_health'
]

# Initialize system logging
chat_logger.info("Enhanced chat services module loaded successfully")
chat_logger.info(f"Available functions: {len(__all__)}")
chat_logger.info("All critical issues have been addressed with comprehensive error handling")