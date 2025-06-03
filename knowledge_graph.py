# knowledge_graph.py
import networkx as nx
from datetime import datetime
import ollama
import json

class ConversationContext:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_session_id = None
    
    def start_session(self, chat_id):
        """Start or resume a conversation session"""
        self.current_session_id = chat_id
        if not self.graph.has_node(chat_id):
            self.graph.add_node(chat_id, type='session', messages=[])
    
    def add_message(self, message_id, content, is_user, timestamp, related_entities=None):
        """Add a message to the conversation graph"""
        if not self.current_session_id:
            raise ValueError("No active session")
        
        # Add message node
        self.graph.add_node(
            message_id, 
            type='message', 
            content=content, 
            is_user=is_user, 
            timestamp=timestamp
        )
        
        # Connect to session
        self.graph.add_edge(self.current_session_id, message_id, type='contains')
        
        # Get session node and append message to its message list
        session_data = self.graph.nodes[self.current_session_id]
        session_data['messages'].append(message_id)
        
        # Connect to related entities (memories, concepts, etc)
        if related_entities:
            for entity_id, entity_type in related_entities:
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id, type=entity_type)
                self.graph.add_edge(message_id, entity_id, type='references')
    
    def get_conversation_history(self, limit=5):
        """Get recent conversation history for the current session"""
        if not self.current_session_id:
            return []
        
        session_data = self.graph.nodes[self.current_session_id]
        message_ids = session_data['messages'][-limit:]
        
        history = []
        for msg_id in message_ids:
            msg_data = self.graph.nodes[msg_id]
            history.append({
                'id': msg_id,
                'content': msg_data['content'],
                'is_user': msg_data['is_user'],
                'timestamp': msg_data['timestamp']
            })
        
        return history
    
    def analyze_question_context(self, question):
        """Analyze a question in context of conversation history"""
        history = self.get_conversation_history(limit=3)
        
        # If no history or just one message, no context to analyze
        if len(history) <= 1:
            return {
                "is_followup": False,
                "referenced_entities": [],
                "expanded_question": question
            }
        
        # Create a context analysis prompt
        history_text = "\n".join([
            f"{'User' if msg['is_user'] else 'Assistant'}: {msg['content']}"
            for msg in history
        ])
        
        prompt = f"""
        Review this conversation history {history_text}, New question: {question}
        """
        
        # Use LLM to analyze the question context
        try:
            response = ollama.generate(
                model="gemma3",
                prompt=prompt
            )
            
            # Parse the response
            response_text = response['response']
            # Find JSON in the response (assuming model might wrap it in markdown)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    analysis = json.loads(json_str)
                    return analysis
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {json_str}")
                    # Fallback if JSON parsing fails
                    pass
        
        except Exception as e:
            print(f"Error analyzing question context: {e}")
        
        # Fallback if anything fails
        return {
            "is_followup": False,
            "referenced_entities": [],
            "expanded_question": question
        }