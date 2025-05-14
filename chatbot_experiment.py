from openai import AzureOpenAI
import tiktoken
import json
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("api_key"),
    api_version=os.getenv("api_version"),
    azure_endpoint=os.getenv("azure_endpoint")
)

class Conversation:
    def __init__(self):
        self.history = []
        # Initialize tokenizer for GPT-3.5 Turbo
        self.tokenizer = tiktoken.encoding_for_model("gpt-35-turbo")
        self.load_history()
        
        # Configuration
        self.KEEP_RECENT_EXCHANGES = 4  # Always keep last 4 full exchanges (8 messages)
        self.COMPRESSION_STAGES = {
            'full': 1500,      # Stage 1: Keep everything (up to 1500 tokens)
            'partial': 2500,   # Stage 2: Keep questions + first line of answers (up to 2500 tokens)
            'questions': 3000  # Stage 3: Keep only questions (up to 3000 tokens)
        }
    
    def _count_tokens(self, message):
        """Count tokens for a single message"""
        return len(self.tokenizer.encode(message['content']))
    
    def _total_tokens(self):
        """Calculate total tokens in conversation history"""
        return sum(self._count_tokens(msg) for msg in self.history)
    
    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.history.append(message)
        self._truncate_history()
        self.save_history()
    
    def _compress_message(self, message, stage='full'):
        """Compress message based on stage"""
        if message['role'] == 'user':
            # Always keep user questions
            return message
        
        if stage == 'full':
            return message
        elif stage == 'partial':
            # Keep first line/sentence of response
            first_line = message['content'].split('\n')[0].split('. ')[0] + '...'
            return {"role": message['role'], "content": first_line}
        elif stage == 'questions':
            # Drop assistant responses entirely
            return None
    
    def _get_recent_messages_indices(self):
        """Get indices of messages that should be kept in full"""
        if len(self.history) <= self.KEEP_RECENT_EXCHANGES * 2:
            return range(len(self.history))
        return range(len(self.history) - (self.KEEP_RECENT_EXCHANGES * 2), len(self.history))
    
    def _truncate_history(self):
        """Progressive compression while preserving recent exchanges"""
        total_tokens = self._total_tokens()
        
        # If we're under the limit, no compression needed
        if total_tokens <= self.COMPRESSION_STAGES['full']:
            return
        
        # Get indices of recent messages to preserve
        preserve_indices = set(self._get_recent_messages_indices())
        compressed_history = []
        
        # Stage 1: Compress older messages to first line
        if total_tokens <= self.COMPRESSION_STAGES['partial']:
            for i, msg in enumerate(self.history):
                if i in preserve_indices:
                    compressed_history.append(msg)  # Keep recent messages in full
                else:
                    compressed = self._compress_message(msg, 'partial')
                    if compressed:
                        compressed_history.append(compressed)
        
        # Stage 2: Keep only questions for oldest messages
        elif total_tokens <= self.COMPRESSION_STAGES['questions']:
            for i, msg in enumerate(self.history):
                if i in preserve_indices:
                    compressed_history.append(msg)  # Keep recent messages in full
                else:
                    compressed = self._compress_message(msg, 'questions')
                    if compressed:
                        compressed_history.append(compressed)
        
        # Stage 3: If still too long, remove oldest messages while preserving recent ones
        else:
            while self._total_tokens() > self.COMPRESSION_STAGES['questions']:
                if len(self.history) > self.KEEP_RECENT_EXCHANGES * 2:
                    self.history.pop(0)  # Remove oldest message
                else:
                    break  # Don't remove any more if we're at our minimum preserved messages
        
        self.history = compressed_history
    
    def get_messages(self):
        # Always include the system message for consistent behavior
        system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
        return [system_message] + self.history
    
    def save_history(self):
        """Save conversation history to JSON file"""
        with open('conversation_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def load_history(self):
        """Load conversation history from JSON file if it exists"""
        try:
            with open('conversation_history.json', 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        except FileNotFoundError:
            self.history = []
    
    def print_conversation(self):
        print("\n=== Conversation History ===")
        total_tokens = 0
        preserve_indices = set(self._get_recent_messages_indices())
        
        for i, msg in enumerate(self.history):
            tokens = self._count_tokens(msg)
            total_tokens += tokens
            
            # Indicate preservation and compression state
            if i in preserve_indices:
                state = '[Preserved-Full]'
            elif '...' in msg.get('content', ''):
                state = '[Compressed]'
            else:
                state = '[Full]'
            
            print(f"\n{msg['role'].upper()}: {msg['content']}")
            print(f"[Tokens: {tokens}] {state}")
        
        print(f"\n=== Total Tokens: {total_tokens} ===")
        #print(f"=== Compression Stage: {self._get_current_stage()} ===")
        #print(f"=== Recent Exchanges Preserved: {self.KEEP_RECENT_EXCHANGES} ===")
        print("=========================")
"""
    def _get_current_stage(self):
        #Determine current compression stage
        tokens = self._total_tokens()
        if tokens <= self.COMPRESSION_STAGES['full']:
            return 'No Compression'
        elif tokens <= self.COMPRESSION_STAGES['partial']:
            return 'Partial Compression'
        elif tokens <= self.COMPRESSION_STAGES['questions']:
            return 'Questions Only'
        return 'Maximum Compression'
"""
def chat_with_gpt(prompt, conversation):
    # Print current conversation state and new prompt
    conversation.print_conversation()
    print(f"\nNew Prompt: {prompt}\n")
    
    try:
        # Add user's message to conversation
        conversation.add_message("user", prompt)
        
        # Get response from GPT
        response = client.chat.completions.create(
            model="gpt-35-turbo-2",
            messages=conversation.get_messages(),
            temperature=0.7,
            max_tokens=1000
        )
        
        assistant_response = response.choices[0].message.content
        # Add assistant's response to conversation
        conversation.add_message("assistant", assistant_response)
        
        print(f"Assistant's Response: {assistant_response}\n")
        return assistant_response
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Clear existing conversation history file
    """
    if os.path.exists('conversation_history.json'):
        with open('conversation_history.json', 'w', encoding='utf-8') as f:
            json.dump([], f)
    """
    conversation = Conversation()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        chat_with_gpt(user_input, conversation)