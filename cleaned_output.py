from openai import AzureOpenAI
import tiktoken

client = AzureOpenAI(
    api_key="A427lLbgiTa75sIvy4Sxm2NabEAlfwHCt6jFkGkbNvdvzIPcMaRMJQQJ99BAAC5T7U2XJ3w3AAAAACOGFUPy",
    api_version="2024-08-01-preview",
    azure_endpoint="https://miche-m6eqvp0j-francecentral.cognitiveservices.azure.com"
)

class Conversation:
    def __init__(self):
        self.history = []
        # Initialize tokenizer for GPT-3.5 Turbo
        self.tokenizer = tiktoken.encoding_for_model("gpt-35-turbo")
        
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

def chat_with_gpt(prompt, conversation):    
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
    conversation = Conversation()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        chat_with_gpt(user_input, conversation)