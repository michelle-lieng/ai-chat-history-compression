# AI Chat History Manager

A Python-based chat application that manages conversations with OpenAI's GPT model while implementing intelligent conversation history compression.

## Features

- Real-time chat with GPT-3.5 Turbo
- Smart conversation history management
- Progressive message compression based on token count
- Persistent conversation storage
- Token-aware conversation pruning
- Environment variable configuration

## Setup

1. Create a `.env` file with your Azure OpenAI credentials:

```
api_key="YOUR_API_KEY"
api_version="YOUR_API_VERSION"
azure_endpoint="YOUR_AZURE_ENDPOINT"
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main chat application:

```bash
python chatbot_experiment.py
```

Or use the simplified version:

```bash
python cleaned_output.py
```

## Conversation Management

The system implements three compression stages:

- **Full (≤1500 tokens)**: Keeps all messages intact
- **Partial (≤2500 tokens)**: Compresses older messages to first line
- **Questions (≤3000 tokens)**: Keeps only user questions for oldest messages

The most recent 4 exchanges are always preserved in full.

## Features In Detail

### Message Compression

- Automatic compression of older messages
- Preservation of recent conversation context
- Token-aware compression stages

### History Management

- JSON-based conversation persistence
- Automatic loading of previous conversations
- Smart truncation of conversation history

### Token Management

- Real-time token counting
- Progressive compression based on token thresholds
- Preservation of conversation context

## Exit Commands

Type any of these to exit the chat:

- `exit`
- `quit`
- `bye`
