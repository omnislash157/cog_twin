"""
Chat Parser Factory - Multi-Provider Chat Log Import
=====================================================

Drop your AI chat logs. Get instant memory.

Supported Providers:
- Anthropic Claude (conversations.json)
- OpenAI ChatGPT (conversations.json from export)
- Grok (X/Twitter AI export)
- Google Gemini (takeout export)

Auto-detection: Factory sniffs file format and routes to correct parser.

Usage:
    factory = ChatParserFactory()
    conversations = factory.parse("my_export.json")  # Auto-detects
    
    # Or explicit:
    conversations = factory.parse("export.json", provider="openai")

Output Format (Normalized):
    {
        'id': str,
        'title': str,
        'created_at': str,
        'updated_at': str,
        'messages': [
            {'id': str, 'role': str, 'content': str, 'created_at': str}
        ],
        'metadata': {
            'source': str,  # 'anthropic', 'openai', 'grok', 'gemini'
            'message_count': int,
            ...provider-specific fields
        }
    }

Version: 14.0.0
"""

import json
import zipfile
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from abc import ABC, abstractmethod

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    ijson = None
    IJSON_AVAILABLE = False


# ============================================================================
# BASE PARSER (Abstract)
# ============================================================================

class BaseParser(ABC):
    """Base class for all chat log parsers."""
    
    PROVIDER: str = "unknown"
    
    def __init__(self, large_file_threshold_mb: int = 100):
        self.large_file_threshold_mb = large_file_threshold_mb
        self.stats = {
            'total_conversations': 0,
            'empty_conversations': 0,
            'total_messages': 0,
            'parse_errors': 0,
            'used_streaming': False
        }
    
    @abstractmethod
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse file and return normalized conversations."""
        pass
    
    @abstractmethod
    def can_parse(self, data: Any) -> bool:
        """Check if this parser can handle the given data structure."""
        pass
    
    def sample_conversations(
        self, 
        conversations: List[Dict[str, Any]], 
        n: int = 0
    ) -> List[Dict[str, Any]]:
        """Randomly sample N conversations. n=0 means all."""
        if n <= 0 or len(conversations) <= n:
            return conversations
        
        shuffled = conversations.copy()
        random.shuffle(shuffled)
        return shuffled[:n]
    
    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."
    
    @staticmethod
    def _safe_timestamp(ts: Any) -> str:
        """Safely convert timestamp to ISO string."""
        if not ts:
            return ""
        if isinstance(ts, (int, float)):
            try:
                return datetime.fromtimestamp(ts).isoformat()
            except (ValueError, OSError):
                return ""
        return str(ts)


# ============================================================================
# ANTHROPIC PARSER
# ============================================================================

class AnthropicParser(BaseParser):
    """
    Parser for Anthropic Claude exports.
    
    Export format: conversations.json (array of conversations)
    Get it: claude.ai -> Settings -> Export Data
    """
    
    PROVIDER = "anthropic"
    
    def can_parse(self, data: Any) -> bool:
        """Anthropic exports have 'chat_messages' in conversations."""
        if not isinstance(data, list) or not data:
            return False
        first = data[0]
        return isinstance(first, dict) and 'chat_messages' in first
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse Anthropic conversations.json."""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        if not isinstance(raw, list):
            raise ValueError("Expected list of conversations")
        
        self.stats['total_conversations'] = len(raw)
        normalized = []
        
        for conv in raw:
            try:
                result = self._normalize(conv)
                if result:
                    normalized.append(result)
            except Exception as e:
                self.stats['parse_errors'] += 1
        
        return normalized
    
    def _normalize(self, conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Anthropic conversation."""
        messages = conv.get('chat_messages', [])
        
        if not messages:
            self.stats['empty_conversations'] += 1
            return None
        
        self.stats['total_messages'] += len(messages)
        
        # Title from name or first message
        title = conv.get('name', '').strip()
        if not title:
            first_human = next((m for m in messages if m.get('sender') == 'human'), None)
            if first_human:
                title = self._truncate(first_human.get('text', ''), 60)
            else:
                title = f"Conversation {conv.get('uuid', 'unknown')[:8]}"
        
        return {
            'id': conv.get('uuid', ''),
            'title': title,
            'created_at': conv.get('created_at', ''),
            'updated_at': conv.get('updated_at', ''),
            'messages': [
                {
                    'id': m.get('uuid', ''),
                    'role': m.get('sender', 'unknown'),
                    'content': m.get('text', ''),
                    'created_at': m.get('created_at', '')
                }
                for m in messages
            ],
            'metadata': {
                'source': 'anthropic',
                'message_count': len(messages),
                'account_uuid': conv.get('account', {}).get('uuid', '')
            }
        }


# ============================================================================
# OPENAI PARSER
# ============================================================================

class OpenAIParser(BaseParser):
    """
    Parser for OpenAI ChatGPT exports.
    
    Export format: conversations.json (array with 'mapping' structure)
    Get it: chatgpt.com -> Settings -> Data Controls -> Export
    
    OpenAI structure is complex - conversations use a tree/mapping structure
    where messages reference parent IDs.
    """
    
    PROVIDER = "openai"
    
    def can_parse(self, data: Any) -> bool:
        """OpenAI exports have 'mapping' in conversations."""
        if not isinstance(data, list) or not data:
            return False
        first = data[0]
        return isinstance(first, dict) and 'mapping' in first
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse OpenAI conversations.json."""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        if not isinstance(raw, list):
            raise ValueError("Expected list of conversations")
        
        self.stats['total_conversations'] = len(raw)
        normalized = []
        
        for conv in raw:
            try:
                result = self._normalize(conv)
                if result:
                    normalized.append(result)
            except Exception as e:
                self.stats['parse_errors'] += 1
        
        return normalized
    
    def _normalize(self, conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize OpenAI conversation with tree traversal."""
        mapping = conv.get('mapping', {})
        
        if not mapping:
            self.stats['empty_conversations'] += 1
            return None
        
        # Extract messages from mapping tree
        messages = self._extract_messages_from_mapping(mapping)
        
        if not messages:
            self.stats['empty_conversations'] += 1
            return None
        
        self.stats['total_messages'] += len(messages)
        
        title = conv.get('title', '').strip()
        if not title:
            first_user = next((m for m in messages if m['role'] == 'user'), None)
            if first_user:
                title = self._truncate(first_user['content'], 60)
            else:
                title = f"Chat {conv.get('id', 'unknown')[:8]}"
        
        return {
            'id': conv.get('id', ''),
            'title': title,
            'created_at': self._safe_timestamp(conv.get('create_time')),
            'updated_at': self._safe_timestamp(conv.get('update_time')),
            'messages': messages,
            'metadata': {
                'source': 'openai',
                'message_count': len(messages),
                'model': self._extract_model(mapping),
                'is_archived': conv.get('is_archived', False)
            }
        }
    
    def _extract_messages_from_mapping(self, mapping: Dict) -> List[Dict[str, Any]]:
        """
        Extract ordered messages from OpenAI's tree mapping.
        
        OpenAI stores messages in a tree where each node has:
        - id: node ID
        - parent: parent node ID
        - message: the actual message content (if any)
        """
        messages = []
        
        # Find root node (parent is None or missing)
        nodes = list(mapping.values())
        
        # Build parent->children map
        children_map: Dict[str, List[str]] = {}
        for node_id, node in mapping.items():
            parent_id = node.get('parent')
            if parent_id:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(node_id)
        
        # Find root (no parent or parent not in mapping)
        root_id = None
        for node_id, node in mapping.items():
            parent = node.get('parent')
            if not parent or parent not in mapping:
                root_id = node_id
                break
        
        if not root_id:
            return messages
        
        # DFS traversal to get messages in order
        def traverse(node_id: str):
            node = mapping.get(node_id, {})
            msg = node.get('message')
            
            if msg and msg.get('content'):
                content = msg['content']
                role = msg.get('author', {}).get('role', 'unknown')
                
                # Skip system messages
                if role not in ['system']:
                    # Extract text from content parts
                    text = self._extract_content_text(content)
                    if text:
                        messages.append({
                            'id': msg.get('id', node_id),
                            'role': 'human' if role == 'user' else 'assistant',
                            'content': text,
                            'created_at': self._safe_timestamp(msg.get('create_time'))
                        })
            
            # Traverse children
            for child_id in children_map.get(node_id, []):
                traverse(child_id)
        
        traverse(root_id)
        return messages
    
    def _extract_content_text(self, content: Any) -> str:
        """Extract text from OpenAI content structure."""
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            parts = content.get('parts', [])
            if parts:
                return '\n'.join(str(p) for p in parts if isinstance(p, str))
        
        return ''
    
    def _extract_model(self, mapping: Dict) -> str:
        """Extract model name from mapping."""
        for node in mapping.values():
            msg = node.get('message')
            if msg:  # Some nodes have message: None
                metadata = msg.get('metadata', {})
                model = metadata.get('model_slug', '')
                if model:
                    return model
        return 'unknown'


# ============================================================================
# GROK PARSER
# ============================================================================

class GrokParser(BaseParser):
    """
    Parser for Grok (X/Twitter AI) exports.
    
    Export format: prod-grok-backend.json from X data export
    Get it: X.com -> Settings -> Your Account -> Download Archive
    
    Structure:
    {
      "conversations": [
        {
          "conversation": {"id": "...", "title": "...", "create_time": "..."},
          "responses": [
            {"response": {"_id": "...", "message": "...", "sender": "human|assistant", 
                          "create_time": {"$date": {"$numberLong": "..."}}}}
          ]
        }
      ]
    }
    """
    
    PROVIDER = "grok"
    
    def can_parse(self, data: Any) -> bool:
        """Detect Grok export format."""
        if isinstance(data, dict):
            convs = data.get('conversations', [])
            if convs and isinstance(convs, list) and len(convs) > 0:
                first = convs[0]
                # New format: {"conversation": {...}, "responses": [...]}
                if isinstance(first, dict) and 'conversation' in first and 'responses' in first:
                    return True
        # Legacy format fallback
        if isinstance(data, list) and data:
            first = data[0]
            return isinstance(first, dict) and ('conversationId' in first or 'grok' in str(first).lower())
        return False
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse Grok conversation export."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.zip':
            return self._parse_from_zip(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        # New format: {"conversations": [{...}, ...]}
        if isinstance(raw, dict) and 'conversations' in raw:
            conversations = raw['conversations']
        elif isinstance(raw, list):
            conversations = raw
        else:
            raise ValueError("Expected object with 'conversations' or array")
        
        self.stats['total_conversations'] = len(conversations)
        normalized = []
        
        for conv in conversations:
            try:
                result = self._normalize(conv)
                if result:
                    normalized.append(result)
            except Exception as e:
                self.stats['parse_errors'] += 1
        
        return normalized
    
    def _parse_from_zip(self, zip_path: Path) -> List[Dict[str, Any]]:
        """Extract and parse Grok data from Twitter archive ZIP."""
        normalized = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            grok_files = [
                n for n in zf.namelist() 
                if 'grok' in n.lower() and n.endswith('.json')
            ]
            
            for filename in grok_files:
                with zf.open(filename) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'conversations' in data:
                        for conv in data['conversations']:
                            result = self._normalize(conv)
                            if result:
                                normalized.append(result)
                    elif isinstance(data, list):
                        for conv in data:
                            result = self._normalize(conv)
                            if result:
                                normalized.append(result)
        
        self.stats['total_conversations'] = len(normalized)
        return normalized
    
    def _parse_mongo_timestamp(self, ts: Any) -> str:
        """Parse MongoDB-style timestamp: {"$date": {"$numberLong": "1764586228275"}}"""
        if not ts:
            return ""
        if isinstance(ts, str):
            return ts
        if isinstance(ts, dict):
            date_obj = ts.get('$date', {})
            if isinstance(date_obj, dict):
                ms = date_obj.get('$numberLong', '')
                if ms:
                    try:
                        return datetime.fromtimestamp(int(ms) / 1000).isoformat()
                    except (ValueError, OSError):
                        return ""
            elif isinstance(date_obj, (int, float)):
                try:
                    return datetime.fromtimestamp(date_obj / 1000).isoformat()
                except (ValueError, OSError):
                    return ""
        return self._safe_timestamp(ts)
    
    def _normalize(self, conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Grok conversation."""
        # New format: {"conversation": {...}, "responses": [...]}
        if 'conversation' in conv and 'responses' in conv:
            meta = conv['conversation']
            responses = conv['responses']
            
            messages = []
            for r in responses:
                resp = r.get('response', r)
                msg_content = resp.get('message', '')
                if not msg_content:
                    continue
                messages.append({
                    'id': resp.get('_id', resp.get('id', '')),
                    'role': 'human' if resp.get('sender') == 'human' else 'assistant',
                    'content': msg_content,
                    'created_at': self._parse_mongo_timestamp(resp.get('create_time'))
                })
            
            if not messages:
                self.stats['empty_conversations'] += 1
                return None
            
            self.stats['total_messages'] += len(messages)
            
            title = meta.get('title', '')
            if not title:
                first_human = next((m for m in messages if m['role'] == 'human'), None)
                if first_human:
                    title = self._truncate(first_human['content'], 60)
                else:
                    title = f"Grok Chat {meta.get('id', 'unknown')[:8]}"
            
            return {
                'id': meta.get('id', ''),
                'title': title,
                'created_at': meta.get('create_time', ''),
                'updated_at': meta.get('modify_time', ''),
                'messages': messages,
                'metadata': {
                    'source': 'grok',
                    'message_count': len(messages),
                    'user_id': meta.get('user_id', ''),
                    'model': messages[-1].get('model', '') if messages else ''
                }
            }
        
        # Legacy format fallback
        messages = conv.get('messages', [])
        if not messages:
            self.stats['empty_conversations'] += 1
            return None
        
        self.stats['total_messages'] += len(messages)
        
        first_user = next(
            (m for m in messages if m.get('role') == 'user' or m.get('sender') == 'user'), 
            None
        )
        title = self._truncate(
            first_user.get('content', first_user.get('text', ''))[:60] if first_user else 'Grok Chat',
            60
        )
        
        return {
            'id': conv.get('conversationId', conv.get('id', '')),
            'title': title,
            'created_at': self._safe_timestamp(conv.get('createdAt', conv.get('created_at'))),
            'updated_at': self._safe_timestamp(conv.get('updatedAt', conv.get('updated_at'))),
            'messages': [
                {
                    'id': m.get('messageId', m.get('id', '')),
                    'role': 'human' if m.get('role', m.get('sender')) == 'user' else 'assistant',
                    'content': m.get('content', m.get('text', '')),
                    'created_at': self._safe_timestamp(m.get('createdAt', m.get('created_at')))
                }
                for m in messages
            ],
            'metadata': {
                'source': 'grok',
                'message_count': len(messages)
            }
        }


# ============================================================================
# GEMINI PARSER
# ============================================================================

class GeminiParser(BaseParser):
    """
    Parser for Google Gemini exports.
    
    Export format: Google Takeout -> Gemini Apps
    Get it: takeout.google.com -> Select Gemini
    
    Gemini exports come as a folder with individual JSON files per conversation,
    or as a bundled export.
    """
    
    PROVIDER = "gemini"
    
    def can_parse(self, data: Any) -> bool:
        """Gemini exports have Google-specific structure."""
        if not isinstance(data, list) or not data:
            # Could be single conversation
            if isinstance(data, dict):
                return 'geminiConversation' in data or 'bard' in str(data).lower()
            return False
        first = data[0]
        return isinstance(first, dict) and (
            'geminiConversation' in first or
            'conversationTurns' in first or
            'bardResponse' in str(first).lower()
        )
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse Gemini export (file or directory)."""
        filepath = Path(filepath)
        
        # Handle directory of conversation files
        if filepath.is_dir():
            return self._parse_directory(filepath)
        
        # Handle ZIP archive
        if filepath.suffix == '.zip':
            return self._parse_from_zip(filepath)
        
        # Single file
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        if isinstance(raw, dict):
            # Single conversation
            result = self._normalize(raw)
            return [result] if result else []
        
        if not isinstance(raw, list):
            raise ValueError("Expected list or dict")
        
        self.stats['total_conversations'] = len(raw)
        normalized = []
        
        for conv in raw:
            try:
                result = self._normalize(conv)
                if result:
                    normalized.append(result)
            except Exception as e:
                self.stats['parse_errors'] += 1
        
        return normalized
    
    def _parse_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Parse directory of Gemini conversation files."""
        normalized = []
        
        for json_file in dir_path.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                result = self._normalize(data)
                if result:
                    normalized.append(result)
            except Exception:
                self.stats['parse_errors'] += 1
        
        self.stats['total_conversations'] = len(normalized)
        return normalized
    
    def _parse_from_zip(self, zip_path: Path) -> List[Dict[str, Any]]:
        """Extract and parse from Takeout ZIP."""
        normalized = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            json_files = [n for n in zf.namelist() if n.endswith('.json')]
            
            for filename in json_files:
                try:
                    with zf.open(filename) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for conv in data:
                                result = self._normalize(conv)
                                if result:
                                    normalized.append(result)
                        else:
                            result = self._normalize(data)
                            if result:
                                normalized.append(result)
                except Exception:
                    self.stats['parse_errors'] += 1
        
        self.stats['total_conversations'] = len(normalized)
        return normalized
    
    def _normalize(self, conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Gemini conversation."""
        # Handle different Gemini export structures
        turns = (
            conv.get('conversationTurns') or 
            conv.get('turns') or 
            conv.get('messages', [])
        )
        
        if not turns:
            self.stats['empty_conversations'] += 1
            return None
        
        messages = []
        for turn in turns:
            # Extract user input
            user_input = turn.get('userInput', turn.get('user', {}))
            if user_input:
                text = user_input.get('text', user_input.get('content', ''))
                if text:
                    messages.append({
                        'id': turn.get('turnId', ''),
                        'role': 'human',
                        'content': text,
                        'created_at': self._safe_timestamp(turn.get('createTime'))
                    })
            
            # Extract model response
            model_response = turn.get('modelResponse', turn.get('response', {}))
            if model_response:
                text = model_response.get('text', model_response.get('content', ''))
                if text:
                    messages.append({
                        'id': turn.get('turnId', '') + '_response',
                        'role': 'assistant',
                        'content': text,
                        'created_at': self._safe_timestamp(turn.get('createTime'))
                    })
        
        if not messages:
            self.stats['empty_conversations'] += 1
            return None
        
        self.stats['total_messages'] += len(messages)
        
        title = conv.get('title', '')
        if not title and messages:
            title = self._truncate(messages[0]['content'], 60)
        
        return {
            'id': conv.get('conversationId', conv.get('id', '')),
            'title': title,
            'created_at': self._safe_timestamp(conv.get('createTime')),
            'updated_at': self._safe_timestamp(conv.get('updateTime')),
            'messages': messages,
            'metadata': {
                'source': 'gemini',
                'message_count': len(messages)
            }
        }


# ============================================================================
# FACTORY
# ============================================================================

ProviderType = Literal['anthropic', 'openai', 'grok', 'gemini', 'auto']

class ChatParserFactory:
    """
    Factory for parsing chat logs from multiple providers.
    
    Auto-detects format or accepts explicit provider hint.
    
    Usage:
        factory = ChatParserFactory()
        
        # Auto-detect
        convs = factory.parse("my_export.json")
        
        # Explicit provider
        convs = factory.parse("chatgpt.json", provider="openai")
        
        # From ZIP archive
        convs = factory.parse("twitter_archive.zip")
    """
    
    PARSERS = {
        'anthropic': AnthropicParser,
        'openai': OpenAIParser,
        'grok': GrokParser,
        'gemini': GeminiParser
    }
    
    def __init__(self):
        self.last_provider: Optional[str] = None
        self.last_stats: Dict[str, Any] = {}
    
    def parse(
        self, 
        filepath: str, 
        provider: ProviderType = 'auto',
        sample_n: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Parse chat log file with auto-detection or explicit provider.
        
        Args:
            filepath: Path to export file (JSON or ZIP)
            provider: 'auto' to detect, or explicit provider name
            sample_n: Random sample size (0 = all)
        
        Returns:
            List of normalized conversation dicts
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Select parser
        if provider == 'auto':
            parser = self._detect_parser(filepath)
        else:
            if provider not in self.PARSERS:
                raise ValueError(f"Unknown provider: {provider}. Options: {list(self.PARSERS.keys())}")
            parser = self.PARSERS[provider]()
        
        self.last_provider = parser.PROVIDER
        
        # Parse
        print(f"Parsing with {parser.PROVIDER} parser...")
        conversations = parser.parse_file(str(filepath))
        
        self.last_stats = parser.stats
        
        # Sample if requested
        if sample_n > 0:
            conversations = parser.sample_conversations(conversations, sample_n)
        
        print(f"Parsed {len(conversations)} conversations from {parser.PROVIDER}")
        
        return conversations
    
    def _detect_parser(self, filepath: Path) -> BaseParser:
        """Auto-detect correct parser based on file contents."""
        print(f"Auto-detecting format for {filepath.name}...")
        
        # Handle ZIP files
        if filepath.suffix == '.zip':
            return self._detect_from_zip(filepath)
        
        # Load and sniff JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read just enough to detect
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        # Try each parser
        for name, parser_class in self.PARSERS.items():
            parser = parser_class()
            if parser.can_parse(data):
                print(f"Detected: {name}")
                return parser
        
        raise ValueError(
            "Could not detect chat export format. "
            "Supported: Anthropic, OpenAI, Grok, Gemini. "
            "Try specifying provider explicitly."
        )
    
    def _detect_from_zip(self, zip_path: Path) -> BaseParser:
        """Detect parser from ZIP archive contents."""
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            names_lower = [n.lower() for n in names]
            
            # Check for provider-specific markers
            if any('grok' in n for n in names_lower):
                print("Detected: grok (from ZIP)")
                return GrokParser()
            
            if any('gemini' in n or 'bard' in n for n in names_lower):
                print("Detected: gemini (from ZIP)")
                return GeminiParser()
            
            # Try to find and sniff a JSON file
            json_files = [n for n in names if n.endswith('.json')]
            for jf in json_files[:5]:  # Check first few
                try:
                    with zf.open(jf) as f:
                        data = json.load(f)
                        for name, parser_class in self.PARSERS.items():
                            parser = parser_class()
                            if parser.can_parse(data):
                                print(f"Detected: {name} (from ZIP)")
                                return parser
                except Exception:
                    continue
        
        raise ValueError("Could not detect format from ZIP contents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics from last parse operation."""
        return {
            'provider': self.last_provider,
            **self.last_stats
        }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Chat Parser Factory - Multi-Provider Import")
        print("=" * 50)
        print("\nUsage:")
        print("  python chat_parser_factory.py <export_file> [provider]")
        print("\nProviders: anthropic, openai, grok, gemini, auto (default)")
        print("\nExamples:")
        print("  python chat_parser_factory.py conversations.json")
        print("  python chat_parser_factory.py chatgpt_export.json openai")
        print("  python chat_parser_factory.py twitter_archive.zip grok")
        sys.exit(0)
    
    filepath = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    
    factory = ChatParserFactory()
    
    try:
        conversations = factory.parse(filepath, provider=provider)
        stats = factory.get_stats()
        
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Provider: {stats['provider']}")
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Empty (skipped): {stats['empty_conversations']}")
        print(f"Parse errors: {stats['parse_errors']}")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Valid conversations: {len(conversations)}")
        
        # Show sample
        if conversations:
            print("\n" + "-" * 50)
            print("SAMPLE (first 5)")
            print("-" * 50)
            for conv in conversations[:5]:
                print(f"\n{conv['title']}")
                print(f"  ID: {conv['id'][:20]}...")
                print(f"  Messages: {len(conv['messages'])}")
                print(f"  Source: {conv['metadata']['source']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)