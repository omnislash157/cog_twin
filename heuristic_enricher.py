"""
heuristic_enricher.py - Fast Signal Extraction for Pre-Computed Filtering
==========================================================================

Computes rich fingerprints at extraction time so retrieval becomes
pure filtering on pre-computed indexes. Zero LLM cost.

Signals extracted:
- intent_type: question/request/statement/complaint/celebration
- complexity: simple/moderate/complex (based on structure)
- domain_signals: 12 domain buckets with confidence scores
- technical_depth: 0-10 (code density, tech term ratio)
- emotional_valence: positive/neutral/negative
- action_required: bool (detects follow-up needs)
- topic_fingerprint: 64-bit hash for fast similarity
- entity_hints: pre-extracted entities for filtering
- conversation_mode: chat/debug/brainstorm/review
- urgency: low/medium/high

Usage:
    enricher = HeuristicEnricher()
    signals = enricher.extract_all(content, metadata)
    # signals is a flat dict - merge into node for indexing

Version: 13.0.0
"""

import re
import hashlib
from typing import Dict, Any, List, Set, Tuple
from collections import Counter


class HeuristicEnricher:
    """
    Fast signal extraction for pre-computed filtering.

    All methods are pure functions - no state, no I/O, no LLM calls.
    Designed to run in <1ms per node.
    """

    # Track which new domains have been reported (class-level)
    _reported_domains = set()

    # Domain patterns with weighted keywords
    DOMAIN_PATTERNS = {
        'code': {
            'strong': ['def ', 'class ', 'import ', 'function', '```', 'async ', 'await '],
            'weak': ['code', 'bug', 'error', 'debug', 'syntax', 'compile', 'runtime']
        },
        'data': {
            'strong': ['sql', 'query', 'database', 'postgresql', 'mongodb', 'redis'],
            'weak': ['data', 'table', 'schema', 'index', 'insert', 'select']
        },
        'infra': {
            'strong': ['docker', 'kubernetes', 'k8s', 'nginx', 'terraform', 'aws', 'gcp'],
            'weak': ['deploy', 'server', 'container', 'cluster', 'pod', 'node']
        },
        'ml': {
            'strong': ['embedding', 'vector', 'transformer', 'llm', 'gpt', 'claude', 'model'],
            'weak': ['train', 'inference', 'batch', 'epoch', 'loss', 'accuracy']
        },
        'architecture': {
            'strong': ['service', 'registry', 'dependency injection', 'singleton', 'factory'],
            'weak': ['pattern', 'design', 'layer', 'module', 'interface', 'abstract']
        },
        'testing': {
            'strong': ['pytest', 'unittest', 'mock', 'fixture', 'assert'],
            'weak': ['test', 'spec', 'coverage', 'integration', 'unit']
        },
        'config': {
            'strong': ['yaml', 'config', 'env', '.env', 'settings', 'namespace'],
            'weak': ['parameter', 'option', 'flag', 'default', 'override']
        },
        'frontend': {
            'strong': ['react', 'vue', 'svelte', 'css', 'html', 'dom'],
            'weak': ['component', 'render', 'state', 'prop', 'hook']
        },
        'api': {
            'strong': ['endpoint', 'rest', 'graphql', 'grpc', 'websocket'],
            'weak': ['request', 'response', 'route', 'middleware', 'handler']
        },
        'security': {
            'strong': ['auth', 'jwt', 'oauth', 'encryption', 'pii', 'guardian'],
            'weak': ['token', 'secret', 'permission', 'role', 'access']
        },
        'ops': {
            'strong': ['prometheus', 'grafana', 'logging', 'metrics', 'observability'],
            'weak': ['monitor', 'alert', 'trace', 'span', 'dashboard']
        },
        'business': {
            'strong': ['revenue', 'client', 'stakeholder', 'roi', 'budget'],
            'weak': ['project', 'deadline', 'meeting', 'goal', 'strategy']
        },
        'exercise': {
            'strong': ['workout', 'gym', 'running', 'lifting', 'cardio', 'training', 'fitness'],
            'weak': ['exercise', 'reps', 'sets', 'weights', 'marathon', 'stretch', 'recovery']
        },
        'spiritual': {
            'strong': ['meditation', 'prayer', 'mindfulness', 'spiritual', 'enlightenment', 'consciousness'],
            'weak': ['faith', 'belief', 'soul', 'divine', 'sacred', 'practice', 'ritual']
        },
        'physical_crafts': {
            'strong': ['woodworking', 'metalworking', 'pottery', 'welding', 'carpentry', 'crafting'],
            'weak': ['build', 'handmade', 'tools', 'workshop', 'forge', 'lathe', 'materials']
        },
        'philosophy': {
            'strong': ['philosophy', 'epistemology', 'metaphysics', 'ethics', 'existential', 'stoicism'],
            'weak': ['meaning', 'truth', 'reality', 'consciousness', 'moral', 'virtue', 'wisdom']
        },
        'relationship': {
            'strong': ['relationship', 'partner', 'dating', 'marriage', 'friendship', 'intimacy'],
            'weak': ['love', 'connection', 'communication', 'trust', 'conflict', 'boundary']
        },
        'creative': {
            'strong': ['art', 'music', 'writing', 'painting', 'drawing', 'composing', 'creative'],
            'weak': ['design', 'inspiration', 'aesthetic', 'artist', 'craft', 'expression']
        },
        'family': {
            'strong': ['family', 'parent', 'parenting', 'children', 'kids', 'marriage', 'sibling'],
            'weak': ['mom', 'dad', 'son', 'daughter', 'child', 'household', 'relatives']
        },
        'psychology': {
            'strong': ['psychology', 'therapy', 'cognitive', 'behavioral', 'mental health', 'anxiety', 'depression'],
            'weak': ['emotion', 'mindset', 'trauma', 'healing', 'growth', 'pattern', 'unconscious']
        },
        'physics': {
            'strong': ['physics', 'quantum', 'relativity', 'thermodynamics', 'mechanics', 'gravity'],
            'weak': ['energy', 'force', 'mass', 'velocity', 'acceleration', 'particle', 'wave']
        },
        'hobbies': {
            'strong': ['hobby', 'collection', 'gaming', 'photography', 'gardening', 'cooking'],
            'weak': ['leisure', 'pastime', 'interest', 'recreational', 'enjoy', 'fun']
        },
        'employer': {
            'strong': ['employer', 'workplace', 'manager', 'hr', 'corporate', 'company policy'],
            'weak': ['job', 'work', 'boss', 'team', 'office', 'career', 'promotion']
        }
    }
    
    # Intent patterns
    INTENT_PATTERNS = {
        'question': [r'\?$', r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does)\b'],
        'request': [r'\b(please|help|need|want|create|make|write|build|fix|implement)\b'],
        'complaint': [r'\b(broken|wrong|bad|hate|annoying|frustrated|stuck|failing)\b'],
        'celebration': [r'\b(works|working|solved|fixed|success|perfect|awesome|great)\b', r'!+$'],
    }
    
    # Urgency patterns
    URGENCY_PATTERNS = {
        'high': [r'\b(urgent|asap|immediately|critical|blocker|production)\b', r'!!!'],
        'medium': [r'\b(soon|important|priority|needed)\b'],
    }
    
    # Conversation mode patterns
    MODE_PATTERNS = {
        'debug': [r'\b(error|traceback|exception|stack|crash|bug)\b', r'File ".*", line \d+'],
        'brainstorm': [r'\b(idea|think|maybe|could|might|what if|alternative)\b'],
        'review': [r'\b(review|check|look at|feedback|opinion|thoughts on)\b'],
    }
    
    def extract_all(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract all heuristic signals from content.
        
        Args:
            content: Text content to analyze
            metadata: Optional metadata (timestamps, etc.)
            
        Returns:
            Flat dict of signals ready to merge into node
        """
        metadata = metadata or {}
        content_lower = content.lower()
        
        return {
            # Intent and mode
            'intent_type': self._detect_intent(content),
            'conversation_mode': self._detect_mode(content_lower),
            'urgency': self._detect_urgency(content_lower),
            
            # Complexity
            'complexity': self._assess_complexity(content),
            'technical_depth': self._compute_technical_depth(content, content_lower),
            
            # Domain signals (multi-label with confidence)
            'domain_signals': self._extract_domain_signals(content_lower),
            'primary_domain': self._get_primary_domain(content_lower),
            
            # Emotional and action signals
            'emotional_valence': self._detect_valence(content_lower),
            'action_required': self._detect_action_required(content_lower),
            
            # Fast filtering aids
            'topic_fingerprint': self._compute_fingerprint(content_lower),
            'entity_hints': self._extract_entity_hints(content),
            'keyword_set': self._extract_keywords(content_lower),
            
            # Structural signals
            'has_code': '```' in content or 'def ' in content or 'class ' in content,
            'has_error': bool(re.search(r'\b(error|exception|traceback)\b', content_lower)),
            'has_urls': bool(re.search(r'https?://', content)),
            'line_count': content.count('\n') + 1,
        }
    
    def _detect_intent(self, content: str) -> str:
        """Detect primary intent: question/request/statement/complaint/celebration."""
        # Check last sentence for question
        sentences = re.split(r'[.!?]+', content)
        last_sentence = sentences[-1].strip() if sentences else content
        
        # Score each intent
        scores = Counter()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    scores[intent] += 1
        
        if scores:
            return scores.most_common(1)[0][0]
        return 'statement'
    
    def _detect_mode(self, content_lower: str) -> str:
        """Detect conversation mode: chat/debug/brainstorm/review."""
        for mode, patterns in self.MODE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return mode
        return 'chat'
    
    def _detect_urgency(self, content_lower: str) -> str:
        """Detect urgency level: low/medium/high."""
        for level in ['high', 'medium']:
            for pattern in self.URGENCY_PATTERNS.get(level, []):
                if re.search(pattern, content_lower):
                    return level
        return 'low'
    
    def _assess_complexity(self, content: str) -> str:
        """Assess complexity: simple/moderate/complex."""
        # Factors: length, code blocks, nested structures, technical density
        score = 0
        
        # Length factor
        if len(content) > 2000:
            score += 2
        elif len(content) > 500:
            score += 1
        
        # Code blocks
        code_blocks = content.count('```')
        if code_blocks > 4:
            score += 2
        elif code_blocks > 0:
            score += 1
        
        # Nested structures (indentation, bullets)
        if re.search(r'^[ \t]{4,}', content, re.MULTILINE):
            score += 1
        
        # Multiple questions
        if content.count('?') > 3:
            score += 1
        
        if score >= 4:
            return 'complex'
        elif score >= 2:
            return 'moderate'
        return 'simple'
    
    def _compute_technical_depth(self, content: str, content_lower: str) -> int:
        """Compute technical depth score 0-10."""
        score = 0
        
        # Code indicators (0-4)
        if '```' in content:
            score += 2
        if re.search(r'\bdef \w+\(', content):
            score += 1
        if re.search(r'\bclass \w+:', content):
            score += 1
        
        # Technical terms (0-3)
        tech_terms = ['api', 'config', 'schema', 'endpoint', 'async', 'await', 
                      'import', 'export', 'module', 'service', 'handler']
        matches = sum(1 for term in tech_terms if term in content_lower)
        score += min(matches, 3)
        
        # Error/debug context (0-2)
        if re.search(r'traceback|exception|error:', content_lower):
            score += 2
        elif 'error' in content_lower:
            score += 1
        
        # File paths (0-1)
        if re.search(r'[/\\]\w+\.(py|js|ts|yaml|json)', content):
            score += 1
        
        return min(score, 10)
    
    def _extract_domain_signals(self, content_lower: str) -> Dict[str, float]:
        """Extract domain signals with confidence scores."""
        signals = {}
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            strong_count = sum(1 for p in patterns['strong'] if p in content_lower)
            weak_count = sum(1 for p in patterns['weak'] if p in content_lower)
            
            # Weighted score: strong=0.3, weak=0.1
            score = (strong_count * 0.3) + (weak_count * 0.1)
            
            if score > 0:
                signals[domain] = min(score, 1.0)  # Cap at 1.0
        
        return signals
    
    def _get_primary_domain(self, content_lower: str) -> str:
        """Get single primary domain (for quick filtering)."""
        signals = self._extract_domain_signals(content_lower)
        if signals:
            primary = max(signals.items(), key=lambda x: x[1])[0]
            # Print new domain detection ONCE for phoenix rebuild monitoring
            new_domains = ['exercise', 'spiritual', 'physical_crafts', 'philosophy',
                          'relationship', 'creative', 'family', 'psychology',
                          'physics', 'hobbies', 'employer']
            if primary in new_domains and primary not in HeuristicEnricher._reported_domains:
                HeuristicEnricher._reported_domains.add(primary)
                print(f"NEW DOMAIN DETECTED [{primary.upper()}]")
            return primary
        return 'general'
    
    def _detect_valence(self, content_lower: str) -> str:
        """Detect emotional valence: positive/neutral/negative."""
        positive = ['great', 'awesome', 'perfect', 'works', 'solved', 'thanks', 
                    'excellent', 'love', 'helpful', 'success']
        negative = ['broken', 'wrong', 'hate', 'frustrated', 'stuck', 'failing',
                    'annoying', 'terrible', 'awful', 'impossible']
        
        pos_count = sum(1 for w in positive if w in content_lower)
        neg_count = sum(1 for w in negative if w in content_lower)
        
        if pos_count > neg_count + 1:
            return 'positive'
        elif neg_count > pos_count + 1:
            return 'negative'
        return 'neutral'
    
    def _detect_action_required(self, content_lower: str) -> bool:
        """Detect if follow-up action is needed."""
        action_words = ['todo', 'follow up', 'next step', 'action item', 
                        'need to', 'should', 'will', 'must', 'remember to']
        return any(w in content_lower for w in action_words)
    
    def _compute_fingerprint(self, content_lower: str) -> str:
        """Compute 64-bit topic fingerprint for fast similarity."""
        # Extract significant words (skip stopwords, keep domain terms)
        words = re.findall(r'\b[a-z]{4,}\b', content_lower)
        
        # Simple hash of sorted unique words
        unique_words = sorted(set(words))[:50]  # Top 50 unique
        fingerprint = hashlib.md5('|'.join(unique_words).encode()).hexdigest()[:16]
        
        return fingerprint
    
    def _extract_entity_hints(self, content: str) -> List[str]:
        """Extract entity hints for filtering (capitalized words, tech terms)."""
        # Capitalized words (likely entities)
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content))
        
        # Tech-specific patterns
        tech_entities = set(re.findall(r'\b(?:v\d+|[A-Z]{2,}|[a-z]+_[a-z]+)\b', content))
        
        combined = entities | tech_entities
        return list(combined)[:20]  # Cap at 20
    
    def _extract_keywords(self, content_lower: str) -> List[str]:
        """Extract top keywords for set-based filtering."""
        # Remove code blocks for cleaner keyword extraction
        clean = re.sub(r'```.*?```', '', content_lower, flags=re.DOTALL)
        
        # Extract words 4+ chars, not stopwords
        stopwords = {
            # Common function words
            'this', 'that', 'with', 'from', 'have', 'been', 'were',
            'they', 'their', 'what', 'when', 'where', 'which', 'there',
            'would', 'could', 'should', 'about', 'some', 'into', 'more',
            'like', 'just', 'your', 'very', 'also', 'here', 'then',
            'than', 'being', 'will', 'does', 'doing', 'those', 'these',
            'such', 'much', 'only', 'other', 'each', 'both', 'after',
            'before', 'over', 'under', 'again', 'because', 'same', 'thought',
            # Chat-specific
            'okay', 'sure', 'yeah', 'thing', 'things', 'something',
            'anything', 'nothing', 'everything', 'really', 'actually',
            'probably', 'basically', 'literally', 'definitely', 'maybe',
        }
        
        words = re.findall(r'\b[a-z]{4,}\b', clean)
        filtered = [w for w in words if w not in stopwords]
        
        # Return top 30 by frequency
        counter = Counter(filtered)
        return [w for w, _ in counter.most_common(30)]


# ============================================================================
# BATCH ENRICHMENT (for pipeline integration)
# ============================================================================

def enrich_nodes_batch(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich a batch of nodes with heuristic signals.
    
    Args:
        nodes: List of node dicts with 'content' field
        
    Returns:
        Same nodes with signals merged in
    """
    enricher = HeuristicEnricher()
    
    for node in nodes:
        content = node.get('content', '')
        metadata = node.get('metadata', {})
        
        signals = enricher.extract_all(content, metadata)
        
        # Merge signals into node (flat structure)
        node['heuristics'] = signals
        
        # Also set top-level fields for fast filtering
        node['primary_domain'] = signals['primary_domain']
        node['technical_depth'] = signals['technical_depth']
        node['complexity'] = signals['complexity']
        node['intent_type'] = signals['intent_type']
    
    return nodes


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    enricher = HeuristicEnricher()
    
    test_content = """Human: I'm getting a weird error with the async handler in my FastAPI endpoint. 
Here's the traceback:
```
Traceback (most recent call last):
  File "server.py", line 45
    TypeError: object NoneType can't be used in 'await' expression
```
Can you help debug this? It's blocking our deploy.
Assistant: The error indicates your async function is trying to await something that returned None. Check if your dependency or service method is returning None instead of a coroutine.
"""
    
    signals = enricher.extract_all(test_content)
    
    print("=" * 60)
    print("HEURISTIC ENRICHMENT TEST")
    print("=" * 60)
    
    for key, value in signals.items():
        print(f"  {key}: {value}")
    
    print("\nKey signals for filtering:")
    print(f"  Primary domain: {signals['primary_domain']}")
    print(f"  Intent: {signals['intent_type']}")
    print(f"  Mode: {signals['conversation_mode']}")
    print(f"  Urgency: {signals['urgency']}")
    print(f"  Technical depth: {signals['technical_depth']}/10")