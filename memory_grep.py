"""
memory_grep.py - Exact keyword search with frequency analysis.
BM25 + inverted index for grep-like precision.
"""
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
import numpy as np


@dataclass
class GrepHit:
    memory_id: str
    count: int
    positions: List[int]
    snippet: str
    timestamp: datetime
    episode_id: Optional[str] = None


@dataclass
class GrepResult:
    term: str
    total_occurrences: int
    unique_memories: int
    hits: List[GrepHit]
    temporal_distribution: Dict[str, int]  # "2024-08" -> count
    co_occurring_terms: List[str]


class MemoryGrep:
    """
    Inverted index + BM25 for precise keyword search.

    Capabilities:
    - Exact term matching (grep-style)
    - Frequency counting per memory
    - Position tracking within documents
    - Temporal distribution
    - Co-occurrence analysis
    """

    def __init__(self, nodes: List[Any]):
        """
        Build indexes from memory nodes.

        Args:
            nodes: List of MemoryNode objects with .id, .combined_content, .created_at
        """
        self.nodes = nodes
        self.node_map = {n.id: n for n in nodes}

        # Build corpus
        self.corpus = [self._get_content(n) for n in nodes]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]

        # BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Inverted index: term -> [(node_idx, positions)]
        self.inverted_index: Dict[str, List[Dict]] = defaultdict(list)
        self._build_inverted_index()

        # Stopwords for co-occurrence filtering
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
            'their', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            'and', 'or', 'but', 'if', 'then', 'else', 'so', 'as',
            'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'once',
            'not', 'no', 'nor', 'only', 'own', 'same', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'can', 'get', 'got', 'like', 'want',
        }

    def _get_content(self, node: Any) -> str:
        """Extract searchable content from a node."""
        if hasattr(node, 'combined_content') and node.combined_content:
            return node.combined_content
        # Fallback: combine human + assistant content
        parts = []
        if hasattr(node, 'human_content') and node.human_content:
            parts.append(node.human_content)
        if hasattr(node, 'assistant_content') and node.assistant_content:
            parts.append(node.assistant_content)
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def _build_inverted_index(self):
        """Build inverted index with position tracking."""
        for doc_idx, tokens in enumerate(self.tokenized_corpus):
            term_positions = defaultdict(list)

            for pos, token in enumerate(tokens):
                term_positions[token].append(pos)

            for term, positions in term_positions.items():
                self.inverted_index[term].append({
                    "doc_idx": doc_idx,
                    "memory_id": self.nodes[doc_idx].id,
                    "count": len(positions),
                    "positions": positions,
                })

    def _phrase_search(self, tokens: List[str]) -> List[Dict]:
        """
        Find documents containing all tokens in the phrase.

        Returns matches in same format as inverted index entries.
        """
        if not tokens:
            return []

        # Get documents containing first token
        first_token_docs = {
            entry["doc_idx"]: entry
            for entry in self.inverted_index.get(tokens[0], [])
        }

        if not first_token_docs:
            return []

        # Filter to documents containing ALL tokens
        matching_docs = set(first_token_docs.keys())
        for token in tokens[1:]:
            token_docs = {
                entry["doc_idx"]
                for entry in self.inverted_index.get(token, [])
            }
            matching_docs &= token_docs
            if not matching_docs:
                return []

        # Build match entries for documents containing all tokens
        matches = []
        for doc_idx in matching_docs:
            # Count phrase occurrences by checking content directly
            content = self.corpus[doc_idx].lower()
            phrase = " ".join(tokens)
            count = content.count(phrase)

            # If exact phrase not found, count as 1 (all words present)
            if count == 0:
                count = 1

            matches.append({
                "doc_idx": doc_idx,
                "memory_id": self.nodes[doc_idx].id,
                "count": count,
                "positions": [],  # Position tracking for phrases is complex, skip for now
            })

        return matches

    def grep(
        self,
        term: str,
        exact: bool = True,
        context_chars: int = 50,
        max_hits: int = 50
    ) -> GrepResult:
        """
        Exact keyword search with frequency analysis.

        Args:
            term: Search term (single word or phrase)
            exact: If False, matches substrings
            context_chars: Characters before/after term in snippet
            max_hits: Maximum results to return

        Returns:
            GrepResult with hits, frequencies, temporal distribution
        """
        term_lower = term.lower()
        term_tokens = self._tokenize(term_lower)

        # Single token: use inverted index directly
        if len(term_tokens) == 1:
            if exact:
                matches = self.inverted_index.get(term_tokens[0], [])
            else:
                matches = []
                for indexed_term, entries in self.inverted_index.items():
                    if term_tokens[0] in indexed_term:
                        matches.extend(entries)
        else:
            # Multi-word phrase: find documents containing ALL tokens
            matches = self._phrase_search(term_tokens)

        if not matches:
            return GrepResult(
                term=term,
                total_occurrences=0,
                unique_memories=0,
                hits=[],
                temporal_distribution={},
                co_occurring_terms=[],
            )

        # Aggregate by memory (same memory may have multiple term matches)
        memory_matches = defaultdict(lambda: {"count": 0, "positions": []})
        for match in matches:
            mem_id = match["memory_id"]
            memory_matches[mem_id]["count"] += match["count"]
            memory_matches[mem_id]["positions"].extend(match["positions"])
            memory_matches[mem_id]["doc_idx"] = match["doc_idx"]

        # Build hits
        hits = []
        temporal_dist = defaultdict(int)

        sorted_matches = sorted(
            memory_matches.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:max_hits]

        for mem_id, data in sorted_matches:
            node = self.node_map[mem_id]
            content = self._get_content(node)

            # Extract snippet around first occurrence
            snippet = self._extract_snippet(content, term, context_chars)

            # Temporal tracking
            if hasattr(node, 'created_at') and node.created_at:
                month_key = node.created_at.strftime("%Y-%m")
                temporal_dist[month_key] += data["count"]

            hits.append(GrepHit(
                memory_id=mem_id,
                count=data["count"],
                positions=data["positions"],
                snippet=snippet,
                timestamp=getattr(node, 'created_at', None),
                episode_id=getattr(node, 'conversation_id', None),
            ))

        # Calculate co-occurring terms
        co_terms = self._find_co_occurring_terms(term_lower, matches)

        return GrepResult(
            term=term,
            total_occurrences=sum(m["count"] for m in memory_matches.values()),
            unique_memories=len(memory_matches),
            hits=hits,
            temporal_distribution=dict(sorted(temporal_dist.items())),
            co_occurring_terms=co_terms,
        )

    def _extract_snippet(self, content: str, term: str, context: int) -> str:
        """Extract snippet with context around term."""
        idx = content.lower().find(term.lower())
        if idx == -1:
            return content[:context * 2] + "..."

        start = max(0, idx - context)
        end = min(len(content), idx + len(term) + context)

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def _find_co_occurring_terms(
        self,
        term: str,
        matches: List[Dict],
        top_k: int = 15
    ) -> List[str]:
        """Find terms that frequently appear alongside the search term."""
        co_counts = Counter()

        for match in matches:
            doc_idx = match["doc_idx"]
            tokens = set(self.tokenized_corpus[doc_idx])
            tokens.discard(term)
            tokens -= self.stopwords
            co_counts.update(tokens)

        return [term for term, _ in co_counts.most_common(top_k)]

    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        BM25 ranked search.

        Returns:
            List of (memory_id, score) tuples
        """
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.nodes[idx].id, float(scores[idx])))

        return results

    def frequency_report(self, term: str) -> Dict[str, Any]:
        """
        Full frequency analysis for a term.
        """
        result = self.grep(term)

        peak_month = None
        peak_count = 0
        if result.temporal_distribution:
            peak_month = max(
                result.temporal_distribution.items(),
                key=lambda x: x[1]
            )
            peak_month, peak_count = peak_month

        return {
            "term": term,
            "total_mentions": result.total_occurrences,
            "unique_memories": result.unique_memories,
            "peak_month": peak_month,
            "peak_count": peak_count,
            "temporal_distribution": result.temporal_distribution,
            "co_occurring_terms": result.co_occurring_terms,
            "top_contexts": [
                {
                    "memory_id": h.memory_id,
                    "count": h.count,
                    "snippet": h.snippet,
                    "timestamp": h.timestamp.isoformat() if h.timestamp else None,
                }
                for h in result.hits[:10]
            ],
        }

    def format_for_context(self, result: GrepResult) -> str:
        """Format grep result for injection into LLM context."""
        if result.total_occurrences == 0:
            return f"GREP: Term '{result.term}' not found in memory vault."

        lines = [
            f"GREP REPORT: '{result.term}'",
            f"Frequency: {result.total_occurrences} occurrences across {result.unique_memories} memories",
        ]

        if result.temporal_distribution:
            peak = max(result.temporal_distribution.items(), key=lambda x: x[1])
            lines.append(f"Peak usage: {peak[0]} ({peak[1]} mentions)")

        if result.co_occurring_terms:
            lines.append(f"Co-occurs with: {', '.join(result.co_occurring_terms[:8])}")

        lines.append("\nTOP CONTEXTS:")
        for hit in result.hits[:5]:
            ts = hit.timestamp.strftime("%Y-%m-%d") if hit.timestamp else "unknown"
            lines.append(f"[{ts}] (x{hit.count}): {hit.snippet}")

        return "\n".join(lines)
