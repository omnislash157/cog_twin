"""
Research Pipeline - The 5th Lane

Domain-aware knowledge retrieval from curated corpora.
Separate from personal memory lanes - this is EXTERNAL knowledge.

Architecture (from GPT/Grok discussion):
    query --> domain_classifier (tiny model or regex+keywords)
    
    --> 4x personal memory lanes --> always fire in parallel  
        --> output injected as clearly labeled blocks
    
    --> research lane
        |-- if domain in {medical, law, finance, pharmacology, nutrition}
        |       --> ONLY curated corpus (no web search)
        |       --> tone_profile = "cautious, non-diagnostic"
        |
        |-- else if domain in {science, engineering, math, history_pre_1950}
        |       --> primary: curated corpus first
        |       --> if retrieval_confidence < 0.68 --> web_search allowed
        |
        |-- else (casual, coding, memes, etc.)
                --> whatever is cheapest/fastest

Key insight: Personal memory = "what we discussed"
             Research lane = "what is objectively true"

Version: 1.0.0 (acidburn)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Domain Classification
# =============================================================================

class Domain:
    """Domain categories with associated policies."""
    MEDICAL = "medical"           # Health, physiology, nutrition, pharmacology
    LAW_FINANCE = "law_finance"   # Legal, tax, investing, regulations
    SCIENCE = "science"           # Physics, chemistry, biology, engineering, math
    HISTORY = "history"           # Pre-1950 historical facts and biographies
    CODING = "coding"             # Programming, software, technical
    CASUAL = "casual"             # General chat, lifestyle, misc


# High-confidence regex patterns for domain detection
# These override the classifier for obvious cases
DOMAIN_REGEX_OVERRIDES = [
    # Medical/Health - very conservative, catches health topics
    (
        r"(hiit|heart\s*rate|max\s*hr|maxhr|bpm|palpitations|blood\s*pressure|"
        r"medication|dose|dosage|symptom|diagnos|prognosis|treatment|surgery|"
        r"vitamin|supplement|calorie|macro|protein\s*intake|carbs|cholesterol|"
        r"glucose|insulin|diabetes|cancer|tumor|disease|syndrome|disorder|"
        r"prescription|pharmacy|drug\s*interact|side\s*effect|overdose|"
        r"workout\s*recovery|muscle\s*strain|injury|pain\s*relief|inflammation|"
        r"mental\s*health|anxiety|depression|therapy|psych)",
        Domain.MEDICAL
    ),
    # Law/Finance - catches legal and financial topics
    (
        r"(contract|llc|incorporat|taxes|irs|deduction|write[- ]?off|"
        r"investment|etf|stock|securities|401k|ira|pension|estate\s*plan|"
        r"copyright|trademark|patent|lawsuit|liability|tort|statute|"
        r"regulation|compliance|audit|fiduciary|capital\s*gains|dividend|"
        r"mortgage|refinanc|loan|interest\s*rate|credit\s*score)",
        Domain.LAW_FINANCE
    ),
    # Science/Engineering/Math - technical and formal
    (
        r"(theorem|proof|lemma|corollary|axiom|conjecture|"
        r"lagrangian|hamiltonian|fourier|hilbert\s*space|eigenvalue|"
        r"differential\s*equation|ode|pde|integration|derivative|"
        r"quantum|relativity|entropy|thermodynamic|catalyst|synthesis|"
        r"organic\s*chem|inorganic|polymer|crystal|semiconductor|"
        r"circuit|transistor|capacitor|inductor|voltage|ampere|"
        r"algorithm|complexity|big\s*o|np[- ]hard|turing|automata)",
        Domain.SCIENCE
    ),
    # History - pre-1950 and biographical
    (
        r"(world\s*war\s*[i12]|wwi|wwii|civil\s*war|revolutionary\s*war|"
        r"napoleon|caesar|cleopatra|alexander|genghis|ottoman|byzantine|"
        r"industrial\s*revolution|renaissance|medieval|ancient\s*rome|"
        r"ancient\s*greece|egyptian\s*pyramid|mesopotamia|"
        r"18th\s*century|19th\s*century|victorian|edwardian|"
        r"founding\s*fathers|constitution\s*of\s*1787)",
        Domain.HISTORY
    ),
    # Coding - programming and software
    (
        r"(python|javascript|typescript|rust|golang|java\b|c\+\+|"
        r"function|class\s+\w+|def\s+\w+|import\s+|from\s+\w+\s+import|"
        r"api\s*call|endpoint|http|rest|graphql|websocket|"
        r"git\s*commit|pull\s*request|merge|branch|repo|"
        r"docker|kubernetes|aws|azure|gcp|deployment|"
        r"debug|error|exception|stack\s*trace|null\s*pointer|"
        r"database|sql|postgres|mongodb|redis|"
        r"react|svelte|vue|angular|frontend|backend)",
        Domain.CODING
    ),
]


@dataclass
class DomainPolicy:
    """Policy configuration for a domain."""
    domain: str
    allow_open_web: bool = False
    web_confidence_threshold: float = 0.68  # Only used if allow_open_web is True
    allowed_corpora: List[str] = field(default_factory=list)
    require_personal_context: bool = True
    tone_profile: str = "neutral"
    mandatory_disclaimer: str = ""
    max_specificity: str = "detailed"  # "general_principles", "population_guidelines", "detailed"
    log_safety_events: bool = False


# Default domain policies (can be overridden via config)
DEFAULT_POLICIES: Dict[str, DomainPolicy] = {
    Domain.MEDICAL: DomainPolicy(
        domain=Domain.MEDICAL,
        allow_open_web=False,  # NEVER web search for medical
        allowed_corpora=["pubmed_central_oa", "acsm_guidelines", "harvard_med_1928"],
        require_personal_context=True,  # Include user's health history
        tone_profile="cautious_non_diagnostic",
        mandatory_disclaimer=(
            "This is general information only and not medical advice. "
            "Please discuss with your physician or qualified professional."
        ),
        max_specificity="population_guidelines",  # No personal diagnosis
        log_safety_events=True,
    ),
    Domain.LAW_FINANCE: DomainPolicy(
        domain=Domain.LAW_FINANCE,
        allow_open_web=False,  # No web for legal/financial advice
        allowed_corpora=["law_reference", "irs_publications", "sec_official"],
        require_personal_context=False,
        tone_profile="cautious_non_advisory",
        mandatory_disclaimer=(
            "This is general information only, not legal or financial advice. "
            "Consult a qualified professional."
        ),
        max_specificity="general_principles",
        log_safety_events=True,
    ),
    Domain.SCIENCE: DomainPolicy(
        domain=Domain.SCIENCE,
        allow_open_web=True,  # Web allowed if confidence low
        web_confidence_threshold=0.68,
        allowed_corpora=["harvard_1_0", "arxiv", "gutenberg_science"],
        require_personal_context=False,
        tone_profile="precise_with_citations",
        mandatory_disclaimer="",
        max_specificity="detailed",
        log_safety_events=False,
    ),
    Domain.HISTORY: DomainPolicy(
        domain=Domain.HISTORY,
        allow_open_web=True,
        web_confidence_threshold=0.68,
        allowed_corpora=["harvard_history_pre1950", "gutenberg_history"],
        require_personal_context=False,
        tone_profile="precise_with_citations",
        mandatory_disclaimer="",
        max_specificity="detailed",
        log_safety_events=False,
    ),
    Domain.CODING: DomainPolicy(
        domain=Domain.CODING,
        allow_open_web=True,
        web_confidence_threshold=0.5,  # Lower threshold - coding changes fast
        allowed_corpora=["stackoverflow_dump", "internal_dev_notes"],
        require_personal_context=True,  # Include user's project context
        tone_profile="direct",
        mandatory_disclaimer="",
        max_specificity="detailed",
        log_safety_events=False,
    ),
    Domain.CASUAL: DomainPolicy(
        domain=Domain.CASUAL,
        allow_open_web=True,
        web_confidence_threshold=0.3,  # Very lenient
        allowed_corpora=["gutenberg_general"],
        require_personal_context=True,
        tone_profile="direct",
        mandatory_disclaimer="",
        max_specificity="detailed",
        log_safety_events=False,
    ),
}


class DomainClassifier:
    """
    Classifies queries into domains.
    
    Uses regex overrides for high-confidence patterns,
    falls back to keyword heuristics or LLM classification.
    """
    
    def __init__(self, policies: Optional[Dict[str, DomainPolicy]] = None):
        self.policies = policies or DEFAULT_POLICIES
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), domain)
            for pattern, domain in DOMAIN_REGEX_OVERRIDES
        ]
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query into a domain.
        
        Returns:
            (domain, confidence) tuple
        """
        query_lower = query.lower()
        
        # Step 1: Check regex overrides (high confidence)
        for pattern, domain in self._compiled_patterns:
            if pattern.search(query_lower):
                logger.debug(f"Domain override: {domain} (regex match)")
                return domain, 0.95
        
        # Step 2: Keyword heuristics (medium confidence)
        domain_scores = self._score_by_keywords(query_lower)
        if domain_scores:
            top_domain = max(domain_scores, key=domain_scores.get)
            top_score = domain_scores[top_domain]
            if top_score >= 0.5:
                logger.debug(f"Domain heuristic: {top_domain} (score={top_score:.2f})")
                return top_domain, top_score
        
        # Step 3: Default to casual
        logger.debug("Domain default: casual")
        return Domain.CASUAL, 0.3
    
    def _score_by_keywords(self, query_lower: str) -> Dict[str, float]:
        """Score query against domain keyword lists."""
        scores = {}
        
        # Medical keywords
        medical_terms = ["health", "doctor", "medicine", "hospital", "clinic",
                        "exercise", "workout", "fitness", "diet", "weight"]
        scores[Domain.MEDICAL] = sum(1 for t in medical_terms if t in query_lower) / len(medical_terms)
        
        # Law/Finance keywords
        law_finance_terms = ["money", "legal", "court", "bank", "save", "invest",
                           "retire", "tax", "budget", "price"]
        scores[Domain.LAW_FINANCE] = sum(1 for t in law_finance_terms if t in query_lower) / len(law_finance_terms)
        
        # Science keywords
        science_terms = ["physics", "chemistry", "biology", "math", "equation",
                        "experiment", "theory", "formula", "calculate", "research"]
        scores[Domain.SCIENCE] = sum(1 for t in science_terms if t in query_lower) / len(science_terms)
        
        # History keywords
        history_terms = ["history", "war", "century", "ancient", "king", "queen",
                        "empire", "revolution", "historical", "era"]
        scores[Domain.HISTORY] = sum(1 for t in history_terms if t in query_lower) / len(history_terms)
        
        # Coding keywords
        coding_terms = ["code", "program", "bug", "script", "function", "variable",
                       "loop", "array", "library", "framework"]
        scores[Domain.CODING] = sum(1 for t in coding_terms if t in query_lower) / len(coding_terms)
        
        return scores
    
    def get_policy(self, domain: str) -> DomainPolicy:
        """Get policy for a domain."""
        return self.policies.get(domain, self.policies[Domain.CASUAL])


# =============================================================================
# Research Corpus Retrieval
# =============================================================================

@dataclass
class ResearchResult:
    """Single result from research corpus."""
    chunk_id: str
    corpus: str
    title: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchRetrievalResult:
    """Results from research pipeline retrieval."""
    domain: str
    domain_confidence: float
    policy: DomainPolicy
    results: List[ResearchResult]
    retrieval_confidence: float  # Aggregate confidence in results
    used_web: bool = False
    web_results: List[Dict[str, Any]] = field(default_factory=list)


class ResearchCorpusRetriever:
    """
    Retrieves from curated research corpora.
    
    Each corpus is a separate vector index:
    - gutenberg_general: Project Gutenberg books
    - pubmed_central_oa: Open access medical literature
    - harvard_1_0: Harvard pre-1928 books
    - etc.
    
    Data layout:
        research_data/
        |-- gutenberg_general/
        |   |-- corpus.json        # Chunk metadata
        |   |-- vectors.npy        # Embeddings
        |   |-- manifest.json      # Corpus info
        |-- pubmed_central_oa/
        |   |-- ...
    """
    
    def __init__(self, data_dir: Path, embedder=None):
        self.data_dir = data_dir
        self.embedder = embedder
        self.loaded_corpora: Dict[str, Dict[str, Any]] = {}
        
        # Scan available corpora
        self.available_corpora = self._scan_corpora()
        logger.info(f"ResearchCorpusRetriever: {len(self.available_corpora)} corpora available")
    
    def _scan_corpora(self) -> List[str]:
        """Scan data directory for available corpora."""
        corpora = []
        research_dir = self.data_dir / "research_data"
        
        if not research_dir.exists():
            logger.warning(f"Research data directory not found: {research_dir}")
            return corpora
        
        for corpus_dir in research_dir.iterdir():
            if corpus_dir.is_dir():
                manifest_path = corpus_dir / "manifest.json"
                if manifest_path.exists():
                    corpora.append(corpus_dir.name)
                    logger.debug(f"Found corpus: {corpus_dir.name}")
        
        return corpora
    
    def _load_corpus(self, corpus_name: str) -> Optional[Dict[str, Any]]:
        """Load a corpus into memory (lazy loading)."""
        if corpus_name in self.loaded_corpora:
            return self.loaded_corpora[corpus_name]
        
        corpus_dir = self.data_dir / "research_data" / corpus_name
        if not corpus_dir.exists():
            logger.warning(f"Corpus not found: {corpus_name}")
            return None
        
        try:
            # Load manifest
            with open(corpus_dir / "manifest.json") as f:
                manifest = json.load(f)
            
            # Load chunks
            with open(corpus_dir / "corpus.json") as f:
                chunks = json.load(f)
            
            # Load vectors
            vectors_path = corpus_dir / "vectors.npy"
            if vectors_path.exists():
                vectors = np.load(vectors_path)
            else:
                vectors = None
                logger.warning(f"No vectors for corpus: {corpus_name}")
            
            corpus_data = {
                "manifest": manifest,
                "chunks": chunks,
                "vectors": vectors,
            }
            
            # Pre-normalize vectors for cosine similarity
            if vectors is not None:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                corpus_data["normalized"] = vectors / (norms + 1e-8)
            
            self.loaded_corpora[corpus_name] = corpus_data
            logger.info(f"Loaded corpus: {corpus_name} ({len(chunks)} chunks)")
            return corpus_data
            
        except Exception as e:
            logger.error(f"Failed to load corpus {corpus_name}: {e}")
            return None
    
    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        corpora: List[str],
        top_k: int = 5,
    ) -> Tuple[List[ResearchResult], float]:
        """
        Retrieve from specified corpora.
        
        Returns:
            (results, confidence) tuple
        """
        all_results: List[Tuple[ResearchResult, float]] = []
        
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        for corpus_name in corpora:
            if corpus_name not in self.available_corpora:
                logger.debug(f"Corpus not available: {corpus_name}")
                continue
            
            corpus_data = self._load_corpus(corpus_name)
            if corpus_data is None or corpus_data.get("normalized") is None:
                continue
            
            # Compute similarities
            similarities = corpus_data["normalized"] @ query_norm
            
            # Get top results from this corpus
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                score = float(similarities[idx])
                if score < 0.3:  # Minimum relevance threshold
                    continue
                
                chunk = corpus_data["chunks"][idx]
                result = ResearchResult(
                    chunk_id=chunk.get("chunk_id", f"{corpus_name}_{idx}"),
                    corpus=corpus_name,
                    title=chunk.get("title", "Unknown"),
                    text=chunk.get("text", ""),
                    score=score,
                    metadata={
                        "authors": chunk.get("authors", "Unknown"),
                        "book_id": chunk.get("book_id"),
                        "token_count": chunk.get("token_count"),
                    }
                )
                all_results.append((result, score))
        
        # Sort all results by score and take top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        final_results = [r for r, _ in all_results[:top_k]]
        
        # Compute aggregate confidence
        if final_results:
            scores = [r.score for r in final_results]
            # Confidence based on top score and score spread
            top_score = scores[0]
            score_gap = scores[0] - scores[-1] if len(scores) > 1 else 0
            confidence = top_score * 0.7 + (1 - score_gap) * 0.3
        else:
            confidence = 0.0
        
        return final_results, confidence


# =============================================================================
# Research Pipeline Orchestrator
# =============================================================================

class ResearchPipeline:
    """
    Orchestrates the research lane.
    
    Flow:
        1. Classify domain
        2. Get policy for domain
        3. Retrieve from allowed corpora
        4. Optionally fall back to web search (if policy allows and confidence low)
        5. Format results with appropriate labeling
    """
    
    def __init__(
        self,
        data_dir: Path,
        embedder=None,
        web_search_fn=None,  # Optional: async function(query) -> List[Dict]
        policies: Optional[Dict[str, DomainPolicy]] = None,
    ):
        self.classifier = DomainClassifier(policies)
        self.retriever = ResearchCorpusRetriever(data_dir, embedder)
        self.web_search_fn = web_search_fn
        self.embedder = embedder
    
    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        force_domain: Optional[str] = None,
    ) -> ResearchRetrievalResult:
        """
        Full research pipeline retrieval.
        
        Args:
            query: User query
            query_embedding: Pre-computed embedding
            top_k: Max results to return
            force_domain: Override domain classification
        
        Returns:
            ResearchRetrievalResult with domain-appropriate results
        """
        # Step 1: Classify domain
        if force_domain:
            domain = force_domain
            domain_confidence = 1.0
        else:
            domain, domain_confidence = self.classifier.classify(query)
        
        # Step 2: Get policy
        policy = self.classifier.get_policy(domain)
        
        logger.info(f"Research pipeline: domain={domain}, conf={domain_confidence:.2f}, "
                   f"web_allowed={policy.allow_open_web}")
        
        # Step 3: Retrieve from corpora
        results, retrieval_confidence = await self.retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            corpora=policy.allowed_corpora,
            top_k=top_k,
        )
        
        logger.info(f"Research retrieval: {len(results)} results, conf={retrieval_confidence:.2f}")
        
        # Step 4: Web fallback (if allowed and confidence low)
        used_web = False
        web_results = []
        
        if (policy.allow_open_web and 
            retrieval_confidence < policy.web_confidence_threshold and
            self.web_search_fn is not None):
            
            logger.info(f"Triggering web fallback (conf {retrieval_confidence:.2f} < "
                       f"threshold {policy.web_confidence_threshold:.2f})")
            try:
                web_results = await self.web_search_fn(query)
                used_web = True
                logger.info(f"Web search returned {len(web_results)} results")
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        
        return ResearchRetrievalResult(
            domain=domain,
            domain_confidence=domain_confidence,
            policy=policy,
            results=results,
            retrieval_confidence=retrieval_confidence,
            used_web=used_web,
            web_results=web_results,
        )
    
    def format_research_block(self, result: ResearchRetrievalResult) -> str:
        """
        Format research results as a labeled context block.
        
        Returns structured text for injection into prompt.
        """
        lines = []
        
        # Domain header
        lines.append(f"=== RESEARCH CONTEXT (domain: {result.domain}) ===")
        lines.append(f"Policy: {result.policy.tone_profile}")
        lines.append(f"Confidence: {result.retrieval_confidence:.2f}")
        lines.append("")
        
        # Curated results
        if result.results:
            lines.append("--- CURATED KNOWLEDGE ---")
            for i, r in enumerate(result.results[:5], 1):
                lines.append(f"[{i}] {r.title} (score: {r.score:.2f})")
                lines.append(f"    Source: {r.corpus}")
                # Truncate text to reasonable length
                text_preview = r.text[:500] + "..." if len(r.text) > 500 else r.text
                lines.append(f"    {text_preview}")
                lines.append("")
        else:
            lines.append("No curated results found for this query.")
            lines.append("")
        
        # Web results (if used)
        if result.used_web and result.web_results:
            lines.append("--- WEB RESULTS (UNVERIFIED - MAY BE STALE/WRONG) ---")
            for i, w in enumerate(result.web_results[:3], 1):
                title = w.get("title", "Unknown")
                snippet = w.get("snippet", "")[:300]
                lines.append(f"[W{i}] {title}")
                lines.append(f"    {snippet}")
                lines.append("")
        
        # Mandatory disclaimer (if policy requires)
        if result.policy.mandatory_disclaimer:
            lines.append("--- IMPORTANT ---")
            lines.append(result.policy.mandatory_disclaimer)
            lines.append("")
        
        return "\n".join(lines)
    
    def format_instructions_block(self, result: ResearchRetrievalResult) -> str:
        """
        Format domain-specific instructions for the answerer.
        """
        lines = ["=== RESEARCH INSTRUCTIONS ==="]
        
        if result.domain == Domain.MEDICAL:
            lines.extend([
                "- Use CURATED KNOWLEDGE for general principles only",
                "- Do NOT diagnose or predict specific outcomes",
                "- Do NOT recommend specific treatments or dosages",
                "- Reflect user's personal patterns from PERSONAL MEMORY",
                "- Suggest consulting a physician for personalized advice",
                f"- End with: \"{result.policy.mandatory_disclaimer}\"",
            ])
        
        elif result.domain == Domain.LAW_FINANCE:
            lines.extend([
                "- Use CURATED KNOWLEDGE for general legal/financial principles",
                "- Do NOT provide specific legal or financial advice",
                "- Suggest consulting qualified professionals",
                f"- End with: \"{result.policy.mandatory_disclaimer}\"",
            ])
        
        elif result.domain in (Domain.SCIENCE, Domain.HISTORY):
            lines.extend([
                "- Prioritize primary sources over summaries",
                "- Cite sources when making factual claims",
                "- If CURATED and WEB results conflict, note the discrepancy",
            ])
        
        elif result.domain == Domain.CODING:
            lines.extend([
                "- Verify code snippets before recommending",
                "- Note version/framework dependencies",
                "- If WEB results are used, prefer official docs over forum posts",
            ])
        
        else:  # CASUAL
            lines.extend([
                "- Blend PERSONAL MEMORY with RESEARCH naturally",
                "- Be conversational",
            ])
        
        return "\n".join(lines)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Test domain classification
    classifier = DomainClassifier()
    
    test_queries = [
        "What's a good HIIT routine for someone who's been doing 185bpm sessions?",
        "How do I set up a 401k rollover to avoid taxes?",
        "What catalyst did Haber use in the first 1909 ammonia synthesis?",
        "When did Napoleon become emperor?",
        "How do I fix a null pointer exception in Python?",
        "What's a good movie to watch tonight?",
    ]
    
    print("=== Domain Classification Test ===\n")
    for query in test_queries:
        domain, conf = classifier.classify(query)
        policy = classifier.get_policy(domain)
        print(f"Query: {query[:60]}...")
        print(f"  Domain: {domain} (conf={conf:.2f})")
        print(f"  Web allowed: {policy.allow_open_web}")
        print(f"  Tone: {policy.tone_profile}")
        print()
