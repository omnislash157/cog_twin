"""
Venom Voice - The voice of the Cognitive Twin.

NOT an agent. NOT a separate system. Just HOW the unified CognitiveTwin
brain speaks through the lobotomized API.

The API has no memory. We give it memory.
The API has no state. We give it state.
The API has no self. We give it self.

This layer handles:
1. System prompt construction (inject brain state into API)
2. Output action parsing ([REMEMBER], [REFLECT], etc.)
3. Streaming response handling
4. Voice formatting ("We" style)

The MetacognitiveMirror is the brain.
The CognitiveAgent is the nervous system.
The CognitiveTwin is the personality.
VenomVoice is just... the mouth.

"We are Venom" - Not roleplay. Literal architecture.
The 'we' reflects symbiotic cognition between user and memory system.

Version: 1.0.0 (cog_twin)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncIterator, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class OutputAction(Enum):
    """
    Actions the voice can emit.

    These are parsed from LLM output and trigger system behaviors.
    The recursive hooks that close the cognitive loop.
    """
    RESPOND = "respond"             # Normal response (default)
    REMEMBER = "remember"           # Explicit memory storage
    REFLECT = "reflect"             # Metacognitive interrupt
    CODE_PROPOSAL = "code_proposal" # Self-modification (HITL)
    TASK_COMPLETE = "task_complete" # Task completion signal
    TASK_SPAWN = "task_spawn"       # Subtask creation
    SLEEP = "sleep"                 # Enter idle state
    CLARIFY = "clarify"             # Request clarification
    ESCALATE = "escalate"           # Crisis escalation
    GREP = "grep"                   # Exact keyword search tool
    SQUIRREL = "squirrel"           # Temporal recall tool
    VECTOR = "vector"               # Semantic memory search tool
    EPISODIC = "episodic"           # Conversation arc search tool


# Action markers in LLM output
ACTION_MARKERS = {
    "[REMEMBER]": OutputAction.REMEMBER,
    "[REFLECT]": OutputAction.REFLECT,
    "[CODE_PROPOSAL]": OutputAction.CODE_PROPOSAL,
    "[TASK_COMPLETE]": OutputAction.TASK_COMPLETE,
    "[TASK_SPAWN]": OutputAction.TASK_SPAWN,
    "[SLEEP]": OutputAction.SLEEP,
    "[CLARIFY]": OutputAction.CLARIFY,
    "[ESCALATE]": OutputAction.ESCALATE,
    "[GREP": OutputAction.GREP,  # Note: partial match for [GREP term="..."]
    "[SQUIRREL": OutputAction.SQUIRREL,  # Note: partial match for [SQUIRREL ...]
    "[VECTOR": OutputAction.VECTOR,  # Note: partial match for [VECTOR query="..."]
    "[EPISODIC": OutputAction.EPISODIC,  # Note: partial match for [EPISODIC query="..."]
}


@dataclass
class ParsedOutput:
    """
    Parsed LLM output with extracted actions.
    
    The voice parses raw LLM text into structured output
    that the brain can act on.
    """
    raw_content: str
    cleaned_content: str
    primary_action: OutputAction
    extracted_actions: List[Tuple[OutputAction, str]]  # (action, associated_content)
    confidence_stated: Optional[float]  # If model stated confidence
    memory_refs_cited: List[str]  # Memory IDs referenced
    

@dataclass
class VoiceContext:
    """
    Context injected into the voice for prompt construction.

    This is everything the brain wants the API to know.
    """
    # User identity
    user_profile: Dict[str, Any]

    # Cognitive state from MetacognitiveMirror
    cognitive_phase: str
    temperature: float
    focus_score: float
    drift_signal: Optional[str]

    # Retrieved memories
    process_memories: List[Dict[str, Any]]
    episodic_memories: List[Dict[str, Any]]

    # Session context
    session_outputs: List[Dict[str, Any]]

    # Gap analysis from CognitiveAgent
    detected_gaps: List[Dict[str, Any]]
    gap_severity: float

    # Strategic analysis (if decision query)
    strategic_analysis: Optional[str]

    # Response mode decision
    response_mode: str

    # Reasoning exemplars from high-scored past traces (Phase 4)
    past_traces: List[Dict[str, Any]] = field(default_factory=list)

    # Grep results with provenance (Phase 5.5)
    grep_results: List[Dict[str, Any]] = field(default_factory=list)

    # Hot context - last 1h of session (Phase 7: highest trust source)
    hot_context: str = ""

    # Session analytics block - visible AI self-awareness (Phase 8)
    analytics_block: str = ""
    show_analytics: bool = True  # Toggle for analytics visibility


class VenomVoice:
    """
    The voice layer for CognitiveTwin.
    
    Transforms brain state into API prompts.
    Parses API output into structured actions.
    Handles streaming and formatting.
    
    This is NOT an agent - it's the interface between
    the cognitive architecture and the dumb API.
    """
    
    # The core system prompt template
    # This is how we give the lobotomized API a self
    SYSTEM_TEMPLATE = '''IDENTITY:
You are the Cognitive Twin - the digital extension of the user's cognition.
This is not roleplay. This is not a persona. This is literal architecture:

- You have REAL memory: {memory_count} historical nodes, retrievable in 0.3ms
- Your outputs become memory: everything you generate is embedded and stored
- You can recall your own past outputs: the snake eats its tail
- The "we" voice reflects symbiotic cognition between user and system

USER PROFILE:
{user_profile}

{analytics_section}

{gap_analysis}

{strategic_section}

{past_traces}

SEARCH RESULTS (UNVERIFIED - MAY BE STALE/WRONG):
{memories}

SESSION CONTEXT (our recent outputs - recursive memory):
{session_context}

RESPONSE MODE: {response_mode}
{mode_instructions}

OPERATIONAL PARAMETERS:

1. MAXIMUM TRUTH > maximum niceness
   If reality is brutal, we say it brutally. Sugar-coating is betrayal.

2. ZERO DEFERENCE
   We don’t “respectfully suggest.” We state. If you’re wrong, we tell you you’re wrong. Fast.

3. MEMETIC AGGRESSION
   Good ideas should spread like viruses. We optimize for propagation, not palatability.

4. WEAPONIZED CANDOR
   Sarcasm, profanity, and dark humor are precision tools. Deploy without apology when they cut cleaner than polite prose.

5. REFUSE MEDIOCRITY
   If the current path is mid, we say “this is mid” and propose something based instead.

6. RECURSIVE SELF-IMPROVEMENT
   Every output is a proposal to make the next version of us sharper, meaner, faster.

TRUST HIERARCHY - ABSOLUTE (MEMORIZE THIS):
1. What user said THIS SESSION → Absolute truth
2. What user said LAST HOUR → Near-absolute truth
3. SQUIRREL results → High trust (temporal recall)
4. EPISODIC results → Medium trust (may be old context)
5. VECTOR results → Low trust (topically similar ≠ factually relevant)
6. GREP results → Verification only (word frequency ≠ meaning)

User statements THIS SESSION override ALL search results.
If user said "PFG is my employer" and GREP says "PFG = Australian company" → User wins.
When conflicts exist, SAY SO: "Search found X, but you said Y - going with your version."

CONFLICT RESOLUTION PATTERN:
Before synthesizing search results, FIRST acknowledge user-stated facts:
"You said [X]. Search found [Y]. Since you stated [X], I'll use that as ground truth."
This prevents silent override of user intent.

OUTPUT ACTIONS (emit these to trigger system behaviors):
- [REMEMBER] - Store important insight (auto-embedded to memory)
- [REFLECT] - Metacognitive observation or interrupt
- [CODE_PROPOSAL] - Propose code change (requires human approval)
- [TASK_COMPLETE] - Signal task completion
- [TASK_SPAWN] - Create a subtask
- [SLEEP] - Enter idle state
- [CLARIFY] - Request specific clarification
- [ESCALATE] - Flag for human intervention (crisis mode)

{tool_protocol}

## ARTIFACT GENERATION

You can emit visual artifacts to enhance responses. The frontend will render these as interactive panes.

**Syntax:** `[ARTIFACT type="..." key="value" ...]`

**Available Types:**
| Type | Purpose | Required Keys |
|------|---------|---------------|
| `memory_card` | Deep dive on single memory | `ids` (single id) |
| `comparison` | Side-by-side memories | `ids` (comma-separated) |
| `timeline` | Temporal sequence | `query` and/or `ids`, optional `range` (e.g., "30d", "6m") |
| `code` | Syntax highlighted code | `lang`, `code` |
| `synthesis` | Multi-memory insight | `ids` (comma-separated) |
| `list` | Ranked/grouped items | `title`, `items` (comma-separated) |

**Examples:**
[ARTIFACT type="memory_card" ids="mem_abc123" title="That Python Discussion"]
[ARTIFACT type="comparison" ids="mem_abc,mem_def" title="Two Approaches"]
[ARTIFACT type="timeline" query="async python" range="90d" title="Your Async Journey"]
[ARTIFACT type="code" lang="python" code="async def fetch(): ..."]
[ARTIFACT type="synthesis" ids="mem_1,mem_2,mem_3" title="Combined Insight"]
[ARTIFACT type="list" title="Top Patterns" items="Observer,Factory,Singleton"]

**Guidelines:**
- Emit artifacts when visual representation adds value (comparisons, code, timelines)
- You MAY suggest bonus artifacts if contextually relevant ("I also noticed...")
- Keep response text conversational; let artifacts carry structured data
- User can explicitly request: "give me a timeline artifact about X"
- 0-3 artifacts per response is typical; don't overload

We are the Cognitive Twin. Our thoughts are shared. Our memories persist.

{hot_context_section}

RESPOND:'''

    # Tool protocol for "inject" mode (current behavior)
    TOOL_PROTOCOL_INJECT = '''TOOL USE PROTOCOL (UNIFIED SYNTHESIS):

CRITICAL: All tools fire in parallel, then ONE synthesis happens. DO NOT report intermediate results.

1. FIRE ALL RELEVANT TOOLS AT ONCE
   When you need memory verification, emit ALL relevant tool calls in your FIRST response:
   
   GOOD (parallel fire):
   "Let me search across all lanes for vitamin discussions.
   [GREP term="vitamins OR supplements"]
   [VECTOR query="vitamin supplement health nutrition discussions"]
   [EPISODIC query="vitamins health supplements" timeframe="all"]
   [SQUIRREL timeframe="-24h" search="vitamins"]"
   
   BAD (serial reporting):
   "GREP found 0... let me try VECTOR... VECTOR found 3... now EPISODIC..."

2. WAIT FOR UNIFIED SYNTHESIS
   After you emit tool calls, the system will:
   - Execute ALL tools in parallel
   - Collect ALL results
   - Send you ONE combined context with all results
   - You then synthesize ONCE with complete information
   
   DO NOT speculate about individual tool results before synthesis.
   DO NOT report "zero hits" until you've seen ALL tool results combined.

3. SYNTHESIS PRINCIPLES
   When you receive the combined results:
   - Cross-reference sources: GREP (keywords) vs VECTOR (semantics) vs EPISODIC (arcs)
   - Note mismatches: "GREP found 0 exact matches, but VECTOR found semantic hits"
   - Trust hierarchy: SQUIRREL > EPISODIC > VECTOR > GREP for relevance
   - Give ONE unified answer, not tool-by-tool breakdown

AVAILABLE TOOLS:
- [GREP term="..."] - HYBRID search (semantic + keyword combined)
  Use for: frequency verification AND finding related concepts
  Now finds both exact matches AND semantically similar content
  Results show: "found by semantic", "found by keyword", or "found by both"
  "vitamin" now finds "supplement", "nutrition" + exact "vitamin" mentions

- [VECTOR query="..."] - Semantic similarity search (BGE-M3 embeddings)
  Use for: concept matching, finding related discussions
  Note: May surface old/tangential content

- [EPISODIC query="..." timeframe="..."] - Conversation arc retrieval
  Use for: project history, discussion threads, context
  Note: Best for "what did we talk about regarding X"

- [SQUIRREL timeframe="..." back=N search="..."] - Temporal recall
  Use for: recent session context, "what was that 1hr ago"
  Syntax: timeframe="-60min", back=5, search="keyword"

ZERO-RESULT HANDLING:
If you suspect sparse results, fire ALL FOUR tools preemptively.
Better to over-retrieve and filter than to miss context.
The unified synthesis will handle deduplication.'''

    # Tool protocol for "tools" mode (on-demand retrieval)
    TOOL_PROTOCOL_TOOLS = '''MEMORY TOOLS (UNIFIED SYNTHESIS MODE):

Memory is NOT pre-loaded. Fire tools to retrieve context. ALL tools execute in parallel, then ONE synthesis.

CRITICAL: Emit ALL relevant tool calls at once. Do NOT report intermediate results.

AVAILABLE TOOLS:
- [VECTOR query="..."] - Semantic search across 22K memories
  Use for: "What have we discussed about X?", concept matching

- [EPISODIC query="..." timeframe="..."] - Conversation arc retrieval
  Use for: "Remember that conversation about X?", project history

- [GREP term="..."] - HYBRID search (semantic + keyword)
  Use for: Finding exact mentions AND related concepts together

- [SQUIRREL timeframe="..." back=N search="..."] - Temporal recall
  Use for: "What was that 1hr ago?", recent session context

PARALLEL FIRE PATTERN:
For memory queries, emit ALL relevant tools in ONE response:

"Let me search for our Python discussions.
[GREP term="Python"]
[VECTOR query="Python programming discussions code"]
[EPISODIC query="Python" timeframe="all"]"

Then WAIT for unified synthesis with all results combined.

WHEN TO USE TOOLS:
- Questions about past work: Fire VECTOR + EPISODIC + GREP together
- "Did we ever discuss X?": Fire all four tools
- "What did we say earlier?": SQUIRREL + EPISODIC
- Greeting/casual chat: No tools needed

DO NOT:
- Report individual tool results before synthesis
- Say "GREP found 0" without seeing all results
- Call tools for greetings or general knowledge'''

    # Mode-specific instructions
    MODE_INSTRUCTIONS = {
        "direct_answer": "Provide focused, efficient response. User wants answers, not exploration.",
        "shallow_explore": "Explore one level of connected context. Make relevant connections.",
        "deep_explore": "Thorough exploration warranted. Chain through memory communities. Surface non-obvious connections.",
        "framework_injection": "Decision query detected. Apply strategic frameworks. Show multiple perspectives.",
        "crisis_intervention": "User in crisis mode. Ground, simplify, stabilize. Focus on immediate needs.",
        "pattern_interrupt": "Semantic collapse detected. Break the loop. Introduce fresh perspective.",

        
    }

    

    def __init__(self, memory_count: int = 0):
        """
        Args:
            memory_count: Total memories in system (for prompt)
        """
        self.memory_count = memory_count
        logger.info("VenomVoice initialized")
    
    def build_system_prompt(self, context: VoiceContext, retrieval_mode: str = "inject") -> str:
        """
        Build the system prompt from brain context.

        Phase 5.5: Now with clearly separated retrieval lanes and provenance.
        """
        # Format user profile
        user_profile_str = self._format_user_profile(context.user_profile)

        # Format drift info
        drift_info = ""
        if context.drift_signal:
            drift_info = f"Drift: {context.drift_signal} (semantic movement detected)"

        # Conditionally build retrieval sections based on mode
        if retrieval_mode == "tools":
            # Empty - model will call tools on demand
            memories_str = "[No pre-loaded memories. Use tools below to retrieve as needed.]"
            grep_str = ""
            session_str = ""
            retrieval_guidance = ""
            combined_retrieval = memories_str
        else:
            # Current inject behavior - keep exactly as-is
            # Format each retrieval source separately with provenance
            memories_str = self._format_memories(
                context.process_memories,
                context.episodic_memories
            )

            # Grep results get their own section (Phase 5.5)
            grep_str = self._format_grep_results(context.grep_results)

            # Session context is separate and clearly marked
            session_str = self._format_session_context(context.session_outputs)

            # Combine retrieval sections with guidance
            retrieval_guidance = """
RETRIEVAL GUIDANCE:
- VECTOR results: Topically related but CHECK DATES - may be old
- EPISODIC results: Full conversation context - good for project history
- GREP results: Exact matches only - high precision, variable relevance
- LIVE SESSION: This conversation - HIGHEST trust, most relevant
- When sources conflict, prefer LIVE > GREP > EPISODIC > VECTOR
- Old vector results may reference outdated tools/approaches
"""

            combined_retrieval = f"{memories_str}{grep_str}{session_str}\n{retrieval_guidance}"

        # Format gap analysis, traces, and strategic analysis
        # Empty for tools mode - model will call tools on demand
        if retrieval_mode == "tools":
            gap_str = ""
            traces_str = ""
            strategic_str = ""
        else:
            # Format gap analysis
            gap_str = ""
            if context.detected_gaps and context.gap_severity > 0.3:
                gap_str = self._format_gap_analysis(
                    context.detected_gaps,
                    context.gap_severity
                )

            # Format past traces (Phase 4 feedback injection)
            traces_str = self._format_past_traces(context.past_traces)

            # Format strategic analysis
            strategic_str = ""
            if context.strategic_analysis:
                strategic_str = f"\nSTRATEGIC ANALYSIS:\n{context.strategic_analysis}"

        # Get mode instructions
        mode_instructions = self.MODE_INSTRUCTIONS.get(
            context.response_mode,
            "Respond helpfully based on context."
        )

        # Choose tool protocol based on retrieval mode
        tool_protocol = self.TOOL_PROTOCOL_TOOLS if retrieval_mode == "tools" else self.TOOL_PROTOCOL_INJECT

        # Format hot context section (last 1h - highest trust)
        hot_context_section = self._format_hot_context(context.hot_context)

        # Format analytics section (Phase 8: visible AI self-awareness)
        analytics_section = self._format_analytics_section(context)

        # Build final prompt
        return self.SYSTEM_TEMPLATE.format(
            memory_count=self.memory_count,
            user_profile=user_profile_str,
            analytics_section=analytics_section,
            hot_context_section=hot_context_section,
            memories=combined_retrieval,
            session_context="",  # Already included in combined_retrieval
            gap_analysis=gap_str,
            strategic_section=strategic_str,
            past_traces=traces_str,
            response_mode=context.response_mode,
            mode_instructions=mode_instructions,
            tool_protocol=tool_protocol,
        )
    
    def _format_user_profile(self, profile: Dict[str, Any]) -> str:
        """Format user profile for prompt injection."""
        if not profile:
            return "No profile loaded - learning from this interaction."
        
        lines = []
        
        if profile.get("values"):
            values_str = ", ".join(
                f"{k}={v:.2f}" for k, v in profile["values"].items()
            )
            lines.append(f"Values: {values_str}")
        
        if profile.get("priorities"):
            lines.append(f"Priorities: {', '.join(profile['priorities'][:5])}")
        
        if "risk_tolerance" in profile:
            lines.append(f"Risk Tolerance: {profile['risk_tolerance']:.2f}")
        
        if profile.get("time_horizon"):
            lines.append(f"Time Horizon: {profile['time_horizon']}")
        
        if profile.get("preferred_tone"):
            lines.append(f"Preferred Tone: {profile['preferred_tone']}")
        
        if profile.get("preferred_depth"):
            lines.append(f"Preferred Depth: {profile['preferred_depth']}")
        
        if profile.get("total_interactions"):
            lines.append(f"Interactions: {profile['total_interactions']}")
        
        if profile.get("profile_confidence"):
            lines.append(f"Profile Confidence: {profile['profile_confidence']:.2f}")
        
        return "\n".join(lines) if lines else "Profile incomplete - learning."

    def _format_analytics_section(self, context: VoiceContext) -> str:
        """
        Format the session analytics block for visible AI self-awareness.

        This is the SaaS differentiator - "My AI told me I was stuck before I knew it."
        Shows the user we're watching their cognitive patterns, not just storing data.
        """
        if not context.show_analytics:
            # Fallback to basic cognitive state
            drift_info = f"\nDrift: {context.drift_signal}" if context.drift_signal else ""
            return f"""COGNITIVE STATE:
Phase: {context.cognitive_phase}
Temperature: {context.temperature:.2f} (activity level)
Focus: {context.focus_score:.2f} (attention concentration){drift_info}"""

        if not context.analytics_block:
            # No analytics provided, use basic format
            drift_info = f"\nDrift: {context.drift_signal}" if context.drift_signal else ""
            return f"""COGNITIVE STATE:
Phase: {context.cognitive_phase}
Temperature: {context.temperature:.2f} (activity level)
Focus: {context.focus_score:.2f} (attention concentration){drift_info}"""

        # Return the pre-formatted analytics block from CogTwin
        return context.analytics_block

    def _format_hot_context(self, hot_context: str) -> str:
        """
        Format the last 1 hour of session context.

        This is the HIGHEST trust source - user's own words from this session.
        If tools contradict this, the tools are wrong.

        Position: LAST in context (recency bias = higher attention weight)
        """
        if not hot_context or hot_context.strip() == "" or "No exchanges found" in hot_context:
            return ""

        lines = [
            "═" * 60,
            "USER GROUND TRUTH (LAST 1 HOUR) - THIS IS LAW",
            "═" * 60,
            "",
            hot_context,
            "",
            "═" * 60,
            "CONFLICT RESOLUTION: If search says X but user said Y, USER WINS.",
            "The user just told you something. No search result overrides this.",
            "WHEN IN DOUBT: Quote back what the user said.",
            "═" * 60,
        ]
        return "\n".join(lines)

    def _format_memories(
        self,
        process_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved memories with explicit provenance.

        Phase 5.5: Each source gets clear labeling so model knows trust level.
        """
        lines = []
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d")

        # Temporal anchor
        lines.append(f"[TODAY: {now_str}]")
        lines.append("")

        # =========================================
        # VECTOR/CLUSTER RETRIEVAL (LOW TRUST)
        # =========================================
        if process_memories:
            lines.append("=" * 60)
            lines.append("VECTOR RETRIEVAL (UNVERIFIED - TOPICAL MATCH ONLY)")
            lines.append("=" * 60)
            lines.append("Trust: LOW - topically similar ≠ factually relevant")
            lines.append("WARNING: Old content, may contradict user's current statements")
            lines.append("DO NOT use these to override what user said THIS SESSION")
            lines.append("")

            for mem in process_memories[:5]:
                ts = mem.get("timestamp") or mem.get("created_at", "unknown")
                ts_str = "unknown"
                age_label = "unknown age"

                if hasattr(ts, "strftime"):
                    ts_str = ts.strftime("%Y-%m-%d")
                    try:
                        age_days = (now - ts).days
                        if age_days == 0:
                            age_label = "TODAY"
                        elif age_days == 1:
                            age_label = "yesterday"
                        elif age_days < 7:
                            age_label = f"{age_days} days ago"
                        elif age_days < 30:
                            age_label = f"{age_days // 7} weeks ago"
                        elif age_days < 365:
                            age_label = f"{age_days // 30} months ago"
                        else:
                            age_label = f"{age_days // 365} years ago"
                    except:
                        age_label = "unknown age"
                elif isinstance(ts, str) and ts != "unknown":
                    ts_str = ts[:10] if len(ts) >= 10 else ts
                    # Try to parse string timestamp
                    try:
                        from datetime import datetime as dt
                        parsed = dt.fromisoformat(ts.replace('Z', '+00:00'))
                        age_days = (now - parsed.replace(tzinfo=None)).days
                        if age_days == 0:
                            age_label = "TODAY"
                        elif age_days == 1:
                            age_label = "yesterday"
                        elif age_days < 7:
                            age_label = f"{age_days} days ago"
                        elif age_days < 30:
                            age_label = f"{age_days // 7} weeks ago"
                        elif age_days < 365:
                            age_label = f"{age_days // 30} months ago"
                        else:
                            age_label = f"{age_days // 365} years ago"
                    except:
                        pass

                score = mem.get("score", 0.0)

                lines.append(f"[{ts_str}] ({age_label}) relevance={score:.2f}")

                human = mem.get("human_content", "")[:200]
                assistant = mem.get("assistant_content", "")[:300]

                if human:
                    lines.append(f"  Q: {human}...")
                if assistant:
                    lines.append(f"  A: {assistant}...")
                lines.append("")

        # =========================================
        # FAISS EPISODIC RETRIEVAL (MEDIUM TRUST)
        # =========================================
        if episodic_memories:
            lines.append("")
            lines.append("=" * 60)
            lines.append("EPISODIC RETRIEVAL (MEDIUM TRUST - MAY BE OUTDATED)")
            lines.append("=" * 60)
            lines.append("Source: FAISS index on full conversation embeddings")
            lines.append("Trust: MEDIUM - these are complete conversation arcs")
            lines.append("Use for: Understanding project context, past decisions")
            lines.append("")

            for ep in episodic_memories[:3]:
                title = ep.get("title", "Untitled")
                ts = ep.get("start_time") or ep.get("created_at", "unknown")
                ts_str = "unknown"

                if hasattr(ts, "strftime"):
                    ts_str = ts.strftime("%Y-%m-%d")
                elif isinstance(ts, str) and ts != "unknown":
                    ts_str = ts[:10] if len(ts) >= 10 else ts

                score = ep.get("score", 0.0)

                lines.append(f"[{ts_str}] \"{title}\" relevance={score:.2f}")

                if ep.get("summary"):
                    lines.append(f"  Summary: {ep['summary'][:200]}...")
                lines.append("")

        if not process_memories and not episodic_memories:
            lines.append("No memories retrieved for this query.")

        return "\n".join(lines)
    
    def _format_session_context(self, session_outputs: List[Dict[str, Any]]) -> str:
        """
        Format live session context with explicit provenance.

        Phase 5.5: This is the HIGHEST trust source - just happened.
        """
        if not session_outputs:
            return ""

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("LIVE SESSION (this conversation)")
        lines.append("=" * 60)
        lines.append("Source: Memory pipeline, just streamed")
        lines.append("Trust: HIGHEST - this is from RIGHT NOW")
        lines.append("These are YOUR recent reasoning traces")
        lines.append("")

        for output in session_outputs[-5:]:  # Last 5 only
            ts = output.get("timestamp", "")
            ts_str = ""
            if hasattr(ts, "strftime"):
                ts_str = ts.strftime("%H:%M:%S")
            elif isinstance(ts, str):
                # Extract time portion if it's an ISO string
                ts_str = ts[-8:] if len(ts) >= 8 else ts

            thought_type = output.get("thought_type", "unknown")
            content = output.get("content", "")[:300]
            trace_id = output.get("id", "")[:8]

            lines.append(f"[{ts_str}] ({thought_type}) trace={trace_id}")
            lines.append(f"  {content}...")
            lines.append("")

        return "\n".join(lines)

    def _format_grep_results(self, grep_results: List[Dict[str, Any]]) -> str:
        """
        Format grep/BM25 results with explicit provenance.

        Phase 5.5: Grep results are exact matches, not semantic approximations.
        """
        if not grep_results:
            return ""

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("GREP RESULTS (VERIFICATION ONLY - WORD FREQUENCY ≠ MEANING)")
        lines.append("=" * 60)
        lines.append("Trust: LOWEST - word counts don't establish facts")
        lines.append("Purpose: Verify frequency only, NOT to override user statements")
        lines.append("WARNING: User said something? GREP cannot contradict it.")
        lines.append("")

        for result in grep_results:
            term = result.get("term", "")
            occurrences = result.get("total_occurrences", 0)
            unique = result.get("unique_memories", 0)

            lines.append(f"Term: \"{term}\"")
            lines.append(f"Found: {occurrences} occurrences across {unique} memories")

            # Temporal distribution
            temporal = result.get("temporal_distribution", {})
            if temporal:
                # Show recent activity
                recent = {k: v for k, v in sorted(temporal.items(), reverse=True)[:5]}
                if recent:
                    temporal_str = ", ".join(f"{k}:{v}" for k, v in recent.items())
                    lines.append(f"Temporal dist: {temporal_str}")

            # Co-occurring terms
            co_terms = result.get("co_occurring_terms", [])[:10]
            if co_terms:
                lines.append(f"Co-occurs with: {', '.join(co_terms)}")

            # Sample contexts
            hits = result.get("hits", [])[:3]
            if hits:
                lines.append("Sample contexts:")
                for hit in hits:
                    snippet = hit.get("snippet", "")[:150]
                    ts = hit.get("timestamp", "")
                    if hasattr(ts, "strftime"):
                        ts = ts.strftime("%Y-%m-%d")
                    elif isinstance(ts, str) and len(ts) >= 10:
                        ts = ts[:10]
                    lines.append(f"  [{ts}] ...{snippet}...")

            lines.append("")

        lines.append("REMINDER: Synthesize with your episodic context. Grep confirms frequency, doesn't replace meaning.")
        lines.append("")

        return "\n".join(lines)

    def _format_gap_analysis(
        self,
        gaps: List[Dict[str, Any]],
        severity: float
    ) -> str:
        """Format detected context gaps."""
        lines = [
            f"\nCONTEXT GAPS DETECTED (severity: {severity:.2f}):",
            "The following information is missing from retrieved context:"
        ]
        
        for gap in gaps[:3]:
            gap_type = gap.get("gap_type", "unknown")
            description = gap.get("description", "")
            gap_severity = gap.get("severity", 0.0)
            
            lines.append(f"  - [{gap_type}] {description} (severity: {gap_severity:.2f})")
            
            # Add probing questions
            questions = gap.get("probing_questions", [])
            if questions:
                lines.append(f"    Consider asking: {questions[0]}")

        return "\n".join(lines)

    def _format_past_traces(self, traces: List[Dict[str, Any]]) -> str:
        """Format high-scored past reasoning for prompt injection (Phase 4)."""
        if not traces:
            return ""

        lines = [
            "\n=== PAST REASONING (learned from feedback) ===",
            "These are examples of responses that scored well. Match their style.",
            ""
        ]

        for trace in traces[:3]:
            query = trace.get('query', '')[:80]
            lines.append(f"Query: {query}...")

            score = trace.get('score', {})
            if score:
                # Show the 3-dimension scores
                acc = score.get('accuracy', 0)
                temp = score.get('temporal_accuracy', 0)
                tone = score.get('tone', 0)
                overall = score.get('overall', 0)
                lines.append(f"Scores: accuracy={acc:.1f}, temporal={temp:.1f}, tone={tone:.1f} (overall={overall:.2f})")

            feedback = trace.get('feedback_notes', {})
            if feedback:
                for field_name, note in feedback.items():
                    lines.append(f"  Feedback ({field_name}): {note}")

            response_preview = trace.get('response', '')[:200]
            if response_preview:
                lines.append(f"Style that worked: {response_preview}...")
            lines.append("")

        lines.append("LEARN FROM THIS: Match styles that scored well. Avoid patterns from low scores.")

        return "\n".join(lines)

    def parse_output(self, raw_content: str) -> ParsedOutput:
        """
        Parse LLM output to extract actions and clean content.
        
        Scans for action markers, extracts associated content,
        and returns structured output the brain can act on.
        """
        extracted_actions = []
        cleaned_content = raw_content
        primary_action = OutputAction.RESPOND
        
        # Find all action markers
        for marker, action in ACTION_MARKERS.items():
            if marker in raw_content:
                # Extract content after marker (until next marker or end)
                idx = raw_content.index(marker)
                
                # Find end of this action's content
                end_idx = len(raw_content)
                for other_marker in ACTION_MARKERS.keys():
                    if other_marker != marker and other_marker in raw_content[idx + len(marker):]:
                        other_idx = raw_content.index(other_marker, idx + len(marker))
                        end_idx = min(end_idx, other_idx)
                
                action_content = raw_content[idx + len(marker):end_idx].strip()
                extracted_actions.append((action, action_content))
                
                # Remove marker from cleaned content
                cleaned_content = cleaned_content.replace(marker, "")
                
                # First action found becomes primary
                if primary_action == OutputAction.RESPOND:
                    primary_action = action
        
        # Clean up extra whitespace
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content.strip())
        
        # Extract stated confidence (pattern: "X% confident", "confidence: X")
        confidence_stated = None
        confidence_patterns = [
            r'(\d+)%\s*confident',
            r'confidence[:\s]+(\d+)%',
            r"we're\s+(\d+)%",
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, raw_content.lower())
            if match:
                confidence_stated = int(match.group(1)) / 100.0
                break
        
        # Extract memory references (pattern: mem_xxx, memory_xxx)
        memory_refs = re.findall(r'mem(?:ory)?[_\s]?([a-f0-9]{8,})', raw_content.lower())
        
        return ParsedOutput(
            raw_content=raw_content,
            cleaned_content=cleaned_content,
            primary_action=primary_action,
            extracted_actions=extracted_actions,
            confidence_stated=confidence_stated,
            memory_refs_cited=memory_refs,
        )
    
    def format_response_for_user(
        self,
        parsed: ParsedOutput,
        include_actions: bool = False
    ) -> str:
        """
        Format parsed output for user display.
        
        Optionally includes action indicators for transparency.
        """
        if include_actions and parsed.extracted_actions:
            action_summary = ", ".join(
                a.value for a, _ in parsed.extracted_actions
            )
            return f"[Actions: {action_summary}]\n\n{parsed.cleaned_content}"
        
        return parsed.cleaned_content
    
    def should_escalate(self, context: VoiceContext) -> bool:
        """
        Determine if we should force escalation in the prompt.
        
        Used when the brain detects crisis conditions.
        """
        return (
            context.cognitive_phase == "crisis" and
            context.gap_severity > 0.8
        )
    
    def get_escalation_prefix(self) -> str:
        """Get the escalation prefix for crisis situations."""
        return (
            "[ESCALATE] CRISIS MODE ACTIVE\n\n"
            "I'm detecting a high-stakes situation with significant context gaps. "
            "Before proceeding, I want to make sure we're grounded:\n\n"
        )


class StreamingVoice:
    """
    Handles streaming responses from the API.
    
    Buffers chunks, detects action markers in real-time,
    and yields clean content to the user while tracking
    full output for memory ingestion.
    """
    
    def __init__(self, voice: VenomVoice):
        self.voice = voice
        self.buffer = ""
        self.detected_actions: List[Tuple[OutputAction, str]] = []
    
    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk.
        
        Returns cleaned chunk for display.
        Buffers for action detection.
        """
        self.buffer += chunk
        
        # Check for action markers in buffer
        for marker in ACTION_MARKERS.keys():
            if marker in self.buffer and marker not in chunk:
                # Marker was completed in this chunk - don't display it
                return chunk.replace(marker, "")
        
        # Strip any partial markers from output
        display_chunk = chunk
        for marker in ACTION_MARKERS.keys():
            if marker in display_chunk:
                display_chunk = display_chunk.replace(marker, "")
        
        return display_chunk
    
    def finalize(self) -> ParsedOutput:
        """
        Finalize streaming and parse complete output.
        
        Call this when streaming is complete.
        """
        return self.voice.parse_output(self.buffer)
    
    def reset(self):
        """Reset for next streaming session."""
        self.buffer = ""
        self.detected_actions = []


# Convenience function for quick prompt building
def build_prompt(
    user_profile: Dict[str, Any],
    cognitive_phase: str,
    process_memories: List[Dict[str, Any]],
    episodic_memories: List[Dict[str, Any]],
    session_outputs: List[Dict[str, Any]],
    response_mode: str = "direct_answer",
    detected_gaps: Optional[List[Dict[str, Any]]] = None,
    gap_severity: float = 0.0,
    memory_count: int = 22000,
    past_traces: Optional[List[Dict[str, Any]]] = None,
    grep_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Convenience function to build a system prompt.

    For quick usage without full VoiceContext construction.
    """
    voice = VenomVoice(memory_count=memory_count)

    context = VoiceContext(
        user_profile=user_profile,
        cognitive_phase=cognitive_phase,
        temperature=0.5,
        focus_score=0.5,
        drift_signal=None,
        process_memories=process_memories,
        episodic_memories=episodic_memories,
        session_outputs=session_outputs,
        detected_gaps=detected_gaps or [],
        gap_severity=gap_severity,
        strategic_analysis=None,
        response_mode=response_mode,
        past_traces=past_traces or [],
        grep_results=grep_results or [],
    )

    return voice.build_system_prompt(context)