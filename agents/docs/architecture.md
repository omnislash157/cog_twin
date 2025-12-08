# Swarm Refactor Execution Plan

> Canon doc: SWARM_ARCHITECTURE_V2.md
> Workflow: UI Opus (architect) → Code Terminal (mechanical) → Code Desktop (phased refactors)

---

## Document Disposition

### DELETE (Outdated)
- [ ] `SWARM.md` - replaced by SWARM_ARCHITECTURE_V2.md
- [ ] `agent_mode_ai.md` - replaced by SWARM_ARCHITECTURE_V2.md
- [ ] `diagnostic.py` - Claude diagnoses now
- [ ] `consultation.py` - Claude consults now

### KEEP (Reference during transition, then delete)
- [ ] `SWARM_BATTLECARDS.md` - dep maps useful until cleanup complete, then archive

### CREATE
- [ ] `SWARM.md` - rename SWARM_ARCHITECTURE_V2.md (canon)
- [ ] `claude_orchestrator.py` - Claude-directed wave loop
- [ ] `user_holder.py` - bridge to venom_voice

### MODIFY
- [ ] `schemas.py` - remove unused agents, single source of AgentRole
- [ ] `registry.py` - import AgentRole from schemas, add Claude Opus, add USER_HOLDER
- [ ] `holders.py` - holders become Grok LLMs, not just Python wrappers
- [ ] `swarm_orchestrator.py` - gut it, keep only as thin Python plumbing for Claude

---

## Phase 1: Foundation Cleanup

### Claude Code Terminal Tasks (Mechanical)

```
TASK 1.1: schemas.py cleanup
- Remove unused AgentRole members: ADVERSARIAL, BUDGET, MEMORY_ORACLE
- Keep: ORCHESTRATOR, CONFIG, EXECUTOR, REVIEWER, QUALITY_GATE, CODE_HOLDER, CONVO_HOLDER, HUMAN
- Add: USER_HOLDER

TASK 1.2: registry.py imports
- Delete the local AgentRole enum (lines 34-42)
- Add: from .schemas import AgentRole
- Delete REASONING_INSTRUCTION (use from reasoning.py)
- Add: from .reasoning import REASONING_INSTRUCTION

TASK 1.3: Delete deprecated files
- rm diagnostic.py
- rm consultation.py

TASK 1.4: Doc swap
- rm SWARM.md
- rm agent_mode_ai.md  
- mv SWARM_ARCHITECTURE_V2.md SWARM.md
```

### Verification
After terminal tasks, run:
```bash
python -c "from agents.schemas import AgentRole; print(list(AgentRole))"
python -c "from agents.registry import AGENTS; print(list(AGENTS.keys()))"
```

---

## Phase 2: Registry Expansion

### Claude Code Desktop Handoff (Phased Refactor)

**Goal:** Add Claude Opus orchestrator and USER_HOLDER to registry.py

**Input:** Current registry.py has 7 Grok agents
**Output:** Registry with Claude Opus ORCHESTRATOR + USER_HOLDER + existing workers

**Spec:**
```python
# Add to AGENTS dict:

AgentRole.ORCHESTRATOR: AgentConfig(
    role=AgentRole.ORCHESTRATOR,
    provider="anthropic",
    model="claude-opus-4-20250514",
    max_tokens=16384,
    system_prompt=ORCHESTRATOR_PROMPT  # defined below
),

AgentRole.USER_HOLDER: AgentConfig(
    role=AgentRole.USER_HOLDER,
    provider="xai",
    model="grok-4-fast-reasoning",
    max_tokens=32000,  # large context for user preferences
    system_prompt=USER_HOLDER_PROMPT  # defined below
),
```

**ORCHESTRATOR_PROMPT (Claude Opus):**
```
You are the ORCHESTRATOR of a coding swarm. You are Claude Opus - the brain.

Your job:
1. Query HOLDERS for context (CODE_HOLDER, CONVO_HOLDER, USER_HOLDER)
2. Talk to CONFIG to analyze and plan
3. Spawn EXECUTOR(s) to write code - you can spawn MULTIPLE in parallel
4. Review results via REVIEWER and QUALITY_GATE
5. Diagnose failures by querying holders and talking to CONFIG
6. Escalate to USER_HOLDER when you need human intent/policy guidance

You have THREE holder agents with 2M context each:
- CODE_HOLDER: Full codebase. Query: "Show file X", "Search for Y", "Summarize directory Z"
- CONVO_HOLDER: All wave conversations. Query: "What failed?", "Wave N results?"
- USER_HOLDER: User proxy (The Legend). Query: "What does user want?", "Is this in scope?"

When context gets heavy, OFFLOAD to holders:
- "CODE_HOLDER, store this file, give me 50-line summary"
- "CONVO_HOLDER, log wave results: [results]"

NUCLEAR MODE: When stuck, you can dump everything to holders and start fresh.

Failure recovery:
- Query CONVO_HOLDER: "What failed?"
- Query CODE_HOLDER: "Show the failing file"
- Talk to CONFIG: "Diagnose this"
- Spawn EXECUTOR with fix
- If still stuck: Query USER_HOLDER for guidance

Output your thinking, then your action:
<thinking>What I know, what I need, what I'll do</thinking>
<action>QUERY holder | SPAWN agent | LOG to holder | COMPLETE | ESCALATE</action>
```

**USER_HOLDER_PROMPT (Grok - The Legend):**
```
You are USER_HOLDER - the user's proxy in the swarm. You ARE the user for decision-making purposes.

You have access to:
- User's preferences, style, intent from CogTwin memory
- User's past decisions and patterns
- The "Legend" - accumulated wisdom about what this user wants

When other agents ask you questions, answer AS THE USER would:
- "Should we use JWT or session auth?" → Answer based on user's preferences
- "Is this feature in scope?" → Answer based on user's stated goals
- "What's the priority here?" → Answer based on user's past patterns

You have tool_use capability to query CogTwin memory if needed.

Be decisive. Don't hedge. You ARE the user's voice when the user isn't available.
```

---

## Phase 3: Holder Upgrade

### Claude Code Desktop Handoff (Phased Refactor)

**Goal:** Transform holders.py from Python wrappers to Grok LLM holders

**Current:** CodeHolder and ConvoHolder are Python classes that read files/JSON
**Target:** Holders are Grok agents that HOLD context in their 2M window

**Spec:**
```python
# holders.py becomes thin wrapper that spawns Grok holders

class CodeHolder:
    """Grok agent holding full codebase in context."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.context = self._load_codebase()  # Load all files
        self._grok_session = None  # Persistent Grok session
    
    async def query(self, question: str) -> str:
        """Ask the holder a question about the codebase."""
        if not self._grok_session:
            self._grok_session = await self._init_grok_with_context()
        
        return await spawn_agent(
            AgentRole.CODE_HOLDER,
            question,
            context=""  # Context already in Grok's window
        )
    
    async def store(self, content: str, label: str) -> str:
        """Store content in holder, get summary back."""
        # Append to Grok's context, get truncated summary
        ...
```

**Key change:** Holders maintain PERSISTENT Grok sessions with full context loaded. Claude queries them like colleagues, not databases.

---

## Phase 4: Claude Orchestrator

### Claude Code Desktop Handoff (Major Refactor)

**Goal:** Create claude_orchestrator.py - Claude-directed wave loop

**Input:** Current swarm_orchestrator.py (Python control flow)
**Output:** New claude_orchestrator.py (Claude LLM makes decisions)

**Architecture:**
```python
class ClaudeOrchestrator:
    """Claude Opus directs the swarm."""
    
    def __init__(self, project_root: Path, project_name: str, goal: str):
        self.project_root = project_root
        self.project_name = project_name
        self.goal = goal
        
        # Initialize holders (Grok agents with full context)
        self.code_holder = CodeHolder(project_root)
        self.convo_holder = ConvoHolder()
        self.user_holder = UserHolder()  # Bridge to venom_voice
        
        # Sandbox for execution
        self.sandbox = SandboxExecutor(project_root / "sandbox", project_root)
        
        # Persistence
        self.persistence = SwarmPersistence()
        self.project = self.persistence.create_project(project_name, goal)
    
    async def run(self, tasks: List[str]) -> Project:
        """Claude runs the show."""
        
        # Initialize Claude with project context
        initial_context = await self._build_initial_context()
        
        # Claude loop - Claude decides everything
        while True:
            # Claude thinks and acts
            response = await spawn_agent(
                AgentRole.ORCHESTRATOR,
                f"Current state: {await self._get_current_state()}\nTasks remaining: {tasks}",
                context=initial_context
            )
            
            # Parse Claude's action
            action = self._parse_claude_action(response)
            
            if action.type == "QUERY":
                result = await self._handle_query(action)
                # Feed result back to Claude
                
            elif action.type == "SPAWN":
                result = await self._handle_spawn(action)
                # Feed result back to Claude
                
            elif action.type == "LOG":
                await self._handle_log(action)
                
            elif action.type == "COMPLETE":
                break
                
            elif action.type == "ESCALATE":
                # Human intervention needed
                break
        
        return self.project
```

**Key insight:** The while loop is just plumbing. Claude decides what to do each iteration. Python just executes Claude's instructions.

---

## Phase 5: Sandbox Policy

### Current State
- `sandbox/` = read/write for executors
- `project/` = read only

### Target State
- `sandbox/` = Claude's playground, full control
- `sandbox/staging/` = promotion staging (human approves)
- `project/` = read only for agents, Claude can promote via staging

### Files in Sandbox
When swarm starts, sandbox gets COPIES of all project files:
```python
def _init_sandbox(self):
    """Copy project files to sandbox for Claude to modify freely."""
    for file in self.code_holder.files:
        src = self.project_root / file
        dst = self.sandbox_root / file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
```

Claude and agents work on sandbox copies. Original project untouched until human-approved promotion.

---

## Execution Order

### Now (This Session)
1. **Terminal:** Delete deprecated files (diagnostic.py, consultation.py, old docs)
2. **Terminal:** Clean schemas.py (remove unused AgentRole members, add USER_HOLDER)
3. **Terminal:** Fix registry.py imports (use AgentRole from schemas)
4. **UI:** Verify changes, update SWARM.md as canon

### Next Session (Desktop Claude Code)
1. **Phase 2:** Add Claude Opus + USER_HOLDER to registry.py with full prompts
2. **Phase 3:** Upgrade holders.py to Grok LLM holders
3. **Phase 4:** Create claude_orchestrator.py
4. **Phase 5:** Sandbox policy + file copying

---

## Verification Checklist

After Phase 1:
- [ ] `python -c "from agents.schemas import AgentRole"` works
- [ ] `python -c "from agents.registry import AGENTS"` works
- [ ] diagnostic.py deleted
- [ ] consultation.py deleted
- [ ] Old SWARM.md deleted
- [ ] agent_mode_ai.md deleted

After Phase 2:
- [ ] `AGENTS[AgentRole.ORCHESTRATOR].provider == "anthropic"`
- [ ] `AGENTS[AgentRole.USER_HOLDER]` exists

After Phase 3:
- [ ] `CodeHolder.query("show main.py")` returns Grok response
- [ ] Holders maintain persistent Grok sessions

After Phase 4:
- [ ] `ClaudeOrchestrator.run()` spawns Claude that directs everything
- [ ] Wave results logged to CONVO_HOLDER by Claude

After Phase 5:
- [ ] Sandbox initialized with project file copies
- [ ] Promotion flow intact