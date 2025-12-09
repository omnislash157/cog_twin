# Enterprise Fork Wiring Documentation

**Date:** December 9, 2024
**Session:** Emergency enterprise pivot + demo recovery
**Status:** Fork 3 (Driscoll Enterprise) fully wired

---

## Monolith Architecture

```
MONOLITH (source of truth)
│
├─→ FORK 1: Cog Twin (ADHD SaaS)
│   ├── Market: ADHD community, productivity
│   ├── Price: $30/month consumer
│   ├── Features: Full memory pipelines, metacognitive mirror
│   └── Status: Ship v1 this week
│
├─→ FORK 2: Cog Twin Medical (Alzheimer's/Pharma)
│   ├── Market: Medical, pharma investors, caregivers
│   ├── Price: Enterprise/institutional
│   ├── Features: Same architecture, compliance wrapper
│   └── Status: Investor on deck, timeline needed
│
└─→ FORK 3: Driscoll Enterprise (Food Service) ← WIRED TODAY
    ├── Market: Distributors, multi-unit accounts
    ├── Tiers:
    │   ├── Purchasing/Ops: Dumb bot, manual-stuffed, $0 internal
    │   ├── Sales ($150k): Hivemind recall, parent/child vaults
    │   └── Invoice Slayer: 5th pipeline tool, one-shot integration
    └── Status: Clean fork after this week's v1
```

---

## What Was Built Today

### Core Concept

Config-driven feature flags that let the same codebase run as:
1. **Full CogTwin** - All 5 memory lanes, FAISS, 23K nodes, metacognitive mirror
2. **Enterprise Lite** - No memory pipelines, just DOCX context stuffing into 2M token window

The switch is a single YAML flag: `features.memory_pipelines: true/false`

---

## Files Created

### 1. `doc_loader.py` (NEW - 490 lines)

DOCX loading and context stuffing engine.

**Key Classes:**
- `DocLoader` - Loads .docx files, caches content, detects divisions from folder structure
- `DivisionContextBuilder` - Builds formatted context strings for LLM prompt injection

**Division Detection Logic:**
```python
# Folder structure: manuals/Driscoll/Warehouse/foo.docx
# docs_dir = manuals/Driscoll
# rel_path = Warehouse/foo.docx
# parts[0] = "warehouse" (division)
```

**Stats from Driscoll manuals:**
- 21 documents
- ~27K tokens
- All in "warehouse" division

### 2. `enterprise_twin.py` (NEW - 295 lines)

Wrapper class that conditionally uses CogTwin or lightweight context stuffing.

**Decision Tree:**
```python
if memory_enabled():
    # Full CogTwin with all 5 lanes
    from cog_twin import CogTwin
    self._twin = CogTwin(data_dir=self.data_dir)
else:
    # Lightweight mode - no FAISS, no embeddings
    # Just load docs and call LLM directly
    self._doc_builder = DivisionContextBuilder(loader)
```

**Key Method:**
```python
async def think(self, user_input, tenant=None, stream=True):
    if self._memory_mode:
        # Delegate to full CogTwin
        async for chunk in self._twin.think(user_input):
            yield chunk
    else:
        # Build doc context and call LLM directly
        doc_context = self._doc_builder.get_context_for_division(division)
        # ... stream response
```

### 3. `enterprise_config.yaml` (NEW - 140 lines)

Complete enterprise configuration with all feature flags.

**Key Sections:**
```yaml
deployment:
  mode: enterprise          # personal | enterprise
  tier: basic               # basic | advanced | full

features:
  memory_pipelines: false   # THE MASTER SWITCH
  context_stuffing: true
  ui:
    swarm_loop: false
    memory_space_3d: false
    chat_basic: true

docs:
  docs_dir: ./manuals/Driscoll
  stuffing:
    max_tokens_per_division: 200000
```

### 4. `enterprise_voice.py` (EXISTING - extended)

Voice templates for different divisions:
- `corporate` - Professional, zero snark
- `troll` - Sarcastic expert dispatcher (for transportation)
- `helpful` - Friendly assistant

Division-to-voice mapping in config.

### 5. `enterprise_tenant.py` (EXISTING)

Tenant context class for multi-tenant support:
```python
@dataclass
class TenantContext:
    tenant_id: str
    division: str
    zone: Optional[str]
    role: str  # user, manager, admin
```

---

## Files Modified

### 1. `config_loader.py` (Added helper functions)

New functions added at bottom:
```python
def get_division_categories(division: str) -> list:
    """Get document categories accessible to a division."""

def get_docs_dir() -> str:
    """Get the documents directory path."""

def get_max_stuffing_tokens() -> int:
    """Get max tokens to stuff per request."""
```

### 2. `backend/app/main.py` (Enterprise imports + /api/config)

**Added imports:**
```python
try:
    from config_loader import (
        load_config as load_enterprise_config,
        cfg as enterprise_cfg,
        memory_enabled,
        is_enterprise_mode,
        get_ui_features,
    )
    from enterprise_twin import EnterpriseTwin
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False
```

**Added endpoint:**
```python
@app.get("/api/config")
async def get_client_config():
    """Return UI feature flags to frontend."""
    return {
        "features": get_ui_features(),
        "tier": enterprise_cfg('deployment.tier', 'full'),
        "mode": enterprise_cfg('deployment.mode', 'personal'),
        "memory_enabled": memory_enabled(),
    }
```

**Modified startup:**
```python
@app.on_event("startup")
async def startup_event():
    if ENTERPRISE_AVAILABLE and is_enterprise_mode():
        engine = EnterpriseTwin(data_dir=settings.data_dir)
    else:
        engine = CogTwin(settings.data_dir)
```

### 3. `frontend/src/lib/stores/config.ts` (NEW)

Svelte store for frontend feature flags:
```typescript
export interface AppConfig {
    features: {
        swarm_loop: boolean;
        memory_space_3d: boolean;
        chat_basic: boolean;
        dark_mode: boolean;
        analytics_dashboard: boolean;
    };
    tier: 'basic' | 'advanced' | 'full';
    mode: 'personal' | 'enterprise';
    memory_enabled: boolean;
}

// Derived stores for easy access
export const showSwarm = derived(config, $config => $config.features.swarm_loop);
export const showMemorySpace = derived(config, $config => $config.features.memory_space_3d);
export const showAnalytics = derived(config, $config => $config.features.analytics_dashboard);
```

### 4. `frontend/src/lib/stores/index.ts` (Added exports)

```typescript
export { config, configLoading, loadConfig, isEnterpriseMode, isBasicTier,
         isMemoryEnabled, showSwarm, showMemorySpace, showAnalytics } from './config';
```

### 5. `frontend/src/routes/+layout.svelte` (Config loading on mount)

```svelte
<script lang="ts">
    import { loadConfig, configLoading } from '$lib/stores/config';

    onMount(async () => {
        const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        await loadConfig(apiBase);
    });
</script>

{#if $configLoading}
    <div class="loading-screen">...</div>
{:else}
    <slot />
{/if}
```

### 6. `frontend/src/routes/+page.svelte` (Conditional panel rendering)

```svelte
import { showSwarm, showMemorySpace, showAnalytics } from '$lib/stores/config';

{#if $showMemorySpace}
<FloatingPanel panelId="memory3d" title="Memory Space" icon="🧠">
    <MemoryCanvas ... />
</FloatingPanel>
{/if}

{#if $showAnalytics}
<FloatingPanel panelId="analytics" title="Cognitive State" icon="📊">
    <AnalyticsDashboard />
</FloatingPanel>
{/if}

{#if $showSwarm}
<FloatingPanel panelId="swarm" title="Swarm Dashboard" icon="🐝">
    <SwarmPanel />
</FloatingPanel>
{/if}
```

---

## Configuration Switching

### To Run Enterprise Mode (Driscoll)

1. Ensure `enterprise_config.yaml` exists (rename from `.bak` if needed)
2. Config loader auto-detects it (priority: enterprise_config.yaml > config.yaml)
3. Backend returns: `{ mode: "enterprise", memory_enabled: false, tier: "basic" }`
4. Frontend hides Swarm, Memory Space, Analytics panels

### To Run Personal Mode (Full CogTwin)

1. Rename `enterprise_config.yaml` → `enterprise_config.yaml.bak`
2. Config loader falls through to `config.yaml`
3. Backend returns: `{ mode: "personal", memory_enabled: true, tier: "full" }`
4. Frontend shows all panels

### Quick Commands

```bash
# Enterprise mode
mv enterprise_config.yaml.bak enterprise_config.yaml

# Personal mode
mv enterprise_config.yaml enterprise_config.yaml.bak

# Restart backend (kill old process first)
netstat -ano | findstr ":8000"  # Find PID
taskkill /F /PID <pid>
cd C:\Users\mthar\projects\cog_twin
.venv311\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend --port 8000
```

---

## Tier System

| Tier | memory_pipelines | context_stuffing | UI Features |
|------|-----------------|------------------|-------------|
| **basic** | false | true | chat only |
| **advanced** | true | false | chat + memory |
| **full** | true | true | everything |

```yaml
TIER_PRESETS = {
    'basic': {
        'features.memory_pipelines': False,
        'features.context_stuffing': True,
        'features.ui.swarm_loop': False,
        'features.ui.memory_space_3d': False,
    },
    'advanced': {
        'features.memory_pipelines': True,
        'features.ui.swarm_loop': False,
    },
    'full': {
        'features.memory_pipelines': True,
        'features.context_stuffing': True,
        'features.ui.swarm_loop': True,
        'features.ui.memory_space_3d': True,
    },
}
```

---

## Issues Encountered & Fixes

### 1. Config Auto-Detection Priority

**Problem:** `config_loader.py` checks for `enterprise_config.yaml` before `config.yaml`, so even setting `COGTWIN_CONFIG=config.yaml` env var didn't work because it was ignored.

**Fix:** The env var IS checked first, but a stale uvicorn process was still running with old config. Kill process on port 8000 and restart.

### 2. Division Detection Bug

**Problem:** Files were being assigned to division based on filename instead of folder.

**Original (wrong):**
```python
division = parts[1].lower() if len(parts) > 1 else parts[0].lower()
```

**Fixed:**
```python
if len(parts) >= 2:
    division = parts[0].lower()  # Folder is division
```

### 3. Missing Helper Functions

**Problem:** `ImportError: cannot import name 'get_docs_dir'`

**Fix:** Added helper functions to `config_loader.py`:
- `get_docs_dir()`
- `get_max_stuffing_tokens()`
- `get_division_categories()`

### 4. venv Path Issues

**Problem:** `.venv311` had hardcoded paths to wrong project (AGI_Engine)

**Fix:** Used `python -m uvicorn` instead of direct exe path

---

## API Response Examples

### Enterprise Mode (basic tier)
```json
{
    "features": {
        "swarm_loop": false,
        "memory_space_3d": false,
        "chat_basic": true,
        "dark_mode": true
    },
    "tier": "basic",
    "mode": "enterprise",
    "memory_enabled": false
}
```

### Personal Mode (full tier)
```json
{
    "features": {
        "swarm_loop": true,
        "memory_space_3d": true,
        "chat_basic": true,
        "dark_mode": true,
        "analytics_dashboard": true
    },
    "tier": "full",
    "mode": "personal",
    "memory_enabled": true
}
```

---

## Next Steps for Driscoll Fork

1. **Multi-tenant SQL** - Wire up `enterprise_tenant.py` to MS SQL for user/division tracking
2. **Usage logging** - Track tokens per user per division
3. **Auth layer** - Domain validation (@driscollfoods.com)
4. **Voice tuning** - Customize troll/corporate voices for actual Driscoll culture
5. **Doc refresh** - Hot-reload DOCX files without restart

---

## File Locations

```
C:\Users\mthar\projects\cog_twin\
├── config.yaml                    # Personal mode config
├── enterprise_config.yaml.bak     # Enterprise config (rename to activate)
├── config_loader.py               # Config loading + helpers
├── doc_loader.py                  # NEW: DOCX loading
├── enterprise_twin.py             # NEW: Wrapper class
├── enterprise_voice.py            # Voice templates
├── enterprise_tenant.py           # Tenant context
├── backend/
│   └── app/
│       └── main.py                # Modified: /api/config + startup
├── frontend/
│   └── src/
│       ├── lib/
│       │   └── stores/
│       │       ├── config.ts      # NEW: Config store
│       │       └── index.ts       # Modified: exports
│       └── routes/
│           ├── +layout.svelte     # Modified: config loading
│           └── +page.svelte       # Modified: conditional rendering
└── manuals/
    └── Driscoll/
        └── Warehouse/
            └── *.docx             # 21 operational manuals
```

---

*Document generated after successful demo recovery. Enterprise fork is fully wired and switchable.*
