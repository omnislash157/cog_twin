**Done! Created embed_gpu_cloud_v2.py with crash-safe features:**

## Key Features

1. **Incremental saves every 50K chunks** - survives crashes
2. **Resume support** - --resume flag skips already-processed chunk_ids
3. **Numbered shards** - embeddings_000.parquet, embeddings_001.parquet, etc.
4. **Each shard uploaded to B2 immediately** - data safe in cloud

## Usage on Vast.ai

```bash
# Set B2 creds
export B2_KEY_ID="005723da756488b0000000002"
export B2_APP_KEY="K005n9fRnHG/Ht0vW5gkW7CMu8mrtpE"
export B2_ENDPOINT="https://s3.us-east-005.backblazeb2.com"
export B2_BUCKET="cogtwinHarvardBooks"

# Fresh run
python embed_gpu_cloud_v2.py --b2-only --batch-size 512

# Resume after crash
python embed_gpu_cloud_v2.py --b2-only --batch-size 512 --resume

# Custom checkpoint interval (default 50K)
python embed_gpu_cloud_v2.py --b2-only --batch-size 512 --checkpoint-interval 25000
```

## What happens on crash

1. Shards saved to /tmp/acidburn_checkpoints/
2. Each shard immediately uploaded to b2://cogtwinHarvardBooks/embeddings/shards/
3. On resume, loads all chunk_ids from existing shards (local + B2)
4. Skips already-processed chunks, continues where you left off

You have a 7814-chunk test file - want me to prep a command to test the clustering pipeline with that while you wait for V2?

# ACIDBURN Pipeline - Complete Playbook

> **Codename:** ACIDBURN (Pipeline 5)  
> **Purpose:** Cluster-assisted retrieval from massive datasets  
> **Budget:** ~$3-5K for full research mode  
> **Created:** December 2024  
> **Status:** Architecture proven, scripts need incremental save fix

---

## The Hack

```
FRONTIER MODEL TRAINING:     $50-100M, static, locked in weights
ACIDBURN:                    $3-5K, updatable, you own it, same data access
```

---

## Architecture Overview

```
PHASE 1: CHUNK (free, local)
    HuggingFace dataset → stream → chunk to ~2K tokens → chunks.jsonl

PHASE 2: EMBED (GPU rental, $$$)
    chunks → BGE-M3 on rented GPU → embeddings.parquet
    
PHASE 3: CLUSTER (CPU on same rental)
    embeddings → HDBSCAN → cluster_ids + centroids
    
PHASE 4: TAG (Grok API, cheap)
    Heuristics + Grok swarm → cluster_map.json with semantic labels
    
PHASE 5: UPLOAD (one-time)
    Parquets → B2 blob storage
    Stream to Supabase pgvector (batched)
    
RUNTIME: Query → Grok Router → Embedder (targeted clusters) → Results
```

---

## Platform: Vast.ai

### Account Setup

1. Go to [cloud.vast.ai](https://cloud.vast.ai/)
2. Create account
3. Add credit ($25-50 for test runs, $500+ for full runs)

### Finding Machines

**Search Filters (left sidebar):**

```
GPU Type:        RTX 4090, RTX 3090, A100 (for embedding)
GPU RAM:         24GB+ (BGE-M3 needs ~4GB, headroom for batching)
Disk Space:      50GB+ (for parquet files)
Reliability:     >95%
Docker:          Yes
```

**Sort by:** $/hr (ascending)

### Recommended Specs by Task

**EMBEDDING (GPU-bound):**

```
Budget option:   1x RTX 4090      ~$0.35/hr    25 chunks/sec
Mid option:      1x RTX 6000      ~$1.00/hr    25 chunks/sec  
Beast option:    4x A100 80GB     ~$5.00/hr    150+ chunks/sec
```

**CLUSTERING (CPU/RAM-bound):**

```
Same machine as embedding works
Need: 64+ cores, 128GB+ RAM for 1M+ vectors
The RTX 6000 Blackwell we used had 128 CPU cores, 290GB RAM - perfect
```

### Renting a Machine

1. Find machine, click **RENT**
2. Configure:
    
    ```
    Docker Image:    pytorch/pytorch:latestDisk Space:      50 GBLaunch Mode:     SSH or Jupyter
    ```
    
3. Click **RENT** to confirm
4. Wait 1-2 min for provisioning

### Accessing the Machine

**Option A: Jupyter Terminal (easier, browser-based)**

1. Click on your instance
2. Go to "Applications"
3. Click "Jupyter Terminal" → Launch Application
4. Terminal opens in browser

**Option B: SSH (traditional)**

1. Find SSH command: `ssh -p XXXXX root@XX.XX.XX.XX`
2. Connect from local terminal

### CRITICAL: Destroy vs Stop

```
STOP:     Pauses instance, STILL BILLS for storage
DESTROY:  Fully kills instance, billing stops completely

ALWAYS DESTROY WHEN DONE.
```

---

## Platform: Backblaze B2

### Account Setup

1. Go to [backblaze.com](https://backblaze.com/)
2. Create account (free tier available)
3. Create bucket: `cogtwinHarvardBooks` (or your name)

### Get Credentials

1. Account → App Keys
2. "Add a New Application Key"
3. Name it, select bucket (or All)
4. **IMMEDIATELY COPY** both values (applicationKey shown only once):
    - keyID
    - applicationKey

### Find Your Endpoint

1. Click on your bucket
2. Look for "Endpoint" - something like:
    
    ```
    s3.us-east-005.backblazeb2.com
    ```
    
3. Note the region - must match in code

### Credentials Format (in .env)

```env
B2_KEY_ID=005723da756488b0000000002
B2_APP_KEY=K005n9fRnHG/Ht0vW5gkW7CMu8mrtpE
B2_ENDPOINT=https://s3.us-east-005.backblazeb2.com
B2_BUCKET=cogtwinHarvardBooks
```

### Test Connection (Python)

```python
import boto3
from botocore.config import Config

client = boto3.client(
    's3',
    endpoint_url='https://s3.us-east-005.backblazeb2.com',  # YOUR endpoint
    aws_access_key_id='YOUR_KEY_ID',
    aws_secret_access_key='YOUR_APP_KEY',
    config=Config(signature_version='s3v4')
)
print(client.list_objects_v2(Bucket='cogtwinHarvardBooks', MaxKeys=5))
```

---

## Platform: Supabase (pgvector)

### Account Setup

1. Go to [supabase.com](https://supabase.com/)
2. Create project
3. Enable pgvector extension:
    
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```
    

### Credentials Format (in .env)

```env
SUPABASE_HOST=db.xxxxxxxxxx.supabase.co
SUPABASE_PORT=5432
SUPABASE_DB=postgres
SUPABASE_USER=postgres
SUPABASE_PASSWORD=your_password_here
```

### Connection String (built in code)

```python
from urllib.parse import quote_plus
password = quote_plus(os.environ['SUPABASE_PASSWORD'])  # handles special chars
conn_string = f"postgresql://postgres:{password}@{host}:5432/postgres"
```

### IMPORTANT: Network Restrictions

Vast.ai cannot reach Supabase directly (blocked).

**Workaround:**

1. Save parquets to B2 from Vast.ai
2. Push to Supabase from local machine (no restrictions)

---

## Scripts Overview

### `acidburn_pipeline/ingest.py` (Phase 1)

- Streams from HuggingFace
- Chunks to ~2K tokens with overlap
- Outputs: `chunks.jsonl`
- Runs locally (free)

```bash
python -m acidburn_pipeline.ingest --limit 100 --output data/chunks.jsonl
python -m acidburn_pipeline.ingest --output data/chunks.jsonl  # full
```

### `acidburn_pipeline/embed_gpu.py` (Phase 2 - simple)

- Takes chunks.jsonl input
- Embeds with sentence-transformers
- Outputs: embeddings.parquet
- For small runs, manual upload

```bash
python embed_gpu.py --input chunks.jsonl --output embeddings.parquet --batch-size 512
```

### `acidburn_pipeline/embed_gpu_cloud.py` (Phase 2 - full)

- Streams HuggingFace directly on GPU
- No local chunking needed
- Modes: `--local-only`, `--b2-only`
- **NEEDS FIX: incremental saves every 50K chunks**

```bash
# Test
python embed_gpu_cloud.py --local-only --limit 100

# Full run (after fix)
python embed_gpu_cloud.py --local-only --batch-size 512
```

### `acidburn_pipeline/cluster_cpu.py` (Phase 3)

- HDBSCAN clustering
- Computes centroids
- Outputs: clustered.parquet + centroids.parquet

```bash
pip install hdbscan
python cluster_cpu.py --input embeddings.parquet --min-cluster-size 100 --min-samples 20
```

### `acidburn_pipeline/upload_to_b2.py` (Phase 5)

- Uploads parquet files to B2 blob storage

---

## Full Run Checklist

### Pre-Flight (Local Machine)

```bash
# 1. Verify .env has all creds
cat .env | grep -E "(B2_|SUPABASE_|XAI_)"

# 2. Test B2 connection locally
python -c "from acidburn_pipeline.config import get_config; print('Config OK')"

# 3. Have scripts ready to upload:
#    - embed_gpu_cloud.py (with incremental save fix)
#    - cluster_cpu.py
#    - upload_to_b2.py
```

### Vast.ai Setup

```bash
# 1. Rent machine (see specs above)
# 2. Open Jupyter Terminal
# 3. Verify GPU
nvidia-smi

# 4. Install deps
pip install sentence-transformers numpy pyarrow tqdm psycopg2-binary boto3 datasets tiktoken hdbscan

# 5. Test model loads
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3'); print('Model loaded')"

# 6. Set environment variables (copy/paste)
export B2_KEY_ID="your_key_id"
export B2_APP_KEY="your_app_key"
export B2_ENDPOINT="https://s3.us-east-005.backblazeb2.com"
export B2_BUCKET="cogtwinHarvardBooks"

# 7. Verify B2 connection
python -c "
import boto3, os
from botocore.config import Config
client = boto3.client('s3', endpoint_url=os.environ['B2_ENDPOINT'],
    aws_access_key_id=os.environ['B2_KEY_ID'],
    aws_secret_access_key=os.environ['B2_APP_KEY'],
    config=Config(signature_version='s3v4'))
print(client.list_objects_v2(Bucket=os.environ['B2_BUCKET'], MaxKeys=5))
"

# 8. Upload scripts via Jupyter file browser
```

### Phase 2: Embedding Run

```bash
# Start embedding (with future incremental save fix)
python embed_gpu_cloud.py --local-only --batch-size 512

# For multi-GPU (if available):
python embed_gpu_cloud.py --local-only --batch-size 2048

# Monitor progress - should see:
# Embedded 512 chunks | Total: XXXXX | Rate: XX.X/sec
```

### Phase 3: Clustering (same machine)

```bash
# After embedding completes:
python cluster_cpu.py --input embeddings.parquet --min-cluster-size 100 --min-samples 20

# For 1M+ vectors, expect 1-2 hours
# Outputs: clustered.parquet, centroids.parquet
```

### Phase 4: Upload to B2

```bash
python -c "
import boto3, os
from botocore.config import Config

client = boto3.client('s3', endpoint_url=os.environ['B2_ENDPOINT'],
    aws_access_key_id=os.environ['B2_KEY_ID'],
    aws_secret_access_key=os.environ['B2_APP_KEY'],
    config=Config(signature_version='s3v4'))

# Upload embeddings
client.upload_file('clustered.parquet', os.environ['B2_BUCKET'], 'gutenberg/clustered.parquet')
client.upload_file('centroids.parquet', os.environ['B2_BUCKET'], 'gutenberg/centroids.parquet')
print('Uploaded!')
"
```

### Post-Run: DESTROY INSTANCE

```
Vast.ai → Instances → Find yours → DESTROY (not Stop)
```

---

## Cost Estimates

### Test Run (~100K chunks)

```
GPU time:     2-3 hours × $1/hr    = $3
Clustering:   included
B2 storage:   ~$0.01/month
─────────────────────────────────────
Total:        ~$5
```

### Single Dataset (Harvard 30K books, ~1.5M chunks)

```
GPU time:     17 hours × $1/hr     = $17
Clustering:   2 hours              = $2
B2 storage:   ~$0.10/month
─────────────────────────────────────
Total:        ~$20
```

### Full Harvard Corpus (983K books, ~48M chunks)

```
GPU time:     3 days × $5/hr       = $360
Clustering:   4-6 hours × $5/hr    = $30
B2 storage:   ~$1/month
Buffer/reruns:                     = $100
─────────────────────────────────────
Total:        ~$500
```

### Full Research Mode (Multiple datasets)

```
Harvard (983K books):              $500
arXiv (2.9M papers):               $600
Wikipedia:                         $800
Textbooks:                         $300
Clustering (combined):             $200
Grok tagging (~20K clusters):      $100
Buffer/reruns/testing:             $500
─────────────────────────────────────
Total:        ~$3,000-5,000
```

---

## Lessons Learned

### 1. Script Must Save Incrementally

**Problem:** Lost 1.5M chunks when Ctrl+Z killed process - script saved only at end.  
**Fix:** Save checkpoint every 50K chunks, implement resume support.

### 2. Vast.ai Network Restrictions

**Problem:** Cannot reach Supabase from Vast.ai (blocked).  
**Workaround:** Save to B2 (works), push to Supabase from local machine later.

### 3. B2 Endpoint Matters

**Problem:** Got "InvalidAccessKeyId" errors.  
**Cause:** Wrong endpoint region (us-west-004 vs us-east-005).  
**Fix:** Check bucket's actual endpoint in Backblaze dashboard.

### 4. Batch Size vs Embedding Dimension

**Clarification:** Batch size (512, 1024, 2048) is just throughput. Embedding dimension is always 1024 (BGE-M3 output). All datasets compatible regardless of batch size used.

### 5. Dataset Size Surprised Us

**Expected:** ~500K chunks for Gutenberg  
**Actual:** Harvard Institutional Books = 983K books = ~48M chunks  
**Lesson:** Check dataset size before committing to full run.

### 6. DESTROY vs STOP on Vast.ai

**STOP:** Still bills for storage  
**DESTROY:** Fully kills instance  
**Always DESTROY when done.**

---

## Future Enhancements

### Multi-GPU Support

```python
# In embed_gpu_cloud.py:
device_count = torch.cuda.device_count()
# DataParallel or split batches across GPUs
```

### Tiered Verification Architecture

```
TIER 1: Knowledge Pool (Harvard, arXiv, Wikipedia)
        → Clustered together, one embedding space
        → Source of facts

TIER 2: Verification Pool (Stack Overflow, Reddit, forums)
        → Separate, not clustered with Tier 1
        → Cross-check layer to prevent hallucination
```

### Grok Router + Quality Gate

```
Query → Grok reads cluster_map → picks target clusters
     → Embedder fetches from specific clusters
     → Grok reviews results
     → Low confidence? Tool call to check more clusters
```

---

## File Locations

```
cog_twin/
├── .env                           # All credentials (check before run)
├── acidburn_pipeline/
│   ├── ingest.py                  # Phase 1: chunking
│   ├── embed_gpu.py               # Phase 2: simple embed
│   ├── embed_gpu_cloud.py         # Phase 2: cloud embed (NEEDS FIX)
│   ├── cluster_cpu.py             # Phase 3: HDBSCAN
│   └── upload_to_b2.py            # Phase 5: B2 upload
└── data/
    └── chunks.jsonl               # Local chunks (test runs)

B2 Bucket (cogtwinHarvardBooks):
├── gutenberg/
│   ├── clustered.parquet          # Embeddings + cluster_ids
│   └── centroids.parquet          # Cluster centroids
└── arxiv/                         # Future
    └── ...

Supabase:
├── chunk_embeddings               # Main vector table
├── cluster_headers                # Centroid + labels
└── chunk_clusters                 # Membership mapping
```

---

## Quick Reference Commands

```bash
# Test B2 connection
python -c "import boto3; ..."

# Test GPU
nvidia-smi

# Test model
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3'); print('OK')"

# Check parquet
python -c "import pyarrow.parquet as pq; t = pq.read_table('embeddings.parquet'); print(f'Rows: {t.num_rows}')"

# Kill stuck process
pkill -9 python

# Check disk space
df -h

# Watch GPU usage
watch -n 1 nvidia-smi
```

---

_"Mess with the best, die like the rest." - Acid Burn, Hackers (1995)_