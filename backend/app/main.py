from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import tempfile
import asyncio
from pathlib import Path

from app.config import settings

import sys
sys.path.insert(0, str(settings.engine_path))

from cog_twin import CogTwin
from chat_parser_agnostic import ChatParserFactory
from ingest import IngestPipeline

from app.artifacts.parser import extract_artifacts
from app.artifacts.actions import ArtifactEmit

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Track ingest jobs
ingest_jobs: dict[str, dict] = {}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "engine_path": str(settings.engine_path),
        "timestamp": datetime.utcnow().isoformat()
    }


# ===== ANALYTICS ENDPOINTS (visible AI self-awareness) =====

@app.get("/api/analytics")
async def get_analytics():
    """
    Get comprehensive session analytics.

    This is the SaaS differentiator - executives love dashboards.
    Shows cognitive patterns, phase distribution, suggestions.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    return engine.get_session_analytics()


@app.get("/api/analytics/cognitive-state")
async def get_cognitive_state():
    """Get current cognitive state (lightweight)."""
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    return engine.get_cognitive_state()


@app.get("/api/analytics/health-check")
async def get_health_check():
    """
    Run metacognitive health check.

    Returns insights and recommendations about system performance.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    return await engine.run_health_check()


@app.get("/api/analytics/session-stats")
async def get_session_stats():
    """Get session statistics (queries, tokens, duration)."""
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    return engine.get_session_stats()


@app.get("/api/analytics/patterns")
async def get_patterns():
    """
    Get recurring query patterns.

    Enterprise gold: "What topics do we keep revisiting?"
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    patterns = engine.mirror.archaeologist.detect_recurring_patterns()
    return {
        "patterns": [
            {
                "topic": p[0],
                "frequency": p[1],
                "recency": p[2],
            }
            for p in patterns[:10]
        ],
        "total_clusters": len(engine.mirror.archaeologist.query_clusters),
        "query_entropy": engine.mirror.archaeologist.calculate_query_entropy(),
    }


@app.get("/api/analytics/hotspots")
async def get_hotspots():
    """
    Get memory hotspots (most frequently accessed).

    Shows where attention is concentrated.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    hotspots = engine.mirror.thermodynamics.detect_hotspots(top_k=20)
    bursts = engine.mirror.thermodynamics.detect_bursts()

    return {
        "hotspots": [
            {"memory_id": mid, "temperature": temp}
            for mid, temp in hotspots
        ],
        "bursts": [
            {"memory_id": mid, "intensity": intensity}
            for mid, intensity in bursts[:10]
        ],
        "access_entropy": engine.mirror.thermodynamics.calculate_access_entropy(),
    }


@app.get("/api/analytics/drift")
async def get_drift():
    """
    Get semantic drift analysis.

    Detects if topics are shifting, expanding, or collapsing.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    drift_mag, drift_signal = engine.mirror.archaeologist.detect_semantic_drift()

    return {
        "magnitude": drift_mag,
        "signal": drift_signal.value,
        "interpretation": {
            "topic_shift": "Natural evolution to new subjects",
            "semantic_expansion": "Vocabulary growing - exploring new areas",
            "semantic_collapse": "Vocabulary narrowing - may indicate fixation",
            "temporal_drift": "Time-based pattern changes",
            "structural_drift": "Access pattern changes",
            "anomalous_spike": "Sudden unexplained changes",
        }.get(drift_signal.value, "Unknown"),
    }


@app.get("/api/analytics/predictions")
async def get_predictions():
    """
    Get prediction performance metrics.

    Shows how well we're anticipating user needs.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    perf = engine.mirror.prefetcher.calculate_prediction_performance()

    return {
        "accuracy_mean": perf.get("accuracy_mean", 0),
        "accuracy_std": perf.get("accuracy_std", 0),
        "total_predictions": perf.get("total_predictions", 0),
        "validated_predictions": perf.get("validated_predictions", 0),
    }


@app.get("/")
async def root():
    return {
        "message": "CogTwin UI Backend",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/{session_id}",
        "upload": "/upload"
    }


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"[WS] Session {session_id} connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"[WS] Session {session_id} disconnected. Active: {len(self.active_connections)}")
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)


manager = ConnectionManager()

# Global CogTwin engine instance
engine: CogTwin | None = None


@app.on_event("startup")
async def startup_event():
    global engine
    print("[STARTUP] Initializing CogTwin engine...")
    engine = CogTwin(settings.data_dir)
    print(f"[STARTUP] CogTwin ready. Process memories: {len(engine.retriever.process.nodes)}")


async def run_ingest(job_id: str, filepath: Path, provider: str):
    """Background task to run ingestion pipeline."""
    global engine
    
    try:
        ingest_jobs[job_id]["status"] = "parsing"
        
        # Parse the file
        factory = ChatParserFactory()
        conversations = factory.parse(str(filepath), provider=provider)
        stats = factory.get_stats()
        
        ingest_jobs[job_id]["parsed"] = len(conversations)
        ingest_jobs[job_id]["provider"] = stats.get("provider", provider)
        ingest_jobs[job_id]["status"] = "ingesting"
        
        if not conversations:
            ingest_jobs[job_id]["status"] = "complete"
            ingest_jobs[job_id]["error"] = "No conversations found"
            return
        
        # Run ingest pipeline
        pipeline = IngestPipeline(
            output_dir=settings.data_dir,
            embedding_batch_size=32,
            embedding_concurrency=8,
        )
        
        await pipeline.run([filepath], provider=provider)
        
        ingest_jobs[job_id]["status"] = "reloading"
        
        # Reload engine with new data
        engine = CogTwin(settings.data_dir)
        
        ingest_jobs[job_id]["status"] = "complete"
        ingest_jobs[job_id]["nodes"] = len(engine.retriever.process.nodes)
        
    except Exception as e:
        ingest_jobs[job_id]["status"] = "error"
        ingest_jobs[job_id]["error"] = str(e)
        print(f"[INGEST] Error: {e}")
    
    finally:
        # Cleanup temp file
        try:
            filepath.unlink()
        except:
            pass


@app.post("/upload")
async def upload_chat_log(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    provider: str = "auto"
):
    """
    Upload a chat log file for ingestion.
    
    Supported providers: anthropic, openai, grok, gemini, auto
    Supported formats: .json, .zip
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".json", ".zip"]:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use .json or .zip")
    
    # Validate provider
    valid_providers = ["auto", "anthropic", "openai", "grok", "gemini"]
    if provider not in valid_providers:
        raise HTTPException(400, f"Invalid provider: {provider}. Options: {valid_providers}")
    
    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")
    
    # Create job
    job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ingest_jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "provider": provider,
        "started": datetime.utcnow().isoformat(),
        "parsed": 0,
        "nodes": 0,
        "error": None
    }
    
    # Start background ingest
    background_tasks.add_task(run_ingest, job_id, tmp_path, provider)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Ingesting {file.filename} with provider={provider}"
    }


@app.get("/upload/status/{job_id}")
async def get_ingest_status(job_id: str):
    """Get status of an ingest job."""
    if job_id not in ingest_jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    return ingest_jobs[job_id]


@app.get("/upload/status")
async def list_ingest_jobs():
    """List all ingest jobs."""
    return ingest_jobs


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif msg_type == "message":
                content = data.get("content", "")

                if engine is None:
                    await websocket.send_json({"type": "error", "message": "Engine not initialized"})
                    continue

                # Stream chunks directly from engine as they arrive
                response_text = ""
                async for chunk in engine.think(content):
                    if isinstance(chunk, str) and chunk:
                        response_text += chunk
                        # Stream each chunk immediately (no artificial delay)
                        await websocket.send_json({
                            "type": "stream_chunk",
                            "content": chunk,
                            "done": False
                        })

                # Send done signal
                await websocket.send_json({
                    "type": "stream_chunk",
                    "content": "",
                    "done": True
                })

                # Parse artifacts from complete response
                clean_text, artifacts = extract_artifacts(response_text)

                # Emit each artifact
                for artifact in artifacts:
                    await websocket.send_json({
                        "type": "artifact_emit",
                        "artifact": artifact.model_dump(),
                        "suggested": False
                    })

                # Send full session analytics (visible AI self-awareness)
                try:
                    analytics = engine.get_session_analytics()
                    await websocket.send_json({
                        "type": "session_analytics",
                        **analytics
                    })
                except Exception as e:
                    # Fallback to basic cognitive state
                    mirror = engine.mirror
                    await websocket.send_json({
                        "type": "cognitive_state",
                        "phase": getattr(mirror.seismograph, 'current_phase', 'idle'),
                        "temperature": 0.5,
                        "error": str(e)
                    })
            
            elif msg_type == "commit":
                await websocket.send_json({
                    "type": "commit_ack",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"[WS] Error in session {session_id}: {e}")
        manager.disconnect(session_id)