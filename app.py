import argparse
import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from download_models import download_required_models
from lib.pipeline_manager import PipelineManager
from lib.schema import (
    HealthResponse,
    PipelineLoadRequest,
    PipelineStatusResponse,
    WebRTCOfferRequest,
    WebRTCOfferResponse,
)
from lib.webrtc import WebRTCManager


class STUNErrorFilter(logging.Filter):
    """Filter to suppress STUN/TURN connection errors that are not critical."""

    def filter(self, record):
        # Suppress STUN  exeception that occurrs always during the stream restart
        if "Task exception was never retrieved" in record.getMessage():
            return False
        return True


logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add the filter to suppress STUN/TURN errors
stun_filter = STUNErrorFilter()
logging.getLogger("asyncio").addFilter(stun_filter)

# Set INFO level for your app modules
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("lib").setLevel(logging.INFO)
logging.getLogger("pipelines").setLevel(logging.INFO)

# Enable verbose logging for specific libraries when needed
if os.getenv("VERBOSE_LOGGING"):
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("aiortc").setLevel(logging.INFO)

# Select pipeline depending on the "PIPELINE" environment variable
PIPELINE = os.getenv("PIPELINE", None)

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        Git commit hash if available, otherwise a fallback message.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,  # 5 second timeout
            cwd=Path(__file__).parent,  # Run in the project directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown (not a git repository)"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "unknown (git error)"
    except FileNotFoundError:
        return "unknown (git not installed)"
    except Exception:
        return "unknown"


def print_version_info():
    """Print version information and exit."""
    try:
        pkg_version = version("hanzo-live")
    except Exception:
        pkg_version = "unknown"

    git_hash = get_git_commit_hash()

    print(f"hanzo-live: {pkg_version}")
    print(f"git commit: {git_hash}")


def configure_static_files():
    """Configure static file serving for production."""
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )
        logger.info(f"Serving static assets from {frontend_dist / 'assets'}")
    else:
        logger.info("Frontend dist directory not found - running in development mode")


# Global WebRTC manager instance
webrtc_manager = None
# Global pipeline manager instance
pipeline_manager = None


async def prewarm_pipeline(pipeline_id: str):
    """Background task to pre-warm the pipeline without blocking startup."""
    try:
        await asyncio.wait_for(
            pipeline_manager.load_pipeline(pipeline_id),
            timeout=300,  # 5 minute timeout for pipeline loading
        )
    except Exception as e:
        logger.error(f"Error pre-warming pipeline {pipeline_id} in background: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown events."""
    # Startup
    global webrtc_manager, pipeline_manager

    # Check hardware availability and auto-detect appropriate backend
    cpu_mode = os.getenv("SCOPE_CPU_MODE") == "1"

    # Check if we have any GPU support (CUDA or MPS)
    has_cuda = torch.cuda.is_available()
    has_mps = platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if not has_cuda and not has_mps and not cpu_mode:
        error_msg = (
            "No GPU detected. This application requires either:\n"
            "  - NVIDIA GPU with CUDA (Linux/Windows)\n"
            "  - Apple Silicon with Metal (macOS)\n"
            "Use --cpu flag to run in CPU-only mode for testing."
        )
        logger.error(error_msg)
        sys.exit(1)

    if cpu_mode:
        logger.warning("Running in CPU-only mode - performance will be limited")
    elif has_cuda:
        logger.info(f"Auto-detected CUDA: {torch.cuda.get_device_name(0)}")
    elif has_mps:
        try:
            import mlx.core as mx
            logger.info(f"Auto-detected Apple Silicon with MLX {mx.__version__}")
        except ImportError:
            logger.info("Auto-detected Apple Silicon with Metal backend")
    else:
        logger.info("Using CPU backend")

    # Download models if needed
    try:
        download_required_models()
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        sys.exit(1)

    # Initialize pipeline manager (but don't load pipeline yet)
    pipeline_manager = PipelineManager()
    logger.info("Pipeline manager initialized")

    # Pre-warm the default pipeline
    if PIPELINE is not None:
        asyncio.create_task(prewarm_pipeline(PIPELINE))

    webrtc_manager = WebRTCManager()
    logger.info("WebRTC manager initialized")

    yield

    # Shutdown
    if webrtc_manager:
        logger.info("Shutting down WebRTC manager...")
        await webrtc_manager.stop()
        logger.info("WebRTC manager shutdown complete")

    if pipeline_manager:
        logger.info("Shutting down pipeline manager...")
        pipeline_manager.unload_pipeline()
        logger.info("Pipeline manager shutdown complete")


def get_webrtc_manager() -> WebRTCManager:
    """Dependency to get WebRTC manager instance."""
    return webrtc_manager


def get_pipeline_manager() -> PipelineManager:
    """Dependency to get pipeline manager instance."""
    return pipeline_manager


app = FastAPI(
    lifespan=lifespan,
    title="Hanzo Live",
    description="A tool for running and customizing real-time, interactive generative AI pipelines and models",
    version=version("hanzo-live"),
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())


@app.get("/")
async def root():
    """Serve the frontend at the root URL."""
    frontend_dist = Path(__file__).parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        return {"message": "Scope API - Frontend not built"}

    # Serve the frontend index.html
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return {"message": "Scope API - Frontend index.html not found"}


@app.post("/api/v1/pipeline/load")
async def load_pipeline(
    request: PipelineLoadRequest,
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Load a pipeline."""
    try:
        # Convert pydantic model to dict for pipeline manager
        load_params_dict = None
        if request.load_params:
            load_params_dict = request.load_params.model_dump()

        success = await pipeline_manager.load_pipeline(
            request.pipeline_id, load_params_dict
        )
        if success:
            return {"message": "Pipeline loading initiated successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load pipeline: {pipeline_manager.error_message}",
            )
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Get current pipeline status."""
    try:
        status_info = pipeline_manager.get_status_info()
        return PipelineStatusResponse(**status_info)
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/webrtc/offer", response_model=WebRTCOfferResponse)
async def handle_webrtc_offer(
    request: WebRTCOfferRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Handle WebRTC offer and return answer."""
    try:
        # Ensure pipeline is loaded before proceeding
        if not pipeline_manager.is_loaded():
            raise HTTPException(
                status_code=400,
                detail="Pipeline not loaded. Please load pipeline first.",
            )

        return await webrtc_manager.handle_offer(request, pipeline_manager)

    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/{path:path}")
async def serve_frontend(request: Request, path: str):
    """Serve the frontend for all non-API routes (fallback for client-side routing)."""
    frontend_dist = Path(__file__).parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")

    # Check if requesting a specific file that exists
    file_path = frontend_dist / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    # Fallback to index.html for SPA routing
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Frontend index.html not found")


def open_browser_when_ready(host: str, port: int, server):
    """Open browser when server is ready, with fallback to URL logging."""
    # Wait for server to be ready
    while not getattr(server, "started", False):
        time.sleep(0.1)

    # Determine the URL to open
    url = (
        f"http://localhost:{port}"
        if host in ["0.0.0.0", "127.0.0.1"]
        else f"http://{host}:{port}"
    )

    try:
        success = webbrowser.open(url)
        if success:
            logger.info(f"🌐 Opened browser at {url}")
    except Exception:
        success = False

    if not success:
        logger.info(f"🌐 UI is available at: {url}")


def main():
    """Main entry point for the hanzo-live command."""
    parser = argparse.ArgumentParser(
        description="Hanzo Live - Real-time AI video generation and streaming"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (disable GPU acceleration)",
    )

    args = parser.parse_args()

    # Store CPU mode in environment for lifespan handler
    if args.cpu:
        os.environ["SCOPE_CPU_MODE"] = "1"

    # Handle version flag
    if args.version:
        print_version_info()
        sys.exit(0)

    # Configure static file serving
    configure_static_files()

    # Check if we're in production mode (frontend dist exists)
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    is_production = frontend_dist.exists()

    if is_production:
        # Create server instance for production mode
        config = uvicorn.Config(
            "app:app", host=args.host, port=args.port, reload=args.reload
        )
        server = uvicorn.Server(config)

        # Start browser opening thread
        browser_thread = threading.Thread(
            target=open_browser_when_ready,
            args=(args.host, args.port, server),
            daemon=True,
        )
        browser_thread.start()

        # Run the server
        try:
            server.run()
        except KeyboardInterrupt:
            pass  # Clean shutdown on Ctrl+C
    else:
        # Development mode - just run normally
        uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
