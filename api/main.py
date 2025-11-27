"""
FastAPI Application for Music Genre Classification

Main application file with FastAPI setup, CORS, middleware, and route registration.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.encoders import jsonable_encoder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models.schemas import ErrorResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Music Genre Classification API")
    logger.info("=" * 60)
    
    # Initialize model loader (lazy loading)
    try:
        from api.routes.prediction import load_model
        load_model()  # Load model on startup
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}. Will load on first request.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI application
app = FastAPI(
    title="Music Genre Classification API",
    description="API for classifying music genres from audio samples",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - Client: {request.client.host}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    # Add process time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(ErrorResponse(
            error=exc.detail,
            detail=f"HTTP {exc.status_code} error"
        ))
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors())
        ))
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ))
    )


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Music Genre Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


# Import and register routes
from api.routes import prediction, training, monitoring

app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(training.router, prefix="/api/v1", tags=["Training"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])


# Startup message
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("API is ready to accept requests")
    logger.info("API Documentation available at: /docs")
    logger.info("ReDoc available at: /redoc")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

