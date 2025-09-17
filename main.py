"""
Main entry point for the FastAPI traffic prediction backend.
Initializes app, routers, middleware, and logging.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import traffic, recommendations
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Traffic Prediction API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://trafficpredictorfrontend.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(traffic.router)
app.include_router(recommendations.router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Traffic Prediction API is running."}