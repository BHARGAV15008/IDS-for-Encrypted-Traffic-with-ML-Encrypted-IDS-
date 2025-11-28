"""
Real-Time Threat Detection API for P22 IDS

This module implements a FastAPI-based real-time threat detection service
for encrypted network traffic analysis with sub-100ms latency requirements.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import torch
import numpy as np
import asyncio
import logging
import time
from datetime import datetime
import json
import redis
from contextlib import asynccontextmanager
import uvicorn

# Import P22 modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.architectures.hybrid_cnn_lstm import HybridCNNLSTM
from features.novel_feature_modules.tls_entropy_analyzer import TLSEntropyAnalyzer
from features.novel_feature_modules.temporal_invariant_features import TemporalInvariantExtractor


# Pydantic models for API
class PacketData(BaseModel):
    """Individual packet data structure."""
    timestamp: float
    size: int
    direction: str = Field(..., regex="^(client_to_server|server_to_client)$")
    payload_hash: Optional[str] = None
    protocol: str = "TCP"


class FlowData(BaseModel):
    """Network flow data structure."""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packets: List[PacketData]
    tls_info: Optional[Dict] = None
    metadata: Optional[Dict] = None


class ThreatDetectionRequest(BaseModel):
    """Threat detection request structure."""
    flows: List[FlowData]
    detection_mode: str = Field(default="real_time", regex="^(real_time|batch|detailed)$")
    include_features: bool = False
    include_explanations: bool = False


class ThreatPrediction(BaseModel):
    """Threat prediction result."""
    flow_id: str
    threat_score: float = Field(..., ge=0.0, le=1.0)
    threat_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., regex="^(low|medium|high|critical)$")
    features: Optional[Dict] = None
    explanation: Optional[Dict] = None


class ThreatDetectionResponse(BaseModel):
    """Threat detection response structure."""
    request_id: str
    timestamp: datetime
    processing_time_ms: float
    predictions: List[ThreatPrediction]
    system_status: Dict[str, Union[str, float]]


class SystemHealth(BaseModel):
    """System health status."""
    status: str
    model_loaded: bool
    cache_status: str
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_available: bool
    requests_processed: int
    average_latency_ms: float


# Global variables for model and components
model_manager = None
feature_extractors = None
redis_client = None
system_stats = {
    "requests_processed": 0,
    "total_processing_time": 0.0,
    "start_time": time.time()
}


class ModelManager:
    """Manages model loading, inference, and caching."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize model manager.
        
        Args:
            model_path: Path to trained model
            device: Device for inference (auto, cpu, cuda)
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.class_names = [
            "Normal", "DoS", "Probe", "U2R", "R2L", 
            "Malware", "Botnet", "Exfiltration", "APT", "Unknown"
        ]
        self.logger = logging.getLogger(__name__)
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup inference device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    async def load_model(self):
        """Load the trained model asynchronously."""
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model architecture
            model_config = checkpoint.get('config', {})
            self.model = HybridCNNLSTM(**model_config.get('model', {}))
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    async def predict(self, features: torch.Tensor) -> Dict[str, Union[float, str, int]]:
        """
        Perform threat prediction.
        
        Args:
            features: Extracted features tensor
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            with torch.no_grad():
                # Move to device
                features = features.to(self.device)
                
                # Model inference
                start_time = time.time()
                logits = self.model(features)
                inference_time = (time.time() - start_time) * 1000
                
                # Convert to probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Get predictions
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
                threat_score = 1.0 - probabilities[0][0].item()  # 1 - normal class probability
                
                return {
                    "threat_class": self.class_names[predicted_class],
                    "threat_score": threat_score,
                    "confidence": confidence,
                    "inference_time_ms": inference_time,
                    "probabilities": probabilities[0].cpu().numpy().tolist()
                }
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise


class FeatureExtractorManager:
    """Manages feature extraction components."""
    
    def __init__(self):
        """Initialize feature extractors."""
        self.tls_analyzer = TLSEntropyAnalyzer()
        self.temporal_extractor = TemporalInvariantExtractor()
        self.logger = logging.getLogger(__name__)
    
    async def extract_features(self, flows: List[FlowData]) -> torch.Tensor:
        """
        Extract features from network flows.
        
        Args:
            flows: List of network flows
            
        Returns:
            Feature tensor for model input
        """
        try:
            # Convert flows to internal format
            flow_dicts = []
            for flow in flows:
                flow_dict = {
                    'packets': [
                        {
                            'timestamp': pkt.timestamp,
                            'size': pkt.size,
                            'direction': pkt.direction
                        }
                        for pkt in flow.packets
                    ],
                    'tls_info': flow.tls_info or {},
                    'metadata': flow.metadata or {}
                }
                flow_dicts.append(flow_dict)
            
            # Extract TLS features
            tls_features = self.tls_analyzer.extract_handshake_features(flow_dicts)
            
            # Extract temporal features
            temporal_features = self.temporal_extractor.extract_flow_invariants(flow_dicts)
            
            # Combine features
            combined_features = self._combine_features(tls_features, temporal_features)
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(combined_features.values)
            
            # Reshape for model input (add sequence dimension if needed)
            if len(feature_tensor.shape) == 2:
                feature_tensor = feature_tensor.unsqueeze(1)  # Add sequence dimension
            
            return feature_tensor
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _combine_features(self, tls_features, temporal_features):
        """Combine different feature types."""
        import pandas as pd
        
        # Ensure same number of rows
        min_rows = min(len(tls_features), len(temporal_features))
        tls_features = tls_features.iloc[:min_rows]
        temporal_features = temporal_features.iloc[:min_rows]
        
        # Combine features
        combined = pd.concat([tls_features, temporal_features], axis=1)
        
        # Handle missing values
        combined = combined.fillna(0.0)
        
        return combined


class CacheManager:
    """Manages Redis caching for performance optimization."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize cache manager."""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.logger = logging.getLogger(__name__)
            self.logger.info("Redis cache connected successfully")
        except Exception as e:
            self.logger.warning(f"Redis cache not available: {e}")
            self.redis_client = None
    
    async def get_cached_prediction(self, flow_hash: str) -> Optional[Dict]:
        """Get cached prediction result."""
        if self.redis_client is None:
            return None
        
        try:
            cached = self.redis_client.get(f"prediction:{flow_hash}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            self.logger.warning(f"Cache read failed: {e}")
        
        return None
    
    async def cache_prediction(self, flow_hash: str, prediction: Dict, ttl: int = 300):
        """Cache prediction result."""
        if self.redis_client is None:
            return
        
        try:
            self.redis_client.setex(
                f"prediction:{flow_hash}",
                ttl,
                json.dumps(prediction)
            )
        except Exception as e:
            self.logger.warning(f"Cache write failed: {e}")


# FastAPI application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global model_manager, feature_extractors, redis_client
    
    # Startup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        model_manager = ModelManager("path/to/trained/model.pth")
        await model_manager.load_model()
        
        feature_extractors = FeatureExtractorManager()
        redis_client = CacheManager()
        
        logger.info("P22 Threat Detection API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("P22 Threat Detection API shutting down")


app = FastAPI(
    title="P22 Encrypted Traffic IDS API",
    description="Real-time threat detection API for encrypted network traffic",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API token (placeholder implementation)."""
    # In production, implement proper token validation
    if credentials.credentials != "your-api-token":
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return {"user_id": "api_user"}


@app.get("/health", response_model=SystemHealth)
async def health_check():
    """System health check endpoint."""
    import psutil
    
    # Calculate average latency
    avg_latency = 0.0
    if system_stats["requests_processed"] > 0:
        avg_latency = system_stats["total_processing_time"] / system_stats["requests_processed"] * 1000
    
    return SystemHealth(
        status="healthy" if model_manager and model_manager.model else "degraded",
        model_loaded=model_manager is not None and model_manager.model is not None,
        cache_status="connected" if redis_client and redis_client.redis_client else "disconnected",
        memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
        cpu_usage_percent=psutil.cpu_percent(),
        gpu_available=torch.cuda.is_available(),
        requests_processed=system_stats["requests_processed"],
        average_latency_ms=avg_latency
    )


@app.post("/detect", response_model=ThreatDetectionResponse)
async def detect_threats(
    request: ThreatDetectionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Main threat detection endpoint.
    
    Analyzes network flows and returns threat predictions with sub-100ms latency.
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000000)}"
    
    try:
        predictions = []
        
        for flow in request.flows:
            # Check cache first
            flow_hash = _calculate_flow_hash(flow)
            cached_result = await redis_client.get_cached_prediction(flow_hash) if redis_client else None
            
            if cached_result and request.detection_mode == "real_time":
                # Use cached prediction
                prediction = ThreatPrediction(**cached_result)
            else:
                # Extract features
                features = await feature_extractors.extract_features([flow])
                
                # Get prediction
                result = await model_manager.predict(features)
                
                # Determine risk level
                risk_level = _determine_risk_level(result["threat_score"])
                
                # Create prediction object
                prediction = ThreatPrediction(
                    flow_id=flow.flow_id,
                    threat_score=result["threat_score"],
                    threat_class=result["threat_class"],
                    confidence=result["confidence"],
                    risk_level=risk_level,
                    features=features.tolist() if request.include_features else None,
                    explanation=_generate_explanation(result) if request.include_explanations else None
                )
                
                # Cache result
                if redis_client:
                    await redis_client.cache_prediction(flow_hash, prediction.dict())
            
            predictions.append(prediction)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update system statistics
        system_stats["requests_processed"] += 1
        system_stats["total_processing_time"] += processing_time / 1000
        
        # Log high-risk detections
        high_risk_flows = [p for p in predictions if p.risk_level in ["high", "critical"]]
        if high_risk_flows:
            background_tasks.add_task(_log_high_risk_detection, high_risk_flows, request_id)
        
        return ThreatDetectionResponse(
            request_id=request_id,
            timestamp=datetime.now(),
            processing_time_ms=processing_time,
            predictions=predictions,
            system_status={
                "model_status": "active",
                "cache_hit_rate": 0.85,  # Placeholder
                "queue_length": 0
            }
        )
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Detection failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/batch_detect")
async def batch_detect(
    flows: List[FlowData],
    user: dict = Depends(get_current_user)
):
    """Batch threat detection for multiple flows."""
    # Implementation for batch processing
    request = ThreatDetectionRequest(flows=flows, detection_mode="batch")
    return await detect_threats(request, BackgroundTasks(), user)


@app.get("/metrics")
async def get_metrics(user: dict = Depends(get_current_user)):
    """Get system performance metrics."""
    uptime = time.time() - system_stats["start_time"]
    
    return {
        "uptime_seconds": uptime,
        "requests_processed": system_stats["requests_processed"],
        "average_latency_ms": system_stats["total_processing_time"] / max(system_stats["requests_processed"], 1) * 1000,
        "throughput_rps": system_stats["requests_processed"] / uptime,
        "model_info": {
            "device": str(model_manager.device) if model_manager else "unknown",
            "loaded": model_manager is not None and model_manager.model is not None
        }
    }


# Utility functions
def _calculate_flow_hash(flow: FlowData) -> str:
    """Calculate hash for flow caching."""
    import hashlib
    
    flow_key = f"{flow.src_ip}:{flow.src_port}-{flow.dst_ip}:{flow.dst_port}-{len(flow.packets)}"
    return hashlib.md5(flow_key.encode()).hexdigest()


def _determine_risk_level(threat_score: float) -> str:
    """Determine risk level based on threat score."""
    if threat_score >= 0.8:
        return "critical"
    elif threat_score >= 0.6:
        return "high"
    elif threat_score >= 0.3:
        return "medium"
    else:
        return "low"


def _generate_explanation(result: Dict) -> Dict:
    """Generate explanation for prediction."""
    return {
        "reasoning": f"Classified as {result['threat_class']} with {result['confidence']:.2%} confidence",
        "key_factors": ["TLS entropy patterns", "Temporal flow characteristics"],
        "recommendation": "Monitor traffic" if result['threat_score'] < 0.5 else "Block connection"
    }


async def _log_high_risk_detection(predictions: List[ThreatPrediction], request_id: str):
    """Log high-risk detections for security monitoring."""
    logger = logging.getLogger(__name__)
    
    for prediction in predictions:
        logger.warning(
            f"HIGH RISK DETECTION - Request: {request_id}, "
            f"Flow: {prediction.flow_id}, "
            f"Class: {prediction.threat_class}, "
            f"Score: {prediction.threat_score:.3f}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "threat_detection_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
