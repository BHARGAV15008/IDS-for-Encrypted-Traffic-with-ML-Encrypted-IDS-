#!/bin/bash
# P22 Encrypted Traffic IDS - Docker Entrypoint Script
# Handles different deployment modes and initialization

set -e

# Default values
MODE=${1:-api}
LOG_LEVEL=${LOG_LEVEL:-INFO}
WORKERS=${WORKERS:-4}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] P22-IDS:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Function to wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    info "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" > /dev/null 2>&1; then
            log "$service_name is ready!"
            return 0
        fi
        sleep 1
    done
    
    error "Timeout waiting for $service_name at $host:$port"
    return 1
}

# Function to check model files
check_model_files() {
    local model_path=${MODEL_PATH:-/app/models/hybrid_model.pth}
    
    if [[ ! -f "$model_path" ]]; then
        warn "Model file not found at $model_path"
        warn "Please ensure the model is properly mounted or available"
        return 1
    fi
    
    log "Model file found at $model_path"
    return 0
}

# Function to initialize directories
init_directories() {
    local dirs=("/app/logs" "/app/cache" "/app/tmp")
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done
    
    # Set permissions
    chmod 755 /app/logs /app/cache /app/tmp
}

# Function to validate environment
validate_environment() {
    log "Validating environment..."
    
    # Check Python version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log "Python version: $python_version"
    
    # Check required environment variables
    local required_vars=("ENVIRONMENT")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            warn "Environment variable $var is not set"
        else
            log "$var=${!var}"
        fi
    done
    
    # Check GPU availability
    if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        log "GPU check completed"
    else
        warn "Could not check GPU availability"
    fi
}

# Function to run database migrations (if needed)
run_migrations() {
    if [[ -f "/app/migrations/migrate.py" ]]; then
        log "Running database migrations..."
        python /app/migrations/migrate.py
    fi
}

# Function to start API server
start_api() {
    log "Starting P22 IDS API server..."
    
    # Wait for dependencies
    if [[ -n "$REDIS_URL" ]]; then
        redis_host=$(echo "$REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
        redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\).*/\1/p')
        wait_for_service "$redis_host" "$redis_port" "Redis" 30
    fi
    
    # Check model files
    if ! check_model_files; then
        error "Model validation failed"
        exit 1
    fi
    
    # Start the API server
    exec uvicorn 04_Source_Code.api.threat_detection_api:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')" \
        --access-log \
        --loop uvloop \
        --http httptools
}

# Function to start batch processor
start_batch() {
    log "Starting P22 IDS batch processor..."
    
    # Wait for dependencies
    if [[ -n "$REDIS_URL" ]]; then
        redis_host=$(echo "$REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
        redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\).*/\1/p')
        wait_for_service "$redis_host" "$redis_port" "Redis" 30
    fi
    
    # Start batch processing
    exec python -m 04_Source_Code.batch.batch_processor
}

# Function to start model trainer
start_trainer() {
    log "Starting P22 IDS model trainer..."
    
    # Check if training data exists
    if [[ ! -d "/app/data/processed" ]] || [[ -z "$(ls -A /app/data/processed)" ]]; then
        error "No training data found in /app/data/processed"
        exit 1
    fi
    
    # Start training
    exec python -m 03_Models.02_Training_Scripts.train_hybrid_model \
        --config /app/config/training_config.yaml \
        --data /app/data/processed \
        --output /app/models
}

# Function to run data preprocessing
start_preprocessing() {
    log "Starting data preprocessing..."
    
    # Check if raw data exists
    if [[ ! -d "/app/data/raw" ]] || [[ -z "$(ls -A /app/data/raw)" ]]; then
        error "No raw data found in /app/data/raw"
        exit 1
    fi
    
    # Start preprocessing
    exec python -m 01_Data.scripts.preprocess_datasets \
        --input /app/data/raw \
        --output /app/data/processed
}

# Function to run evaluation
start_evaluation() {
    log "Starting model evaluation..."
    
    # Check if model and test data exist
    if ! check_model_files; then
        error "Model validation failed"
        exit 1
    fi
    
    if [[ ! -d "/app/data/processed/test" ]]; then
        error "No test data found in /app/data/processed/test"
        exit 1
    fi
    
    # Start evaluation
    exec python -m 05_Evaluation.01_Metrics_Calculators.p22_kpi_calculator \
        --model-path "${MODEL_PATH:-/app/models/hybrid_model.pth}" \
        --test-data /app/data/processed/test \
        --output /app/results
}

# Function to run shell
start_shell() {
    log "Starting interactive shell..."
    exec /bin/bash
}

# Main execution logic
main() {
    log "P22 Encrypted Traffic IDS - Starting in $MODE mode"
    
    # Initialize
    init_directories
    validate_environment
    
    # Handle different modes
    case "$MODE" in
        "api")
            start_api
            ;;
        "batch")
            start_batch
            ;;
        "trainer"|"train")
            start_trainer
            ;;
        "preprocess")
            start_preprocessing
            ;;
        "evaluate"|"eval")
            start_evaluation
            ;;
        "shell"|"bash")
            start_shell
            ;;
        *)
            error "Unknown mode: $MODE"
            error "Available modes: api, batch, trainer, preprocess, evaluate, shell"
            exit 1
            ;;
    esac
}

# Trap signals for graceful shutdown
trap 'log "Received shutdown signal, exiting..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"
