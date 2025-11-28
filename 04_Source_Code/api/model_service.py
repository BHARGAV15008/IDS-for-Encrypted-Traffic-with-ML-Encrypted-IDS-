from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title="Model Service", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, Any]:

    return {"status": "ok", "service": "model"}


