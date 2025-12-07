from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any

app = FastAPI(title="Data Service", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, Any]:

    return {"status": "ok", "service": "data"}


@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)) -> Dict[str, Any]:

    # Placeholder: route to CSV ingestion service
    return {"message": "CSV received", "filename": file.filename}


@app.post("/ingest/pcap")
async def ingest_pcap(file: UploadFile = File(...)) -> Dict[str, Any]:

    # Placeholder: route to PCAP ingestion service
    return {"message": "PCAP received", "filename": file.filename}


