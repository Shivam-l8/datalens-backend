from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from .eda import run_automated_eda
from .schemas import AnalysisResult, ErrorResponse


app = FastAPI(
    title="DataLens - Automated Insight Generation API",
    description="MVP backend for automated EDA and insight generation.",
    version="0.1.0",
)

# CORS: allow local frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def _validate_file(upload: UploadFile) -> None:
    filename = upload.filename or ""
    ext = filename.lower().rsplit(".", 1)
    ext = f".{ext[1]}" if len(ext) == 2 else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a CSV or Excel file.",
        )


@app.post(
    "/analyze",
    response_model=AnalysisResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analyze_dataset(file: UploadFile = File(...)) -> AnalysisResult:
    """
    Accept a dataset (CSV/XLSX), perform automated EDA, and return
    structured outputs including summary stats, chart data, and insights.
    """
    _validate_file(file)

    # Enforce simple size check (FastAPI may already check, but we double-guard)
    # We read into memory because we avoid persisting user data on disk.
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.2f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        # Reconstruct a buffer for pandas
        import io

        buffer = io.BytesIO(raw)
        ext = file.filename.lower().rsplit(".", 1)[-1] if file.filename else ""

        if ext == "csv":
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = run_automated_eda(df)
        return result
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Do not log the data or content; only the error message.
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze dataset: {exc}",
        ) from exc


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok"})


