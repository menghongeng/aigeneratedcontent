import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from cleaning import clean_text
from features import compute_features
from model import AIDetector
from schemas import DetectRequest, DetectResponse, ErrorResponse
from utils import logger

FRONTEND_ORIGIN = "http://localhost:3000"   # exact frontend origin, never "*"
MODEL_PATH = Path(__file__).resolve().parents[2] / "datasets" / "ai_detector.joblib"
INFERENCE_TIMEOUT_SECONDS = 5               # DoS defence: no single request may hog the server
MAX_HISTORY = 1000                          # DoS defence: memory use of the history store is capped
MAX_CONCURRENT_INFERENCES = 2

inference_slots = threading.BoundedSemaphore(
    MAX_CONCURRENT_INFERENCES
)

inference_executor = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_INFERENCES
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Finish any inference already running before the server shuts down.
    await asyncio.to_thread(inference_executor.shutdown, wait=True)


app = FastAPI(title="AI Content Detector API", lifespan=lifespan)
detector = AIDetector(MODEL_PATH)           # loaded once at startup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
history = OrderedDict()                     # check_id -> result, newest last

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


#----------------------------------------------------------------------------
# ONE error format for every error: {"error": ..., "detail": ...}
#----------------------------------------------------------------------------

def error_response(status_code, error, detail):
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # only field name + message are returned, never the submitted text
    problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return error_response(422, "validation_error", problems)


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException):
    return error_response(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return error_response(429, "rate_limited", "Too many requests. Please wait a moment and try again.")


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc!r}")   # full detail stays in the log
    return error_response(500, "internal_error", "Something went wrong. Please try again.")


#----------------------------------------------------------------------------
# ROUTES
#----------------------------------------------------------------------------

def error_doc(description, error, detail):
    # one documented example per error code, so /docs shows the right example for each
    return {"model": ErrorResponse, "description": description,
            "content": {"application/json": {"example": {"error": error, "detail": detail}}}}


def run_detection(text):
    import time
    time.sleep(7)
    feats = compute_features(text)   # text is already cleaned by the route
    row = pd.DataFrame([feats])
    return detector.predict(row)


@app.post("/detect", response_model=DetectResponse,
          summary="Check a text for AI-generated content",
          description="Send some text and get back a verdict ('ai' or 'human'), the model's probability that it is AI-generated, "
                      "and a check_id you can use with GET /history/{check_id}. Limited to 10 requests per minute per IP.",
          response_description="The verdict for the submitted text.",
          responses={
              422: error_doc("Invalid input (empty, missing or over 20000 characters).", "validation_error",
                             "text: String should have at most 20000 characters"),
              429: error_doc("Too many requests from this IP address.", "rate_limited",
                             "Too many requests. Please wait a moment and try again."),
              500: error_doc("Unexpected server error. Details are logged, never returned.", "internal_error",
                             "Something went wrong. Please try again."),
              503: error_doc("Server inference capacity reached.", "http_error",
                             "Server busy. Please try again later."),
              504: error_doc("The check took longer than the 5 second limit.", "http_error",
                             "The check took too long. Try a shorter text."),
          })
@limiter.limit("10/minute")   # rate limiting defence, 10 checks per minute per IP
async def detect(request: Request, payload: DetectRequest):
    cleaned = clean_text(payload.text)   # same cleaning the model was trained on
    if cleaned == "":
        raise HTTPException(status_code=422, detail="The text is empty after cleaning.")

    # Reject excess inference jobs immediately instead of queueing them.
    if not inference_slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Server busy. Please try again later.")

    try:
        job = inference_executor.submit(run_detection, cleaned)
    except Exception:
        # Submission failed; no worker owns this slot.
        inference_slots.release()
        raise

    # The callback releases the slot only when the underlying job really
    # finishes (or is cancelled), not when an HTTP request times out.
    job.add_done_callback(lambda completed: inference_slots.release())

    try:
        # Shielding prevents wait_for from cancelling the underlying job.
        label, proba = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(job)),
            timeout=INFERENCE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.warning("Inference timed out")
        raise HTTPException(status_code=504, detail="The check took too long. Try a shorter text.")

    check_id = str(uuid.uuid4())
    history[check_id] = {"check_id": check_id, "label": label, "ai_probability": proba}
    if len(history) > MAX_HISTORY:
        history.popitem(last=False)   # drop the oldest
    logger.info(f"Detection {check_id}: {label} ({proba:.3f})")
    return history[check_id]


@app.get("/history/{check_id}", response_model=DetectResponse,
         summary="Get a previous result",
         description="Look up the result of an earlier POST /detect call using its check_id. "
                     "Results are kept in memory, so they are lost when the server restarts.",
         response_description="The stored verdict for that check_id.",
         responses={404: error_doc("No result exists for that check_id.", "http_error",
                                   "No result found for that check_id."),
                    422: error_doc("Invalid check_id.", "validation_error", "check_id: Input should be a valid string")})
async def get_history(check_id: str):
    result = history.get(check_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No result found for that check_id.")
    return result
