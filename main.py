import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

BASE_DIR = Path(__file__).parent
PERSIST_DIR = BASE_DIR / "storage"
DATA_DIR = BASE_DIR / "data"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"


def build_index() -> VectorStoreIndex:
    # A stale-but-empty storage dir (crashed run, fresh clone) would make
    # load_index_from_storage fail, so require actual contents.
    if PERSIST_DIR.is_dir() and any(PERSIST_DIR.iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        return load_index_from_storage(storage_context)

    if not DATA_DIR.is_dir() or not any(DATA_DIR.iterdir()):
        raise RuntimeError(f"No documents to index: {DATA_DIR} is missing or empty")

    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    return index

'''
Recursive Chunking
from llama_index.core.node_parser import SentenceSplitter

recursive_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

index = VectorStoreIndex.from_documents(documents, transformations=[recursive_splitter])

'''

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env or the environment)")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Groq(model=LLM_MODEL, api_key=api_key, temperature=0)

    app.state.query_engine = build_index().as_query_engine()
    yield
    app.state.query_engine = None


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


@app.post("/query")
def query(req: QueryRequest):
    query_engine = getattr(app.state, "query_engine", None)
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Index is not ready")

    response = query_engine.query(req.question)
    return {"answer": str(response)}
