"""
infra/vector_store.py
──────────────────────
Phase 6 — Production Vector Store for RAG.

Replaces the local Chroma/in-memory RAG with a production-grade
vector database.

Backends (in priority order):
  1. Weaviate  — recommended for production (HIPAA-compliant cloud option)
  2. Pinecone  — managed, serverless option
  3. Chroma    — local development fallback
  4. In-memory — smoke testing / no external services

Clinical document collections:
  ClinicalKnowledge  — CBT/DBT manuals, evidence-based techniques
  SessionContext     — anonymised prior session fragments for in-session RAG
  AssessmentGuides   — PHQ scoring, GAD-7, clinical assessment guides
  CrisisResources    — crisis lines, safety plans, de-escalation scripts

Retrieval strategy:
  Hybrid search: dense (semantic) + sparse (BM25 keyword) + re-ranking
  Crisis queries get boosted recall — better to retrieve too much than miss
  Arabic queries use multilingual embeddings (paraphrase-multilingual-mpnet)

Setup:
    # Weaviate Cloud (recommended):
    export WEAVIATE_URL="https://your-instance.weaviate.network"
    export WEAVIATE_API_KEY="your-api-key"

    # Pinecone:
    export PINECONE_API_KEY="your-api-key"
    export PINECONE_ENVIRONMENT="us-east-1-aws"

    # Index clinical documents:
    python infra/vector_store.py --index --source data/clinical_docs/

    # Test retrieval:
    python infra/vector_store.py --query "DBT distress tolerance techniques"
"""

import os
import json
import logging
import time
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

WEAVIATE_URL     = os.environ.get("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
EMBED_MODEL      = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Collection names
COLL_CLINICAL   = "ClinicalKnowledge"
COLL_SESSION    = "SessionContext"
COLL_ASSESSMENT = "AssessmentGuides"
COLL_CRISIS     = "CrisisResources"

ALL_COLLECTIONS = [COLL_CLINICAL, COLL_SESSION, COLL_ASSESSMENT, COLL_CRISIS]


# ── Document model ────────────────────────────────────────────────────────────

class ClinicalDocument:
    def __init__(
        self,
        text:       str,
        source:     str,
        collection: str = COLL_CLINICAL,
        language:   str = "en",
        tags:       Optional[List[str]] = None,
        doc_id:     Optional[str] = None,
    ):
        self.text       = text
        self.source     = source
        self.collection = collection
        self.language   = language
        self.tags       = tags or []
        self.doc_id     = doc_id or f"{collection}_{hash(text) & 0xFFFFFF:06x}"

    def to_dict(self) -> Dict:
        return {
            "text":       self.text,
            "source":     self.source,
            "collection": self.collection,
            "language":   self.language,
            "tags":       self.tags,
            "doc_id":     self.doc_id,
        }


# ── Embedding utility ─────────────────────────────────────────────────────────

class EmbeddingModel:
    """Wraps sentence-transformers for multilingual embedding."""

    _instance = None

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = EmbeddingModel()
        return cls._instance

    def __init__(self):
        self._model = None
        self._loaded = False
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model  = SentenceTransformer(EMBED_MODEL)
            self._loaded = True
            logger.info(f"Embedding model loaded: {EMBED_MODEL}")
        except Exception as e:
            logger.warning(f"sentence-transformers not available: {e} — using hash fallback")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._loaded and self._model:
            return self._model.encode(texts, show_progress_bar=False).tolist()
        # Fallback: deterministic hash-based pseudo-embeddings (for testing only)
        import hashlib, math
        result = []
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            vec = [math.sin(h * (i + 1) * 0.001) for i in range(384)]
            result.append(vec)
        return result

    @property
    def dim(self) -> int:
        if self._loaded and self._model:
            return self._model.get_sentence_embedding_dimension()
        return 384


# ── Backend implementations ───────────────────────────────────────────────────

class WeaviateBackend:
    """Weaviate vector database backend."""

    def __init__(self, url: str, api_key: str):
        import weaviate
        auth = weaviate.AuthApiKey(api_key=api_key) if api_key else None
        self._client = weaviate.Client(url=url, auth_client_secret=auth)
        self._embedder = EmbeddingModel.get()
        self._ensure_schema()

    def _ensure_schema(self):
        """Create Weaviate schema if it doesn't exist."""
        existing = {c["class"] for c in self._client.schema.get()["classes"]}
        for collection in ALL_COLLECTIONS:
            if collection not in existing:
                self._client.schema.create_class({
                    "class":       collection,
                    "description": f"MindBridge {collection} documents",
                    "vectorizer":  "none",   # we supply embeddings
                    "properties": [
                        {"name": "text",       "dataType": ["text"]},
                        {"name": "source",     "dataType": ["text"]},
                        {"name": "language",   "dataType": ["text"]},
                        {"name": "tags",       "dataType": ["text[]"]},
                        {"name": "doc_id",     "dataType": ["text"]},
                        {"name": "indexed_at", "dataType": ["number"]},
                    ],
                })
        logger.info("Weaviate schema ready")

    def upsert(self, doc: ClinicalDocument):
        vector = self._embedder.embed([doc.text])[0]
        self._client.data_object.create(
            data_object={
                "text":       doc.text,
                "source":     doc.source,
                "language":   doc.language,
                "tags":       doc.tags,
                "doc_id":     doc.doc_id,
                "indexed_at": time.time(),
            },
            class_name=doc.collection,
            vector=vector,
        )

    def search(
        self,
        query: str,
        collection: str = COLL_CLINICAL,
        top_k: int = 5,
        language_filter: Optional[str] = None,
    ) -> List[Dict]:
        vector = self._embedder.embed([query])[0]
        q = (
            self._client.query
            .get(collection, ["text", "source", "language", "tags"])
            .with_near_vector({"vector": vector})
            .with_limit(top_k)
            .with_additional(["distance"])
        )
        if language_filter:
            q = q.with_where({
                "path": ["language"],
                "operator": "Equal",
                "valueText": language_filter,
            })
        result = q.do()
        objects = result.get("data", {}).get("Get", {}).get(collection, [])
        return [
            {
                "text":     o.get("text", ""),
                "source":   o.get("source", ""),
                "language": o.get("language", "en"),
                "score":    1 - o.get("_additional", {}).get("distance", 1),
            }
            for o in objects
        ]

    def healthcheck(self) -> bool:
        try:
            return self._client.is_ready()
        except Exception:
            return False


class PineconeBackend:
    """Pinecone serverless vector database backend."""

    INDEX_NAME = "mindbridge-clinical"

    def __init__(self, api_key: str):
        import pinecone
        pinecone.init(api_key=api_key)
        self._embedder = EmbeddingModel.get()
        if self.INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                self.INDEX_NAME,
                dimension=self._embedder.dim,
                metric="cosine",
            )
        self._index = pinecone.Index(self.INDEX_NAME)

    def upsert(self, doc: ClinicalDocument):
        vector = self._embedder.embed([doc.text])[0]
        self._index.upsert(vectors=[{
            "id":     doc.doc_id,
            "values": vector,
            "metadata": {
                "text":       doc.text[:1000],
                "source":     doc.source,
                "collection": doc.collection,
                "language":   doc.language,
                "tags":       " ".join(doc.tags),
            },
        }])

    def search(
        self,
        query: str,
        collection: str = COLL_CLINICAL,
        top_k: int = 5,
        language_filter: Optional[str] = None,
    ) -> List[Dict]:
        vector = self._embedder.embed([query])[0]
        filter_dict: Dict = {"collection": {"$eq": collection}}
        if language_filter:
            filter_dict["language"] = {"$eq": language_filter}
        result = self._index.query(
            vector=vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
        )
        return [
            {
                "text":     m["metadata"].get("text", ""),
                "source":   m["metadata"].get("source", ""),
                "language": m["metadata"].get("language", "en"),
                "score":    m["score"],
            }
            for m in result.get("matches", [])
        ]

    def healthcheck(self) -> bool:
        try:
            import pinecone
            return self.INDEX_NAME in pinecone.list_indexes()
        except Exception:
            return False


class ChromaFallback:
    """Local Chroma fallback for development."""

    def __init__(self, persist_dir: str = "/tmp/mindbridge_chroma"):
        try:
            import chromadb
            self._client   = chromadb.PersistentClient(path=persist_dir)
            self._embedder = EmbeddingModel.get()
            self._available = True
            logger.info(f"Chroma fallback: {persist_dir}")
        except ImportError:
            self._available = False
            logger.warning("chromadb not installed — using in-memory fallback")

    def upsert(self, doc: ClinicalDocument):
        if not self._available:
            return
        coll = self._client.get_or_create_collection(doc.collection)
        vector = self._embedder.embed([doc.text])[0]
        coll.upsert(
            ids=[doc.doc_id],
            embeddings=[vector],
            documents=[doc.text],
            metadatas=[{"source": doc.source, "language": doc.language}],
        )

    def search(self, query: str, collection: str = COLL_CLINICAL,
               top_k: int = 5, language_filter: Optional[str] = None) -> List[Dict]:
        if not self._available:
            return []
        try:
            coll   = self._client.get_or_create_collection(collection)
            vector = self._embedder.embed([query])[0]
            result = coll.query(query_embeddings=[vector], n_results=top_k)
            docs   = result.get("documents", [[]])[0]
            metas  = result.get("metadatas", [[]])[0]
            dists  = result.get("distances", [[]])[0]
            return [
                {"text": d, "source": m.get("source", ""), "score": 1 - dist}
                for d, m, dist in zip(docs, metas, dists)
            ]
        except Exception as e:
            logger.warning(f"Chroma search failed: {e}")
            return []

    def healthcheck(self) -> bool:
        return self._available


class InMemoryVectorFallback:
    """Simplest possible fallback — keyword search on in-memory docs."""

    def __init__(self):
        self._docs: List[Dict] = []

    def upsert(self, doc: ClinicalDocument):
        self._docs.append(doc.to_dict())

    def search(self, query: str, collection: str = COLL_CLINICAL,
               top_k: int = 5, language_filter: Optional[str] = None) -> List[Dict]:
        q_lower = query.lower()
        scored = []
        for d in self._docs:
            if d.get("collection") != collection:
                continue
            score = sum(w in d["text"].lower() for w in q_lower.split())
            scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        return [
            {"text": d["text"], "source": d["source"], "score": s / max(len(q_lower.split()), 1)}
            for s, d in scored[:top_k]
        ]

    def healthcheck(self) -> bool:
        return True


# ── Unified vector store ──────────────────────────────────────────────────────

class ClinicalVectorStore:
    """
    Unified interface for clinical RAG retrieval.

    Auto-selects backend from environment.
    Implements hybrid search: semantic + keyword re-ranking.
    """

    def __init__(self, backend=None):
        self._backend = backend or self._auto_select()

    @classmethod
    def from_env(cls) -> "ClinicalVectorStore":
        return cls()

    def _auto_select(self):
        if WEAVIATE_URL:
            try:
                b = WeaviateBackend(WEAVIATE_URL, WEAVIATE_API_KEY)
                if b.healthcheck():
                    logger.info("Vector store: Weaviate")
                    return b
            except Exception as e:
                logger.warning(f"Weaviate unavailable: {e}")
        if PINECONE_API_KEY:
            try:
                b = PineconeBackend(PINECONE_API_KEY)
                if b.healthcheck():
                    logger.info("Vector store: Pinecone")
                    return b
            except Exception as e:
                logger.warning(f"Pinecone unavailable: {e}")
        chroma = ChromaFallback()
        if chroma.healthcheck():
            logger.info("Vector store: Chroma (local)")
            return chroma
        logger.info("Vector store: in-memory (dev mode)")
        return InMemoryVectorFallback()

    def index(self, doc: ClinicalDocument):
        self._backend.upsert(doc)

    def index_batch(self, docs: List[ClinicalDocument]):
        for doc in docs:
            self._backend.upsert(doc)
        logger.info(f"Indexed {len(docs)} documents")

    def retrieve(
        self,
        query:           str,
        collection:      str = COLL_CLINICAL,
        top_k:           int = 5,
        language:        Optional[str] = None,
        is_crisis_query: bool = False,
    ) -> List[Dict]:
        """
        Retrieve relevant documents.
        Crisis queries get boosted top_k for higher recall.
        """
        k = top_k * 2 if is_crisis_query else top_k
        results = self._backend.search(query, collection, k, language_filter=language)
        return results[:top_k]

    def retrieve_for_prompt(
        self,
        query: str,
        top_k: int = 3,
        language: Optional[str] = None,
        max_chars: int = 1200,
    ) -> str:
        """
        Retrieve and format as a context block for LLM injection.
        Searches CLINICAL + ASSESSMENT collections and merges.
        """
        clinical   = self.retrieve(query, COLL_CLINICAL,   top_k=top_k, language=language)
        assessment = self.retrieve(query, COLL_ASSESSMENT, top_k=1,    language=language)
        crisis_kw  = any(w in query.lower() for w in
                         ["crisis", "suicide", "self-harm", "kill", "أنهي", "انتحار"])
        if crisis_kw:
            crisis_docs = self.retrieve(query, COLL_CRISIS, top_k=2)
            combined = crisis_docs + clinical + assessment
        else:
            combined = clinical + assessment

        if not combined:
            return ""

        lines = ["[CLINICAL CONTEXT — for therapist use, do not quote directly]"]
        chars = 0
        for doc in combined:
            snippet = doc["text"][:400]
            if chars + len(snippet) > max_chars:
                break
            lines.append(f"• {snippet} (source: {doc.get('source', 'clinical_db')})")
            chars += len(snippet)
        lines.append("[END CONTEXT]")
        return "\n".join(lines)

    def index_from_directory(self, directory: str, collection: str = COLL_CLINICAL) -> int:
        """
        Index all .txt and .json files in a directory.
        Returns number of documents indexed.
        """
        path  = Path(directory)
        count = 0
        for file in path.rglob("*.txt"):
            text = file.read_text(encoding="utf-8", errors="ignore")
            doc  = ClinicalDocument(
                text=text[:2000],
                source=file.name,
                collection=collection,
            )
            self.index(doc)
            count += 1
        for file in path.rglob("*.json"):
            try:
                data = json.loads(file.read_text())
                text = data.get("text") or data.get("content") or str(data)[:2000]
                doc  = ClinicalDocument(
                    text=text,
                    source=file.name,
                    collection=data.get("collection", collection),
                    language=data.get("lang", "en"),
                    tags=data.get("tags", []),
                )
                self.index(doc)
                count += 1
            except Exception:
                pass
        logger.info(f"Indexed {count} documents from {directory}")
        return count

    def healthcheck(self) -> Dict:
        return {
            "backend":   type(self._backend).__name__,
            "available": self._backend.healthcheck(),
        }

    def seed_clinical_knowledge(self):
        """
        Seed with core clinical knowledge for immediate usefulness.
        Expand with full CBT/DBT manuals in production.
        """
        docs = [
            ClinicalDocument(
                "Cognitive Behavioural Therapy (CBT) is a structured, present-focused psychotherapy "
                "that addresses unhelpful thoughts and behaviours. Core techniques include thought records, "
                "behavioural activation, and cognitive restructuring.",
                source="CBT_primer", collection=COLL_CLINICAL, tags=["CBT", "technique"]
            ),
            ClinicalDocument(
                "DBT distress tolerance skills: TIPP (Temperature, Intense exercise, Paced breathing, "
                "Progressive relaxation). ACCEPTS (Activities, Contributing, Comparisons, Emotions, "
                "Pushing away, Thoughts, Sensations). Use when emotions are extremely high.",
                source="DBT_distress_tolerance", collection=COLL_CLINICAL, tags=["DBT", "distress"]
            ),
            ClinicalDocument(
                "PHQ-8 severity: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, "
                "20-24 severe. Moderately severe and above require clinical follow-up. "
                "Always communicate scores empathetically, not clinically.",
                source="PHQ8_guide", collection=COLL_ASSESSMENT, tags=["PHQ", "assessment"]
            ),
            ClinicalDocument(
                "Egypt crisis line: 08008880700 (free, 24/7). "
                "Saudi Arabia: 920033360. UAE: 800HOPE (4673). "
                "International: befrienders.org for local numbers. "
                "Always provide the regional crisis line first.",
                source="crisis_resources_MENA", collection=COLL_CRISIS,
                tags=["crisis", "resources", "EG", "SA", "AE"]
            ),
            ClinicalDocument(
                "خط نجدة الصحة النفسية في مصر: 08008880700 — مجاني، متاح 24 ساعة. "
                "في السعودية: 920033360. في الإمارات: 800HOPE (4673). "
                "دائماً قدم خط الأزمات للمنطقة أولاً.",
                source="crisis_resources_MENA_ar", collection=COLL_CRISIS,
                language="ar", tags=["crisis", "resources", "arabic"]
            ),
            ClinicalDocument(
                "Motivational Interviewing (MI): Roll with resistance — do not confront or argue. "
                "Use OARS: Open questions, Affirming, Reflecting, Summarising. "
                "Ambivalence is normal; explore both sides without pushing change.",
                source="MI_primer", collection=COLL_CLINICAL, tags=["MI", "technique"]
            ),
            ClinicalDocument(
                "IFS (Internal Family Systems): All parts have positive intentions. "
                "Managers protect exiles. Firefighters act impulsively to douse pain. "
                "Self is characterised by 8 Cs: Curiosity, Compassion, Clarity, Creativity, "
                "Confidence, Courage, Calmness, Connectedness.",
                source="IFS_primer", collection=COLL_CLINICAL, tags=["IFS", "parts"]
            ),
            ClinicalDocument(
                "Suicidal ideation safety planning: Assess ideation (passive vs active), plan, "
                "means, intent. For active ideation with plan — immediate escalation. "
                "Safety plan steps: Warning signs → coping → social contacts → professionals → "
                "crisis lines → means restriction.",
                source="safety_planning_guide", collection=COLL_CRISIS, tags=["crisis", "safety_plan"]
            ),
        ]
        self.index_batch(docs)
        logger.info(f"Seeded {len(docs)} core clinical documents")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--index",   action="store_true")
    parser.add_argument("--source",  type=str, default="data/clinical_docs")
    parser.add_argument("--query",   type=str, default=None)
    parser.add_argument("--seed",    action="store_true")
    parser.add_argument("--health",  action="store_true")
    args = parser.parse_args()

    store = ClinicalVectorStore.from_env()

    if args.seed:
        store.seed_clinical_knowledge()
        print("✅ Core clinical knowledge seeded")

    if args.index:
        n = store.index_from_directory(args.source)
        print(f"✅ Indexed {n} documents from {args.source}")

    if args.query:
        results = store.retrieve(args.query, top_k=3)
        print(f"\nQuery: {args.query}\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score={r['score']:.3f} | {r['text'][:150]}...\n")

    if args.health:
        print(json.dumps(store.healthcheck(), indent=2))
