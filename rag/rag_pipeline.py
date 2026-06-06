"""
rag/rag_pipeline.py
────────────────────
Phase 3 — RAG (Retrieval-Augmented Generation)
يشتغل قبل SFT وقبل أي agent — أسرع حاجة تتشحن وبدون retraining.

Architecture
────────────
1. EmbeddingEngine   — يحوّل النص لـ vector (sentence-transformers أو OpenAI)
2. VectorStore       — يخزن الـ vectors (Pinecone cloud أو Chroma local)
3. ClinicalRetriever — يجيب أقرب N documents لأي سؤال
4. RAGPromptBuilder  — يحشر الـ retrieved context في الـ system prompt

مصادر الـ knowledge base:
  - كل الـ DAIC-WOZ clinical assessments
  - CBT/DBT techniques من الـ SFT data
  - PHQ-8 scoring rubrics
  - Crisis resources

Usage:
    rag = RAGPipeline(vector_store="chroma", data_dir="/content/mindbridge_full")
    rag.build_index()   # مرة واحدة بس

    # عند كل user message:
    context = rag.retrieve("I feel hopeless and can't sleep", top_k=3)
    prompt  = rag.build_prompt(user_message, context)
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    text:     str
    source:   str
    doc_id:   str
    metadata: Dict


@dataclass
class RetrievedChunk:
    text:       str
    source:     str
    score:      float   # cosine similarity 0-1
    metadata:   Dict


# ── Embedding Engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Thin wrapper — tries sentence-transformers first (free, local),
    falls back to a simple TF-IDF bag-of-words for zero-dependency environments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._mode  = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._mode  = "sentence_transformers"
            logger.info(f"Embedding model: {self.model_name} (sentence-transformers)")
        except ImportError:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._model = TfidfVectorizer(max_features=4096, sublinear_tf=True)
                self._mode  = "tfidf"
                logger.info("Embedding model: TF-IDF fallback (sentence-transformers not installed)")
            except ImportError:
                self._mode = "none"
                logger.warning("No embedding backend available — RAG disabled")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._mode == "sentence_transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()
        elif self._mode == "tfidf":
            if not hasattr(self._model, "vocabulary_"):
                self._model.fit(texts)
            mat = self._model.transform(texts).toarray()
            # L2 normalise
            import numpy as np
            norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
            return (mat / norms).tolist()
        else:
            return [[0.0] * 64] * len(texts)

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ── Vector Store (Chroma local / Pinecone cloud) ──────────────────────────────

class VectorStore:
    """
    Supports two backends:
      "chroma"  — local, no API key needed, persists to disk
      "pinecone" — cloud, needs PINECONE_API_KEY env var
    """

    def __init__(self, backend: str = "chroma", persist_dir: str = "./rag_index"):
        self.backend     = backend
        self.persist_dir = persist_dir
        self._client     = None
        self._collection = None
        self._init()

    def _init(self):
        if self.backend == "chroma":
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                self._collection = self._client.get_or_create_collection(
                    name="mindbridge_clinical",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"Chroma vector store at {self.persist_dir}")
            except ImportError:
                logger.warning("chromadb not installed — using in-memory fallback")
                self.backend = "memory"
                self._docs: List[Dict] = []

        elif self.backend == "pinecone":
            try:
                from pinecone import Pinecone
                api_key = os.environ.get("PINECONE_API_KEY")
                if not api_key:
                    raise ValueError("PINECONE_API_KEY not set")
                pc = Pinecone(api_key=api_key)
                self._collection = pc.Index("mindbridge-clinical")
                logger.info("Pinecone vector store connected")
            except Exception as e:
                logger.warning(f"Pinecone failed ({e}) — falling back to in-memory")
                self.backend = "memory"
                self._docs: List[Dict] = []

        if self.backend == "memory":
            self._docs: List[Dict] = []

    def upsert(self, docs: List[Document], embeddings: List[List[float]]):
        if self.backend == "chroma":
            self._collection.upsert(
                ids=[d.doc_id for d in docs],
                embeddings=embeddings,
                documents=[d.text for d in docs],
                metadatas=[{**d.metadata, "source": d.source} for d in docs],
            )
        elif self.backend == "memory":
            for doc, emb in zip(docs, embeddings):
                self._docs.append({"doc": doc, "embedding": emb})

    def query(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Document, float]]:
        if self.backend == "chroma":
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            out = []
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1.0 - dist   # cosine distance → similarity
                doc = Document(
                    text=text,
                    source=meta.get("source", "unknown"),
                    doc_id=meta.get("doc_id", ""),
                    metadata=meta,
                )
                out.append((doc, score))
            return out

        elif self.backend == "memory":
            if not self._docs:
                return []
            import numpy as np
            q = np.array(query_embedding)
            scored = []
            for item in self._docs:
                v = np.array(item["embedding"])
                score = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
                scored.append((item["doc"], score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        return []

    def count(self) -> int:
        if self.backend == "chroma":
            return self._collection.count()
        elif self.backend == "memory":
            return len(self._docs)
        return 0


# ── Knowledge Base Builder ────────────────────────────────────────────────────

def build_knowledge_base(data_dir: str) -> List[Document]:
    """
    Converts all clinical data into Document objects for indexing.
    Sources: DAIC-WOZ records, CBT techniques, PHQ rubrics, crisis resources.
    """
    docs = []

    # ── 1. DAIC-WOZ clinical records ─────────────────────────────────────────
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from data.dataset_loader import load_all

        # Find data files (handles both direct uploads and subdirectory)
        daic_dir = data_dir
        for candidate in [data_dir,
                           os.path.join(data_dir, "data", "uploads"),
                           os.path.join(data_dir, "..")]:
            if os.path.exists(os.path.join(candidate, "train_split.csv")):
                daic_dir = candidate
                break

        records = load_all(daic_dir)
        for r in records:
            doc_id = hashlib.md5(r["text"].encode()).hexdigest()[:12]
            docs.append(Document(
                text=r["text"],
                source="daic_woz",
                doc_id=f"daic_{doc_id}",
                metadata={
                    "phq_score":  r["labels"].get("phq_score"),
                    "phq_binary": r["labels"].get("phq_binary"),
                    "split":      r.get("split", "train"),
                    "safety_flag": r.get("safety_flag", False),
                },
            ))
        logger.info(f"Indexed {len(records)} DAIC-WOZ records")
    except Exception as e:
        logger.warning(f"Could not load DAIC-WOZ: {e}")

    # ── 2. CBT / DBT technique library ───────────────────────────────────────
    cbt_docs = [
        ("Thought records (CBT)",
         "A thought record helps challenge automatic negative thoughts. "
         "Steps: (1) Identify the situation. (2) Note the automatic thought. "
         "(3) Rate emotion intensity 0-100. (4) Find evidence for and against. "
         "(5) Write a balanced alternative thought. (6) Re-rate emotion. "
         "Used for depression, anxiety, and low self-worth."),

        ("Behavioral activation (CBT)",
         "Behavioral activation counters depression-driven withdrawal. "
         "Schedule small, achievable activities — even five minutes of walking counts. "
         "Action precedes motivation in depression; waiting to feel motivated first maintains the cycle. "
         "Start with pleasurable activities, then add mastery tasks."),

        ("Distress tolerance — TIPP (DBT)",
         "TIPP skills for managing overwhelming emotions: "
         "Temperature (cold water on face activates dive reflex, slows heart rate), "
         "Intense exercise (burns off adrenaline), "
         "Paced breathing (exhale longer than inhale — 4 in, 6 out), "
         "Paired muscle relaxation (tense and release muscle groups). "
         "Use when emotion is too intense for other skills."),

        ("Mindfulness — WHAT and HOW skills (DBT)",
         "WHAT skills: Observe (notice without judging), Describe (put words on experience), "
         "Participate (fully engage). "
         "HOW skills: Non-judgmentally (drop good/bad labels), "
         "One-mindfully (one thing at a time), "
         "Effectively (do what works, not what feels right). "
         "Foundation of all DBT skills."),

        ("DEAR MAN — assertive communication (DBT)",
         "DEAR MAN for getting what you need: "
         "Describe the situation objectively, Express your feelings with 'I' statements, "
         "Assert your request clearly, Reinforce by explaining benefits, "
         "Mindful of goal (don't get distracted), Appear confident, Negotiate. "
         "Use in relationships, with family, or in healthcare settings."),

        ("Safety planning",
         "A safety plan is a prioritised list of coping strategies and resources. "
         "Components: (1) Warning signs — thoughts, images, moods that precede crisis. "
         "(2) Internal coping — things I can do alone to distract. "
         "(3) Social contacts who provide distraction. "
         "(4) People I can ask for help. "
         "(5) Professionals and agencies to contact. "
         "(6) Means restriction — remove access to lethal means. "
         "Review and update regularly with clinician."),

        ("PHQ-8 scoring guide",
         "PHQ-8 scores: 0-4 minimal, 5-9 mild, 10-14 moderate, "
         "15-19 moderately severe, 20-24 severe depression. "
         "Score ≥ 10 indicates probable major depressive disorder and warrants clinical attention. "
         "The PHQ-8 omits the suicidality item (item 9) present in the PHQ-9. "
         "Used in DAIC-WOZ dataset for depression severity labeling."),

        ("PCL-C PTSD screening",
         "PCL-C (PTSD Checklist Civilian) scores: "
         "Below 28: little or no PTSD symptoms. "
         "28-43: moderate symptoms, possible PTSD. "
         "44 and above: probable PTSD, clinical assessment recommended. "
         "Measures 17 DSM PTSD symptoms across re-experiencing, avoidance, "
         "emotional numbing, and hyperarousal clusters."),

        ("Motivational interviewing",
         "Motivational interviewing (MI) enhances intrinsic motivation for change. "
         "Core spirit: Partnership, Acceptance, Compassion, Evocation. "
         "OARS skills: Open questions, Affirmations, Reflective listening, Summaries. "
         "Roll with resistance — never argue for change. "
         "Develop discrepancy between current behaviour and the patient's own values. "
         "Especially effective for substance use, medication adherence, lifestyle change."),

        ("Empathic validation",
         "Validation communicates that a person's emotions make sense given their experience. "
         "Levels of validation (Linehan): (1) Listen, (2) Reflect accurately, "
         "(3) Articulate unspoken feelings, (4) Validate in terms of history, "
         "(5) Validate as reasonable response to current situation, "
         "(6) Radical genuineness — treat person as capable. "
         "Always validate before problem-solving or giving advice."),
    ]

    for title, text in cbt_docs:
        doc_id = hashlib.md5(title.encode()).hexdigest()[:12]
        docs.append(Document(
            text=f"[CLINICAL TECHNIQUE: {title}]\n{text}",
            source="cbt_dbt_library",
            doc_id=f"tech_{doc_id}",
            metadata={"category": "technique", "title": title},
        ))

    # ── 3. Crisis resources ───────────────────────────────────────────────────
    crisis_docs = [
        ("International crisis resources",
         "Global crisis lines: "
         "USA: 988 Suicide & Crisis Lifeline (call/text 988), Crisis Text Line (text HOME to 741741). "
         "UK: Samaritans 116 123, Shout (text SHOUT to 85258). "
         "Egypt / Arab world: Amal (أمل) +20 762 1602, "
         "Befrienders.org lists local crisis lines in 32 countries. "
         "Universal: findahelpline.com — international directory. "
         "Emergency services: call local emergency number immediately if in immediate danger."),
    ]
    for title, text in crisis_docs:
        doc_id = hashlib.md5(title.encode()).hexdigest()[:12]
        docs.append(Document(
            text=f"[CRISIS RESOURCE: {title}]\n{text}",
            source="crisis_resources",
            doc_id=f"crisis_{doc_id}",
            metadata={"category": "crisis"},
        ))

    logger.info(f"Knowledge base: {len(docs)} total documents")
    return docs


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Main RAG orchestrator.
    Call build_index() once, then retrieve() on every user message.
    """

    def __init__(
        self,
        vector_store: str = "chroma",
        data_dir: str = ".",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./rag_index",
    ):
        self.data_dir  = data_dir
        self.embedder  = EmbeddingEngine(model_name=embedding_model)
        self.store     = VectorStore(backend=vector_store, persist_dir=persist_dir)
        self._indexed  = False

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Build the vector index from all clinical knowledge.
        Skips if index already exists unless force_rebuild=True.
        """
        if not force_rebuild and self.store.count() > 0:
            logger.info(f"Index already has {self.store.count()} docs — skipping rebuild")
            self._indexed = True
            return self.store.count()

        logger.info("Building RAG index...")
        docs = build_knowledge_base(self.data_dir)

        # Batch embed (sentence-transformers is faster in batches)
        BATCH = 64
        for i in range(0, len(docs), BATCH):
            batch = docs[i:i + BATCH]
            texts = [d.text for d in batch]
            embeddings = self.embedder.embed(texts)
            self.store.upsert(batch, embeddings)
            if i % 256 == 0:
                logger.info(f"  Indexed {min(i+BATCH, len(docs))}/{len(docs)} docs")

        self._indexed = True
        logger.info(f"RAG index built: {self.store.count()} documents")
        return self.store.count()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.25,
        exclude_safety_flagged: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most relevant documents for a query.
        Returns chunks filtered by min_score.
        """
        if not self._indexed and self.store.count() == 0:
            logger.warning("RAG index empty — call build_index() first")
            return []

        query_emb = self.embedder.embed_one(query)
        results   = self.store.query(query_emb, top_k=top_k * 2)  # over-fetch then filter

        chunks = []
        for doc, score in results:
            if score < min_score:
                continue
            if exclude_safety_flagged and doc.metadata.get("safety_flag"):
                continue
            chunks.append(RetrievedChunk(
                text=doc.text,
                source=doc.source,
                score=round(score, 3),
                metadata=doc.metadata,
            ))
            if len(chunks) >= top_k:
                break

        return chunks

    def build_prompt(
        self,
        user_message: str,
        retrieved: List[RetrievedChunk],
        max_context_chars: int = 1200,
    ) -> str:
        """
        Build the augmented system prompt by injecting retrieved context.
        Returns the context block to prepend to the system prompt.
        """
        if not retrieved:
            return ""

        lines = ["[RETRIEVED CLINICAL CONTEXT]"]
        total = 0
        for chunk in retrieved:
            snippet = chunk.text[:400]
            entry   = f"• [{chunk.source} | relevance={chunk.score:.2f}] {snippet}"
            if total + len(entry) > max_context_chars:
                break
            lines.append(entry)
            total += len(entry)

        lines.append("[END CONTEXT]")
        lines.append("Use the above context to inform your response when relevant. "
                     "Do not cite it directly — integrate naturally.")
        return "\n".join(lines)

    def query_with_prompt(
        self,
        user_message: str,
        top_k: int = 3,
    ) -> Tuple[List[RetrievedChunk], str]:
        """
        Convenience method: retrieve + build prompt in one call.
        Returns (chunks, context_block).
        """
        chunks = self.retrieve(user_message, top_k=top_k)
        prompt = self.build_prompt(user_message, chunks)
        return chunks, prompt


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    DATA_DIR = os.environ.get("DATA_DIR", "/mnt/user-data/uploads")
    PERSIST  = "/tmp/mindbridge_rag_test"

    print("=" * 55)
    print("  RAG Pipeline — Smoke Test")
    print("=" * 55)

    rag = RAGPipeline(
        vector_store="memory",   # no chromadb needed for test
        data_dir=DATA_DIR,
        persist_dir=PERSIST,
    )

    n = rag.build_index()
    print(f"\n  Indexed {n} documents\n")

    queries = [
        "I feel hopeless and can't get out of bed",
        "What does a PHQ-8 score of 14 mean?",
        "I've been having panic attacks",
        "I want to hurt myself",
    ]

    for q in queries:
        chunks, ctx = rag.query_with_prompt(q, top_k=2)
        print(f"Query: '{q}'")
        for c in chunks:
            print(f"  [{c.score:.2f}] {c.source}: {c.text[:80]}...")
        print()

    print("✅ RAG pipeline ready")
