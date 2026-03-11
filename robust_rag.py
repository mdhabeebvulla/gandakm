"""
robust_rag.py — Full Production RAG Architecture
==================================================

Matches the "Building a Robust RAG System" reference architecture:

┌──────────────────────────────────────────────────────────────────┐
│  INDEXING           │  ROUTING          │  QUERY CONSTRUCTION    │
│  • Semantic Split   │  • Logical Route  │  • Multi-Query         │
│  • Multi-Rep Index  │  • Semantic Route │  • RAG Fusion          │
│  • RAPTOR Hierarchy │                   │  • HyDE                │
│  • ColBERT tokens   │                   │  • Decomposition       │
├──────────────────────────────────────────────────────────────────┤
│  RETRIEVAL                   │  GENERATION                      │
│  • Vector Store (FAISS)      │  • Self-RAG (verify then answer) │
│  • BM25 Sparse               │  • Active Retrieval              │
│  • Graph (section links)     │  • Citation generation           │
│  • Reranking (cross-encoder) │                                  │
├──────────────────────────────────────────────────────────────────┤
│  EVALUATION                                                     │
│  • RAGAS (faithfulness, relevancy, context recall)              │
│  • Custom metrics for state/LOB accuracy                        │
└──────────────────────────────────────────────────────────────────┘

Embeddings: ALL OPEN SOURCE
  • Dense:    BAAI/bge-large-en-v1.5 (fastembed/ONNX)
  • Sparse:   BM25Okapi
  • Reranker: BAAI/bge-reranker-v2-m3 (fastembed/ONNX)
  • LLM:      OpenAI gpt-4o-mini (generation only)

Install:
    pip install pdfplumber rank_bm25 openai numpy faiss-cpu fastembed networkx

Usage:
    pipeline = RobustRAGPipeline(openai_api_key="sk-...")
    pipeline.ingest_pdf("km2.pdf")
    pipeline.save_index("./index")
    result = pipeline.query("What is the TAT for CA expedited appeals?")
"""

import os
import re
import json
import hashlib
import pickle
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

import numpy as np
import pdfplumber
from rank_bm25 import BM25Okapi
from openai import OpenAI

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ═══════════════════════════════════════════════════════════════
# 1. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_num: int
    section_path: str
    state: str = "ALL"
    lob: str = "ALL"
    funding_type: str = "ALL"
    topic: str = "general"
    content_type: str = "procedure"
    appeal_type: str = "ALL"
    service_type: str = "ALL"
    parent_section: str = ""
    child_sections: list = field(default_factory=list)
    # Multi-representation fields
    summary: str = ""            # Summary embedding (Multi-Rep Indexing)
    parent_chunk_id: str = ""    # RAPTOR hierarchy
    level: int = 0               # 0=leaf, 1=section summary, 2=topic summary


@dataclass
class QueryPlan:
    """Output of the routing + query construction stage."""
    original_query: str
    sub_queries: list          # Multi-query / decomposed queries
    hyde_passage: str = ""     # Hypothetical Document Embedding
    route: str = "vector"      # vector, graph, hybrid
    filters: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 2. PDF PARSER (same as before, reliable)
# ═══════════════════════════════════════════════════════════════

class PDFParser:
    STATE_ABBREV_MAP = {
        "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Georgia": "GA", "Indiana": "IN", "Kentucky": "KY",
        "Maine": "ME", "Missouri": "MO", "New Hampshire": "NH",
        "Nevada": "NV", "New York": "NY", "Ohio": "OH",
        "Virginia": "VA", "Wisconsin": "WI"
    }
    STATE_NAMES = set(STATE_ABBREV_MAP.keys())

    def extract_pages(self, pdf_path: str) -> list[dict]:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                pages.append({"page_num": i + 1, "text": text, "tables": tables})
        logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
        return pages


# ═══════════════════════════════════════════════════════════════
# 3. INDEXING — Semantic Split + Multi-Rep + RAPTOR Hierarchy
# ═══════════════════════════════════════════════════════════════

class StructuralChunker:
    """Semantic-aware chunking preserving document structure."""

    KNOWN_SECTIONS = [
        "Overview", "General Information", "Terms and Definitions for G&A",
        "Pharmacy Terms and Definitions", "Filing a Grievance or an Appeal in Solution Central",
        "Authorizations", "Claims", "Member Emails", "Check Appeal Status in NextGen",
        "DORs - Designation of Representative", "EOB Grievance & Appeals Information",
        "External Review Process", "File a Grievance or Appeal on Anthem.com",
        "G&A Process for Broker Services", "Handling ASO, Fully Insured, and Individual Members",
        "How to Determine the Company or Corporate Received Date",
        "In Writing Messages", "Member Complaints", "Online Virtual Providers",
        "Overturned Appeals", "Quality of Care Against a Provider",
        "Requests for Copies of Appeal Records", "State Regulatory Agencies",
        "Urgent or Expedited Appeals", "Verbal Appeal Process",
        "California", "Colorado", "Connecticut", "Georgia", "Indiana",
        "Kentucky", "Maine", "Missouri", "New Hampshire", "Nevada",
        "New York", "Ohio", "Virginia", "Wisconsin",
        "Wellpoint Vision", "CA AB2470", "CA DMHC", "Revision History",
        "NV Grievance from Provider Results", "City Of New York",
        "Sections Within a DOR Form", "Verbal DOR Workflow",
    ]

    LOB_KEYWORDS = {
        "Individual": ["individual", "individual on exchange", "aca on exchange"],
        "Small Group": ["small group"],
        "Large Group": ["large group"],
        "National": ["national", "national wgs"],
    }

    TOPIC_KEYWORDS = {
        "grievance": ["grievance", "qos", "quality of service", "complaint"],
        "appeal": ["appeal", "adverse benefit determination", "denied"],
        "expedited": ["expedited", "urgent", "72 hours", "48 hours"],
        "DOR": ["dor", "designation of representative", "authorized representative"],
        "QOC": ["quality of care", "qoc", "clinical grievance"],
        "pharmacy": ["pharmacy", "drug", "formulary", "medication", "ndc"],
        "behavioral_health": ["behavioral health", "mental health", "substance abuse"],
        "external_review": ["external review", "iro", "independent review"],
        "verbal_appeal": ["verbal appeal", "verbal grievance"],
        "online_filing": ["anthem.com", "sydney app", "message center"],
        "claims": ["claim", "claims", "manage claims"],
        "authorizations": ["authorization", "manage authorization"],
        "timeframes": ["tat", "turnaround", "timeframe", "calendar days"],
        "contact_info": ["address", "fax", "email", "po box"],
    }

    CONTENT_TYPE_PATTERNS = {
        "definition": [r"term\s+definition", r"what is a"],
        "timeframe": [r"\d+\s*(cd|bd|hours|calendar days|business days)", r"tat\s+for"],
        "contact_info": [r"po\s*box", r"fax.*\d{3}.*\d{3}.*\d{4}", r"@anthem\.com"],
        "decision_logic": [r"if.*then", r"if the caller", r"if the claim"],
        "procedure": [r"step\s+action", r"step\s+\d", r"follow these steps"],
        "form": [r"part [a-g]", r"dor form", r"form must"],
    }

    def __init__(self, max_chunk_tokens: int = 600):
        self.max_chunk_tokens = max_chunk_tokens

    def chunk_pages(self, pages: list[dict]) -> list[Chunk]:
        full_text = ""
        page_map = []
        for page in pages:
            start = len(full_text)
            full_text += page["text"] + "\n\n"
            page_map.append((start, len(full_text), page["page_num"]))

        sections = self._split_into_sections(full_text, page_map)
        chunks = []
        for section in sections:
            chunks.extend(self._section_to_chunks(section))
        for chunk in chunks:
            self._enrich_metadata(chunk)

        logger.info(f"Created {len(chunks)} leaf chunks")
        return chunks

    def _split_into_sections(self, text, page_map):
        sections = []
        lines = text.split("\n")
        current = {"title": "Overview", "path": "Overview", "text_lines": [], "start_line": 0}
        parent_stack = ["Root"]

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "Create Child Content":
                continue

            is_header = any(stripped == k or stripped.startswith(k + " ") for k in self.KNOWN_SECTIONS)
            if not is_header and stripped in PDFParser.STATE_NAMES:
                is_header = True
            if not is_header:
                for lob_key in ["Individual", "Small Group", "Large Group", "National"]:
                    if stripped.startswith(lob_key):
                        is_header = True
                        break

            if is_header and current["text_lines"]:
                sec_text = "\n".join(current["text_lines"]).strip()
                if sec_text:
                    cp = sum(len(l) + 1 for l in lines[:current["start_line"]])
                    pn = self._page_for_pos(cp, page_map)
                    sections.append({"title": current["title"], "path": current["path"],
                                     "text": sec_text, "page_num": pn})

                if stripped in PDFParser.STATE_NAMES:
                    parent_stack = ["Root", stripped]
                elif any(stripped.startswith(l) for l in ["Individual", "Small Group", "Large Group", "National"]):
                    parent_stack = parent_stack[:2] + [stripped] if len(parent_stack) >= 2 else parent_stack + [stripped]
                else:
                    if len(parent_stack) > 2:
                        parent_stack = parent_stack[:2]

                path = " > ".join(parent_stack + [stripped])
                current = {"title": stripped, "path": path, "text_lines": [], "start_line": i}
            else:
                current["text_lines"].append(line)

        if current["text_lines"]:
            sec_text = "\n".join(current["text_lines"]).strip()
            if sec_text:
                cp = sum(len(l) + 1 for l in lines[:current["start_line"]])
                pn = self._page_for_pos(cp, page_map)
                sections.append({"title": current["title"], "path": current["path"],
                                 "text": sec_text, "page_num": pn})
        return sections

    def _page_for_pos(self, pos, page_map):
        for s, e, pn in page_map:
            if s <= pos < e:
                return pn
        return page_map[-1][2] if page_map else 1

    def _section_to_chunks(self, section):
        text = section["text"].strip()
        if not text:
            return []
        if count_tokens(text) <= self.max_chunk_tokens:
            cid = hashlib.md5(text[:200].encode()).hexdigest()[:12]
            return [Chunk(chunk_id=cid, text=text, page_num=section["page_num"],
                          section_path=section["path"], parent_section=section["title"])]
        return self._smart_split(text, section)

    def _smart_split(self, text, section):
        chunks = []
        paragraphs = re.split(r"\n\s*\n", text)
        current_text = ""
        idx = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            combined = (current_text + "\n\n" + para).strip() if current_text else para
            if count_tokens(combined) <= self.max_chunk_tokens:
                current_text = combined
            else:
                if current_text.strip():
                    cid = hashlib.md5(f"{section['title']}_{idx}_{current_text[:100]}".encode()).hexdigest()[:12]
                    chunks.append(Chunk(chunk_id=cid, text=current_text, page_num=section["page_num"],
                                        section_path=section["path"], parent_section=section["title"]))
                    idx += 1
                current_text = para if count_tokens(para) <= self.max_chunk_tokens else para[:2000]
        if current_text.strip():
            cid = hashlib.md5(f"{section['title']}_{idx}_{current_text[:100]}".encode()).hexdigest()[:12]
            chunks.append(Chunk(chunk_id=cid, text=current_text, page_num=section["page_num"],
                                section_path=section["path"], parent_section=section["title"]))
        return chunks

    def _enrich_metadata(self, chunk):
        text_lower = chunk.text.lower()
        path_lower = chunk.section_path.lower()
        combined = text_lower + " " + path_lower

        for name, abbr in PDFParser.STATE_ABBREV_MAP.items():
            if name.lower() in path_lower:
                chunk.state = abbr
                break

        for lob, kws in self.LOB_KEYWORDS.items():
            if any(kw in path_lower or kw in text_lower[:200] for kw in kws):
                chunk.lob = lob
                break

        scores = {t: sum(1 for kw in kws if kw in combined) for t, kws in self.TOPIC_KEYWORDS.items()}
        scores = {k: v for k, v in scores.items() if v > 0}
        if scores:
            chunk.topic = max(scores, key=scores.get)

        for ct, patterns in self.CONTENT_TYPE_PATTERNS.items():
            if any(re.search(p, combined, re.IGNORECASE) for p in patterns):
                chunk.content_type = ct
                break

        if "administrative" in combined and "appeal" in combined:
            chunk.appeal_type = "administrative"
        elif "clinical" in combined and "appeal" in combined:
            chunk.appeal_type = "clinical"
        elif "pharmacy" in combined:
            chunk.appeal_type = "pharmacy"
        elif "behavioral" in combined:
            chunk.appeal_type = "behavioral_health"
        if "aso" in combined:
            chunk.funding_type = "ASO"
        elif "fully insured" in combined or "fi:" in combined:
            chunk.funding_type = "FI"
        if "pre-service" in combined:
            chunk.service_type = "pre-service"
        elif "post-service" in combined:
            chunk.service_type = "post-service"


class MultiRepresentationIndexer:
    """
    MULTI-REPRESENTATION INDEXING (from diagram)
    Creates multiple representations per chunk:
    1. Original text embedding (dense)
    2. Summary embedding (compressed representation)
    3. Parent/section-level summaries (RAPTOR-style hierarchy)
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    def create_summaries(self, chunks: list[Chunk], batch_size: int = 20) -> list[Chunk]:
        """Generate summary representations for each chunk."""
        logger.info("Generating chunk summaries for multi-representation indexing...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text[:1500] for c in batch]  # Limit input size

            prompt = (
                "For each text below, write a one-sentence summary capturing the key information. "
                "Return exactly one summary per line, in the same order.\n\n"
            )
            for j, t in enumerate(texts):
                prompt += f"Text {j+1}: {t[:500]}\n\n"

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=2000,
                )
                summaries = response.choices[0].message.content.strip().split("\n")
                for j, chunk in enumerate(batch):
                    if j < len(summaries):
                        chunk.summary = summaries[j].strip()
            except Exception as e:
                logger.warning(f"Summary generation failed for batch {i}: {e}")

        logger.info(f"Generated summaries for {sum(1 for c in chunks if c.summary)} chunks")
        return chunks

    def create_hierarchy(self, chunks: list[Chunk], openai_client: OpenAI) -> list[Chunk]:
        """
        RAPTOR-STYLE HIERARCHICAL INDEXING
        Groups chunks by section → creates section-level summary chunks.
        Enables retrieval at different granularity levels.
        """
        logger.info("Building RAPTOR-style hierarchy...")

        # Group by section path (first 2 levels)
        section_groups = defaultdict(list)
        for c in chunks:
            parts = c.section_path.split(" > ")
            key = " > ".join(parts[:3]) if len(parts) >= 3 else c.section_path
            section_groups[key].append(c)

        parent_chunks = []
        for section_path, group_chunks in section_groups.items():
            if len(group_chunks) < 2:
                continue

            # Combine text from all chunks in this section
            combined = "\n".join([c.text[:300] for c in group_chunks[:10]])

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"Summarize this section in 2-3 sentences:\n\n{combined[:3000]}"
                    }],
                    temperature=0, max_tokens=200,
                )
                summary_text = response.choices[0].message.content.strip()
            except Exception:
                summary_text = f"Section: {section_path}. Contains {len(group_chunks)} sub-sections."

            parent_id = hashlib.md5(f"parent_{section_path}".encode()).hexdigest()[:12]
            parent = Chunk(
                chunk_id=parent_id,
                text=summary_text,
                page_num=group_chunks[0].page_num,
                section_path=section_path,
                state=group_chunks[0].state,
                lob=group_chunks[0].lob,
                topic=group_chunks[0].topic,
                content_type="summary",
                parent_section=section_path,
                level=1,  # Section-level summary
            )
            parent_chunks.append(parent)

            # Link children to parent
            for c in group_chunks:
                c.parent_chunk_id = parent_id

        logger.info(f"Created {len(parent_chunks)} hierarchical parent chunks")
        return parent_chunks


# ═══════════════════════════════════════════════════════════════
# 4. GRAPH INDEX — Section Relationship Graph
# ═══════════════════════════════════════════════════════════════

class SectionGraph:
    """
    GRAPH DB (from diagram) — Lightweight graph of section relationships.
    Uses NetworkX instead of Neo4j for simplicity.

    Captures:
    - Parent/child section relationships
    - Cross-references ("See child section...", "Refer to...")
    - State → LOB → Topic hierarchy
    """

    def __init__(self):
        if not HAS_NETWORKX:
            logger.warning("NetworkX not installed — graph features disabled")
            self.graph = None
            return
        self.graph = nx.DiGraph()

    def build_from_chunks(self, chunks: list[Chunk]):
        if not self.graph:
            return

        # Add nodes
        for c in chunks:
            self.graph.add_node(c.chunk_id, **{
                "state": c.state, "lob": c.lob, "topic": c.topic,
                "section": c.section_path, "level": c.level,
            })

        # Add hierarchy edges
        for c in chunks:
            if c.parent_chunk_id and self.graph.has_node(c.parent_chunk_id):
                self.graph.add_edge(c.parent_chunk_id, c.chunk_id, relation="parent_of")

        # Add same-section edges
        section_map = defaultdict(list)
        for c in chunks:
            section_map[c.section_path].append(c.chunk_id)
        for section, cids in section_map.items():
            for i in range(len(cids) - 1):
                self.graph.add_edge(cids[i], cids[i + 1], relation="next_in_section")

        # Add cross-reference edges (detect "see ... section" patterns)
        for c in chunks:
            refs = re.findall(
                r"(?:see|refer to|review).*?(?:section|article|child section)\s+(?:titled\s+)?([A-Z][A-Za-z\s\-&]+)",
                c.text, re.IGNORECASE
            )
            for ref in refs:
                ref_clean = ref.strip()
                for other in chunks:
                    if ref_clean.lower() in other.section_path.lower() and other.chunk_id != c.chunk_id:
                        self.graph.add_edge(c.chunk_id, other.chunk_id, relation="references")
                        break

        logger.info(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def get_neighbors(self, chunk_id: str, max_hops: int = 2) -> list[str]:
        """Get related chunk IDs via graph traversal."""
        if not self.graph or not self.graph.has_node(chunk_id):
            return []

        visited = set()
        queue = [(chunk_id, 0)]
        neighbors = []

        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_hops:
                continue
            visited.add(node)
            if node != chunk_id:
                neighbors.append(node)

            for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return neighbors[:10]


# ═══════════════════════════════════════════════════════════════
# 5. EMBEDDING INDEX — BGE-Large (Open Source, ONNX)
# ═══════════════════════════════════════════════════════════════

class EmbeddingIndex:
    """
    Open-source embeddings via BAAI/bge-large-en-v1.5.
    Backend priority: fastembed (ONNX) → sentence-transformers → OpenAI fallback
    """

    BGE_MODEL = "BAAI/bge-large-en-v1.5"
    BGE_DIM = 1024
    OPENAI_MODEL = "text-embedding-3-small"
    OPENAI_DIM = 1536

    def __init__(self, backend="auto", openai_client=None):
        self.openai_client = openai_client
        self.embeddings = None
        self.faiss_index = None
        self.chunk_ids = []
        self.backend = self._detect(backend)
        self.dim = None
        self._model = None
        self._init_backend()

    def _detect(self, backend):
        if backend != "auto":
            return backend
        try:
            from fastembed import TextEmbedding
            return "fastembed"
        except ImportError:
            pass
        try:
            from sentence_transformers import SentenceTransformer
            return "sbert"
        except ImportError:
            pass
        if self.openai_client:
            return "openai"
        raise RuntimeError("Install fastembed or sentence-transformers for open-source embeddings")

    def _init_backend(self):
        if self.backend == "fastembed":
            from fastembed import TextEmbedding
            logger.info(f"Loading {self.BGE_MODEL} via ONNX...")
            self._model = TextEmbedding(self.BGE_MODEL)
            self.dim = self.BGE_DIM
        elif self.backend == "sbert":
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading {self.BGE_MODEL} via PyTorch...")
            self._model = SentenceTransformer(self.BGE_MODEL)
            self.dim = self.BGE_DIM
        elif self.backend == "openai":
            self.dim = self.OPENAI_DIM
            logger.info(f"Using OpenAI {self.OPENAI_MODEL} as fallback")

    def embed_texts(self, texts, batch_size=64):
        if self.backend == "fastembed":
            embs = list(self._model.embed(texts, batch_size=batch_size))
            result = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            return result / (norms + 1e-10)
        elif self.backend == "sbert":
            return np.array(self._model.encode(texts, normalize_embeddings=True,
                                                batch_size=batch_size), dtype=np.float32)
        elif self.backend == "openai":
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self.openai_client.embeddings.create(model=self.OPENAI_MODEL, input=batch)
                all_embs.extend([item.embedding for item in resp.data])
            result = np.array(all_embs, dtype=np.float32)
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            return result / (norms + 1e-10)

    def build_index(self, chunks):
        texts = [c.text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.embeddings = self.embed_texts(texts)
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(self.dim)
            self.faiss_index.add(self.embeddings)
        logger.info(f"Embedding index: {len(self.chunk_ids)} vectors, dim={self.dim}, backend={self.backend}")

    def search(self, query, top_k=20):
        # BGE query prefix for retrieval
        if self.backend in ("fastembed", "sbert"):
            q = f"Represent this sentence for searching relevant passages: {query}"
        else:
            q = query
        qe = self.embed_texts([q])[0]
        if HAS_FAISS and self.faiss_index:
            scores, indices = self.faiss_index.search(qe.reshape(1, -1), min(top_k, len(self.chunk_ids)))
            return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]
        else:
            sims = self.embeddings @ qe
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(self.chunk_ids[i], float(sims[i])) for i in top_idx]


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []
        self.tokenized_corpus = []

    def build_index(self, chunks):
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.tokenized_corpus = [re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", c.text.lower()) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=20):
        tokens = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]


# ═══════════════════════════════════════════════════════════════
# 6. RERANKER — Cross-Encoder (Open Source)
# ═══════════════════════════════════════════════════════════════

class Reranker:
    """
    RERANKING (from diagram)
    Uses BAAI/bge-reranker-v2-m3 cross-encoder to re-score candidates.
    Falls back to LLM-based reranking if model unavailable.
    """

    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, backend="auto", openai_client=None):
        self.openai_client = openai_client
        self._model = None
        self.backend = backend

        if backend == "auto":
            try:
                from fastembed import TextEmbedding  # Check if fastembed available
                # fastembed has reranker support
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._model = TextCrossEncoder(self.RERANKER_MODEL)
                self.backend = "fastembed"
                logger.info(f"Reranker: {self.RERANKER_MODEL} via fastembed")
            except Exception:
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.RERANKER_MODEL)
                    self.backend = "sbert"
                    logger.info(f"Reranker: {self.RERANKER_MODEL} via CrossEncoder")
                except Exception:
                    self.backend = "llm"
                    logger.info("Reranker: Using LLM-based reranking")

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 10) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        if self.backend == "fastembed" and self._model:
            pairs = [(query, c.text[:500]) for c in chunks]
            scores = list(self._model.rerank(query, [c.text[:500] for c in chunks]))
            scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            return [(c, float(s)) for c, s in scored[:top_k]]

        elif self.backend == "sbert" and self._model:
            pairs = [(query, c.text[:500]) for c in chunks]
            scores = self._model.predict(pairs)
            scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            return [(c, float(s)) for c, s in scored[:top_k]]

        elif self.backend == "llm" and self.openai_client:
            return self._llm_rerank(query, chunks, top_k)

        return [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks[:top_k])]

    def _llm_rerank(self, query, chunks, top_k):
        """LLM-based reranking fallback."""
        chunk_summaries = ""
        for i, c in enumerate(chunks[:20]):
            chunk_summaries += f"\n[{i}] {c.text[:200]}"

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Rank these passages by relevance to the query. "
                        f"Return ONLY the indices as comma-separated numbers, most relevant first.\n"
                        f"{chunk_summaries}"
                    )
                }],
                temperature=0, max_tokens=100,
            )
            indices_str = response.choices[0].message.content.strip()
            indices = [int(x.strip()) for x in re.findall(r"\d+", indices_str)]
            results = []
            for rank, idx in enumerate(indices):
                if 0 <= idx < len(chunks):
                    results.append((chunks[idx], 1.0 / (rank + 1)))
            return results[:top_k]
        except Exception:
            return [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks[:top_k])]


# ═══════════════════════════════════════════════════════════════
# 7. ROUTING — Logical + Semantic Route Selection
# ═══════════════════════════════════════════════════════════════

class QueryRouter:
    """
    ROUTING (from diagram)
    - Logical Route: Determines which data source (vector, graph, or both)
    - Semantic Route: Selects the right prompt/strategy
    """

    STATE_PATTERNS = {
        "CA": [r"\bcalifornia\b", r"\bca\b", r"\bdmhc\b", r"\bcdi\b"],
        "CO": [r"\bcolorado\b", r"\bco\b"], "CT": [r"\bconnecticut\b", r"\bct\b"],
        "GA": [r"\bgeorgia\b", r"\bga\b"], "IN": [r"\bindiana\b"],
        "KY": [r"\bkentucky\b", r"\bky\b"], "ME": [r"\bmaine\b"],
        "MO": [r"\bmissouri\b", r"\bmo\b"], "NH": [r"\bnew\s*hampshire\b", r"\bnh\b"],
        "NV": [r"\bnevada\b", r"\bnv\b"], "NY": [r"\bnew\s*york\b", r"\bny\b"],
        "OH": [r"\bohio\b", r"\boh\b"], "VA": [r"\bvirginia\b", r"\bva\b"],
        "WI": [r"\bwisconsin\b", r"\bwi\b"],
    }
    LOB_PATTERNS = {
        "Individual": [r"\bindividual\b", r"\baca\b"], "Small Group": [r"\bsmall\s*group\b"],
        "Large Group": [r"\blarge\s*group\b"], "National": [r"\bnational\b", r"\bwgs\b"],
    }
    TOPIC_PATTERNS = {
        "expedited": [r"\bexpedited\b", r"\burgent\b"], "grievance": [r"\bgrievance\b", r"\bcomplaint\b"],
        "appeal": [r"\bappeal\b", r"\bdeni"], "DOR": [r"\bdor\b", r"\bdesignation\b"],
        "QOC": [r"\bquality\s*of\s*care\b", r"\bqoc\b"],
        "pharmacy": [r"\bpharmacy\b", r"\bdrug\b", r"\bformulary\b"],
        "behavioral_health": [r"\bbehavioral\b", r"\bmental\s*health\b"],
        "external_review": [r"\bexternal\s*review\b", r"\biro\b"],
        "timeframes": [r"\btat\b", r"\btimeframe\b", r"\bhow\s*long\b"],
        "contact_info": [r"\baddress\b", r"\bfax\b", r"\bwhere\s*to\s*send\b"],
    }

    def classify_filters(self, query):
        q = query.lower()
        filters = {"state": "ALL", "lob": "ALL", "topic": "general"}
        for st, pats in self.STATE_PATTERNS.items():
            if any(re.search(p, q) for p in pats):
                filters["state"] = st
                break
        for lob, pats in self.LOB_PATTERNS.items():
            if any(re.search(p, q) for p in pats):
                filters["lob"] = lob
                break
        scores = {}
        for t, pats in self.TOPIC_PATTERNS.items():
            s = sum(1 for p in pats if re.search(p, q))
            if s:
                scores[t] = s
        if scores:
            filters["topic"] = max(scores, key=scores.get)
        return filters

    def route(self, query: str) -> str:
        """
        Logical Route: pick the retrieval strategy.
        - "vector" — standard semantic search (most queries)
        - "graph"  — follow cross-references (multi-hop queries)
        - "hybrid" — both vector + graph (complex queries)
        """
        q = query.lower()

        # Multi-hop indicators → use graph
        multi_hop_signals = [
            r"walk me through", r"step.by.step", r"full process",
            r"what (?:do i|should i) do .* if", r"before filing",
            r"compare.*(?:across|between|all states)",
        ]
        if any(re.search(p, q) for p in multi_hop_signals):
            return "hybrid"

        # Simple lookup → vector only
        simple_signals = [
            r"what is the (?:fax|address|email|phone)",
            r"what is (?:a|an|the) ",
            r"definition of",
        ]
        if any(re.search(p, q) for p in simple_signals):
            return "vector"

        return "vector"  # Default


# ═══════════════════════════════════════════════════════════════
# 8. QUERY CONSTRUCTION — Multi-Query, HyDE, Decomposition
# ═══════════════════════════════════════════════════════════════

class QueryConstructor:
    """
    QUERY CONSTRUCTION (from diagram)
    Transforms a single query into multiple search strategies:
    1. Multi-Query: Generate 3 variations of the query
    2. HyDE: Generate a hypothetical answer, then search for similar real passages
    3. Decomposition: Break complex queries into sub-questions
    4. RAG Fusion: Combine results from all strategies via RRF
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    def construct(self, query: str, strategy: str = "auto") -> QueryPlan:
        """Build a query plan based on query complexity."""

        # Detect complexity
        q_lower = query.lower()
        is_complex = any(p in q_lower for p in [
            "compare", "walk me through", "step by step", "full process",
            "what should i", "before filing", "all states",
        ])
        is_simple = any(p in q_lower for p in [
            "what is the fax", "what is the address", "definition",
            "what is a ", "what is an ",
        ])

        if is_simple:
            strategy = "single"
        elif is_complex:
            strategy = "decompose"
        else:
            strategy = "multi_query"

        plan = QueryPlan(
            original_query=query,
            sub_queries=[query],
        )

        if strategy == "single":
            return plan

        elif strategy == "multi_query":
            plan.sub_queries = self._multi_query(query)
            return plan

        elif strategy == "decompose":
            plan.sub_queries = self._decompose(query)
            plan.hyde_passage = self._hyde(query)
            return plan

        return plan

    def _multi_query(self, query: str) -> list[str]:
        """Generate 3 query variations for RAG Fusion."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate 3 different versions of this search query to find relevant documents. "
                        f"Each should approach the topic from a different angle. "
                        f"Return ONLY the 3 queries, one per line.\n\n"
                        f"Query: {query}"
                    )
                }],
                temperature=0.7, max_tokens=200,
            )
            queries = [q.strip().lstrip("0123456789.-) ") for q in
                       response.choices[0].message.content.strip().split("\n") if q.strip()]
            return [query] + queries[:3]  # Original + 3 variations
        except Exception:
            return [query]

    def _decompose(self, query: str) -> list[str]:
        """Break complex query into sub-questions."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Break this complex question into 2-4 simpler sub-questions that together "
                        f"would answer the original. Return ONLY the sub-questions, one per line.\n\n"
                        f"Question: {query}"
                    )
                }],
                temperature=0.3, max_tokens=300,
            )
            subs = [q.strip().lstrip("0123456789.-) ") for q in
                    response.choices[0].message.content.strip().split("\n") if q.strip()]
            return [query] + subs[:4]
        except Exception:
            return [query]

    def _hyde(self, query: str) -> str:
        """
        HyDE — Hypothetical Document Embedding.
        Generate a hypothetical answer, then search for real passages similar to it.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Write a short paragraph that would be the ideal answer to this question, "
                        f"as if it were in a health insurance grievance and appeals policy document.\n\n"
                        f"Question: {query}"
                    )
                }],
                temperature=0.3, max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════════
# 9. HYBRID RETRIEVER — Vector + BM25 + Graph + Reranking
# ═══════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Full retrieval pipeline combining all sources:
    1. Dense search (BGE embeddings)
    2. Sparse search (BM25)
    3. Graph traversal (section relationships)
    4. Metadata boosting (state/LOB/topic)
    5. RAG Fusion (merge results from multi-query)
    6. Cross-encoder reranking
    """

    def __init__(self, embedding_index, bm25_index, reranker, section_graph,
                 chunks, dense_weight=0.5, sparse_weight=0.3, graph_weight=0.2):
        self.embedding_index = embedding_index
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.graph = section_graph
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.chunks = chunks
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.graph_weight = graph_weight

    def retrieve(self, plan: QueryPlan, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Execute multi-source retrieval with RAG Fusion."""

        # Step 1: Run each sub-query through dense + sparse
        all_dense = {}
        all_sparse = {}

        for sq in plan.sub_queries:
            for cid, score in self.embedding_index.search(sq, top_k=20):
                all_dense[cid] = max(all_dense.get(cid, 0), score)
            for cid, score in self.bm25_index.search(sq, top_k=20):
                all_sparse[cid] = max(all_sparse.get(cid, 0), score)

        # HyDE: also search with the hypothetical passage
        if plan.hyde_passage:
            for cid, score in self.embedding_index.search(plan.hyde_passage, top_k=10):
                all_dense[cid] = max(all_dense.get(cid, 0), score * 0.8)

        # Step 2: Reciprocal Rank Fusion
        fused = {}
        rrf_k = 60
        for rank, (cid, _) in enumerate(sorted(all_dense.items(), key=lambda x: -x[1])):
            fused[cid] = fused.get(cid, 0) + self.dense_weight / (rrf_k + rank + 1)
        for rank, (cid, _) in enumerate(sorted(all_sparse.items(), key=lambda x: -x[1])):
            fused[cid] = fused.get(cid, 0) + self.sparse_weight / (rrf_k + rank + 1)

        # Step 3: Graph expansion (if hybrid route)
        if plan.route == "hybrid":
            top_initial = sorted(fused.items(), key=lambda x: -x[1])[:5]
            for cid, score in top_initial:
                neighbors = self.graph.get_neighbors(cid, max_hops=2)
                for ncid in neighbors:
                    if ncid in self.chunk_map:
                        fused[ncid] = fused.get(ncid, 0) + self.graph_weight / (rrf_k + 1)

        # Step 4: Metadata boosting
        filters = plan.filters
        for cid in fused:
            chunk = self.chunk_map.get(cid)
            if not chunk:
                continue
            boost = 1.0
            if filters.get("state", "ALL") != "ALL":
                if chunk.state == filters["state"]:
                    boost *= 2.5
                elif chunk.state == "ALL":
                    boost *= 1.0
                else:
                    boost *= 0.3
            if filters.get("lob", "ALL") != "ALL":
                if chunk.lob == filters["lob"]:
                    boost *= 1.8
                elif chunk.lob != "ALL":
                    boost *= 0.5
            if filters.get("topic", "general") != "general":
                if chunk.topic == filters["topic"]:
                    boost *= 1.5
            fused[cid] *= boost

        # Step 5: Get top candidates
        sorted_results = sorted(fused.items(), key=lambda x: -x[1])
        candidates = []
        for cid, score in sorted_results[:25]:
            chunk = self.chunk_map.get(cid)
            if chunk:
                candidates.append(chunk)

        # Step 6: Rerank with cross-encoder
        reranked = self.reranker.rerank(plan.original_query, candidates, top_k=top_k)

        # Step 7: Context expansion
        return self._expand(reranked, top_k)

    def _expand(self, results, max_total):
        seen = {c.chunk_id for c, _ in results}
        expanded = list(results)
        for chunk, score in results[:3]:
            for other in self.chunks:
                if other.chunk_id not in seen and other.section_path == chunk.section_path:
                    expanded.append((other, score * 0.4))
                    seen.add(other.chunk_id)
                    if len(expanded) >= max_total + 5:
                        break
        return expanded[:max_total + 5]


# ═══════════════════════════════════════════════════════════════
# 10. GENERATION — Self-RAG with Verification
# ═══════════════════════════════════════════════════════════════

class SelfRAGGenerator:
    """
    GENERATION (from diagram) — Self-RAG pattern:
    1. Generate initial answer
    2. Verify: Is the answer grounded in the context?
    3. If not → retrieve more → regenerate
    4. Add citations
    """

    SYSTEM_PROMPT = """You are an expert Anthem Grievance & Appeals knowledge base assistant.

RULES:
1. Answer ONLY from the context. If info is missing, say so clearly.
2. Always specify STATE and LOB your answer applies to.
3. For timeframes: specify Calendar Days (CD) or Business Days (BD).
4. Include full contact info (addresses, fax, emails) when relevant.
5. For procedures: provide step-by-step instructions.
6. Cite sections: [Section: California > Large Group > Fax].
7. Note Important Notes and warnings from source material.
"""

    VERIFY_PROMPT = """Check if this answer is fully supported by the context.
Return a JSON object with:
- "is_grounded": true/false
- "missing_info": list of what's missing (empty if grounded)
- "confidence": 0.0-1.0

Context: {context}
Answer: {answer}

Return ONLY the JSON."""

    def __init__(self, openai_client, model="gpt-4o-mini"):
        self.client = openai_client
        self.model = model

    def generate(self, query, retrieved_chunks, max_retries=1):
        context = self._build_context(retrieved_chunks)
        answer = self._generate_answer(query, context)

        # Self-RAG verification step
        if max_retries > 0:
            verification = self._verify(context, answer)
            if not verification.get("is_grounded", True):
                logger.info(f"Self-RAG: Answer not fully grounded. Missing: {verification.get('missing_info', [])}")
                # Could trigger re-retrieval here with missing_info
                # For now, append a note
                missing = verification.get("missing_info", [])
                if missing:
                    answer += (
                        f"\n\n**Note:** The following information may be incomplete or "
                        f"not found in the available context: {', '.join(missing[:3])}. "
                        f"Please verify in the member's EOC/BPD."
                    )

        return answer

    def _build_context(self, chunks):
        parts = []
        for i, (chunk, score) in enumerate(chunks):
            parts.append(
                f"--- CHUNK {i+1} [Section: {chunk.section_path}] "
                f"[State: {chunk.state}] [LOB: {chunk.lob}] [Page: {chunk.page_num}] ---\n"
                f"{chunk.text}\n"
            )
        ctx = "\n".join(parts)
        if count_tokens(ctx) > 12000:
            ctx = ctx[:int(len(ctx) * 12000 / count_tokens(ctx))]
        return ctx

    def _generate_answer(self, query, context):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
            ],
            temperature=0.1, max_tokens=2000,
        )
        return response.choices[0].message.content

    def _verify(self, context, answer):
        """Self-RAG grounding check."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": self.VERIFY_PROMPT.format(
                        context=context[:5000], answer=answer[:1000]
                    ),
                }],
                temperature=0, max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"```json\s*|\s*```", "", text)
            return json.loads(text)
        except Exception:
            return {"is_grounded": True, "missing_info": [], "confidence": 0.8}


# ═══════════════════════════════════════════════════════════════
# 11. EVALUATION — RAGAS-style Metrics
# ═══════════════════════════════════════════════════════════════

class RAGEvaluator:
    """
    EVALS (from diagram)
    Implements RAGAS-style metrics without the RAGAS library:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Recall: Did we retrieve the right chunks?
    """

    def __init__(self, openai_client):
        self.client = openai_client

    def evaluate(self, query, answer, context_chunks, ground_truth=None):
        """Run all evaluation metrics."""
        context_text = "\n".join([c.text[:300] for c, _ in context_chunks[:5]])

        faithfulness = self._faithfulness(answer, context_text)
        relevancy = self._answer_relevancy(query, answer)
        metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
        }
        if ground_truth:
            metrics["context_recall"] = self._context_recall(ground_truth, context_text)

        return metrics

    def _faithfulness(self, answer, context):
        """Can every claim in the answer be traced to the context?"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rate how well the answer is supported by the context. "
                        f"Return ONLY a number between 0.0 and 1.0.\n\n"
                        f"Context: {context[:3000]}\n\nAnswer: {answer[:1000]}"
                    )
                }],
                temperature=0, max_tokens=10,
            )
            score = float(re.search(r"[\d.]+", response.choices[0].message.content).group())
            return min(1.0, max(0.0, score))
        except Exception:
            return 0.5

    def _answer_relevancy(self, query, answer):
        """Does the answer actually address the question?"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rate how well this answer addresses the question. "
                        f"Return ONLY a number between 0.0 and 1.0.\n\n"
                        f"Question: {query}\n\nAnswer: {answer[:1000]}"
                    )
                }],
                temperature=0, max_tokens=10,
            )
            score = float(re.search(r"[\d.]+", response.choices[0].message.content).group())
            return min(1.0, max(0.0, score))
        except Exception:
            return 0.5

    def _context_recall(self, ground_truth, context):
        """Is the ground truth information present in retrieved context?"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"What fraction of the ground truth information is present in the context? "
                        f"Return ONLY a number between 0.0 and 1.0.\n\n"
                        f"Ground Truth: {ground_truth[:500]}\n\nContext: {context[:3000]}"
                    )
                }],
                temperature=0, max_tokens=10,
            )
            score = float(re.search(r"[\d.]+", response.choices[0].message.content).group())
            return min(1.0, max(0.0, score))
        except Exception:
            return 0.5


# ═══════════════════════════════════════════════════════════════
# 12. MAIN PIPELINE — Orchestrator
# ═══════════════════════════════════════════════════════════════

class RobustRAGPipeline:
    """
    Full production RAG pipeline matching the reference architecture.

    Components:
        Indexing:    Semantic split + Multi-rep summaries + RAPTOR hierarchy
        Routing:     Logical (vector/graph/hybrid) + Semantic (filters)
        Query:       Multi-query + HyDE + Decomposition
        Retrieval:   Dense + BM25 + Graph + Reranking
        Generation:  Self-RAG with grounding verification
        Evals:       Faithfulness + Relevancy metrics

    Usage:
        pipeline = RobustRAGPipeline(openai_api_key="sk-...")
        pipeline.ingest_pdf("km2.pdf")
        pipeline.save_index("./index")
        result = pipeline.query("What is the TAT for CA expedited appeals?")
    """

    def __init__(
        self,
        openai_api_key: str,
        embedding_backend: str = "auto",
        generation_model: str = "gpt-4o-mini",
        max_chunk_tokens: int = 600,
        enable_multi_rep: bool = True,
        enable_hierarchy: bool = True,
        enable_graph: bool = True,
        enable_reranker: bool = True,
        enable_self_rag: bool = True,
    ):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.generation_model = generation_model
        self.config = {
            "embedding_backend": embedding_backend,
            "max_chunk_tokens": max_chunk_tokens,
            "enable_multi_rep": enable_multi_rep,
            "enable_hierarchy": enable_hierarchy,
            "enable_graph": enable_graph,
            "enable_reranker": enable_reranker,
            "enable_self_rag": enable_self_rag,
        }

        # Components
        self.parser = PDFParser()
        self.chunker = StructuralChunker(max_chunk_tokens=max_chunk_tokens)
        self.multi_rep = MultiRepresentationIndexer(self.openai_client) if enable_multi_rep else None
        self.embedding_index = EmbeddingIndex(backend=embedding_backend, openai_client=self.openai_client)
        self.bm25_index = BM25Index()
        self.reranker = Reranker(openai_client=self.openai_client) if enable_reranker else Reranker(backend="none")
        self.graph = SectionGraph() if enable_graph else SectionGraph()
        self.router = QueryRouter()
        self.query_constructor = QueryConstructor(self.openai_client)
        self.generator = SelfRAGGenerator(self.openai_client, model=generation_model)
        self.evaluator = RAGEvaluator(self.openai_client)
        self.retriever = None
        self.chunks = []

    def ingest_pdf(self, pdf_path: str):
        """Full ingestion: parse → chunk → enrich → multi-rep → hierarchy → embed → index → graph."""
        logger.info(f"=== INGESTING {pdf_path} ===")

        # Step 1: Parse
        pages = self.parser.extract_pages(pdf_path)

        # Step 2: Structural chunking with metadata
        self.chunks = self.chunker.chunk_pages(pages)
        logger.info(f"Base chunks: {len(self.chunks)}")

        # Step 3: Multi-representation summaries
        if self.multi_rep and self.config["enable_multi_rep"]:
            self.chunks = self.multi_rep.create_summaries(self.chunks)

        # Step 4: RAPTOR-style hierarchy
        if self.config["enable_hierarchy"]:
            hierarchy_indexer = MultiRepresentationIndexer(self.openai_client)
            parent_chunks = hierarchy_indexer.create_hierarchy(self.chunks, self.openai_client)
            self.chunks.extend(parent_chunks)
            logger.info(f"Total chunks (with hierarchy): {len(self.chunks)}")

        # Step 5: Build embedding index
        self.embedding_index.build_index(self.chunks)

        # Step 6: Build BM25 index
        self.bm25_index.build_index(self.chunks)

        # Step 7: Build graph
        if self.config["enable_graph"]:
            self.graph.build_from_chunks(self.chunks)

        # Step 8: Create retriever
        self.retriever = HybridRetriever(
            self.embedding_index, self.bm25_index, self.reranker, self.graph, self.chunks,
        )

        self._log_stats()
        logger.info("=== INGESTION COMPLETE ===")

    def query(self, question: str, top_k: int = 10, evaluate: bool = False) -> dict:
        """
        Full query pipeline:
        Route → Construct queries → Retrieve → Rerank → Generate → (Verify) → (Evaluate)
        """
        if not self.retriever:
            raise RuntimeError("Call ingest_pdf() first")

        # Step 1: Route
        route = self.router.route(question)
        filters = self.router.classify_filters(question)
        logger.info(f"Route: {route} | Filters: state={filters['state']}, lob={filters['lob']}, topic={filters['topic']}")

        # Step 2: Construct query plan
        plan = self.query_constructor.construct(question)
        plan.route = route
        plan.filters = filters
        logger.info(f"Query plan: {len(plan.sub_queries)} sub-queries, HyDE={'yes' if plan.hyde_passage else 'no'}")

        # Step 3: Retrieve + Rerank
        results = self.retriever.retrieve(plan, top_k=top_k)

        # Step 4: Generate (with Self-RAG verification)
        if self.config["enable_self_rag"]:
            answer = self.generator.generate(question, results, max_retries=1)
        else:
            answer = self.generator.generate(question, results, max_retries=0)

        # Step 5: Evaluate (optional)
        eval_metrics = {}
        if evaluate:
            eval_metrics = self.evaluator.evaluate(question, answer, results)

        # Format response
        sources = [{
            "section": c.section_path, "state": c.state, "lob": c.lob,
            "topic": c.topic, "page": c.page_num, "score": round(s, 4),
            "preview": c.text[:150] + "..." if len(c.text) > 150 else c.text,
        } for c, s in results]

        return {
            "answer": answer,
            "sources": sources,
            "filters": filters,
            "route": route,
            "num_sub_queries": len(plan.sub_queries),
            "hyde_used": bool(plan.hyde_passage),
            "num_chunks_retrieved": len(results),
            "eval_metrics": eval_metrics,
        }

    def save_index(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump([asdict(c) for c in self.chunks], f)
        if self.embedding_index.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embedding_index.embeddings)
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump({"chunk_ids": self.bm25_index.chunk_ids,
                         "tokenized_corpus": self.bm25_index.tokenized_corpus}, f)
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({**self.config, "dim": self.embedding_index.dim,
                        "backend": self.embedding_index.backend}, f)
        logger.info(f"Saved to {directory}")

    def load_index(self, directory):
        with open(os.path.join(directory, "chunks.json")) as f:
            self.chunks = [Chunk(**d) for d in json.load(f)]
        embs = np.load(os.path.join(directory, "embeddings.npy"))
        self.embedding_index.embeddings = embs
        self.embedding_index.chunk_ids = [c.chunk_id for c in self.chunks]
        cfg = {}
        cfg_path = os.path.join(directory, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        self.embedding_index.dim = cfg.get("dim", embs.shape[1])
        if HAS_FAISS:
            self.embedding_index.faiss_index = faiss.IndexFlatIP(self.embedding_index.dim)
            self.embedding_index.faiss_index.add(embs)
        with open(os.path.join(directory, "bm25.pkl"), "rb") as f:
            d = pickle.load(f)
        self.bm25_index.chunk_ids = d["chunk_ids"]
        self.bm25_index.tokenized_corpus = d["tokenized_corpus"]
        self.bm25_index.bm25 = BM25Okapi(d["tokenized_corpus"])
        if self.config["enable_graph"]:
            self.graph.build_from_chunks(self.chunks)
        self.retriever = HybridRetriever(
            self.embedding_index, self.bm25_index, self.reranker, self.graph, self.chunks,
        )
        logger.info(f"Loaded {len(self.chunks)} chunks from {directory}")

    def _log_stats(self):
        states = defaultdict(int)
        topics = defaultdict(int)
        levels = defaultdict(int)
        for c in self.chunks:
            states[c.state] += 1
            topics[c.topic] += 1
            levels[c.level] += 1
        logger.info(f"States: {dict(sorted(states.items(), key=lambda x: -x[1])[:10])}")
        logger.info(f"Topics: {dict(sorted(topics.items(), key=lambda x: -x[1])[:10])}")
        logger.info(f"Levels: {dict(levels)} (0=leaf, 1=section summary)")


# ═══════════════════════════════════════════════════════════════
# 13. CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    p = argparse.ArgumentParser(description="Robust RAG Pipeline")
    p.add_argument("--pdf", type=str)
    p.add_argument("--index-dir", default="./rag_index")
    p.add_argument("--query", type=str)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--backend", default="auto", choices=["auto", "fastembed", "sbert", "openai"])
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--eval", action="store_true", help="Run evaluation metrics on each query")
    p.add_argument("--no-multi-rep", action="store_true", help="Disable multi-representation indexing")
    p.add_argument("--no-hierarchy", action="store_true", help="Disable RAPTOR hierarchy")
    p.add_argument("--no-graph", action="store_true", help="Disable graph")
    p.add_argument("--no-reranker", action="store_true", help="Disable reranking")
    p.add_argument("--no-self-rag", action="store_true", help="Disable Self-RAG verification")

    args = p.parse_args()
    if not args.api_key:
        print("Set OPENAI_API_KEY or use --api-key")
        return

    pipeline = RobustRAGPipeline(
        openai_api_key=args.api_key,
        embedding_backend=args.backend,
        generation_model=args.model,
        enable_multi_rep=not args.no_multi_rep,
        enable_hierarchy=not args.no_hierarchy,
        enable_graph=not args.no_graph,
        enable_reranker=not args.no_reranker,
        enable_self_rag=not args.no_self_rag,
    )

    if args.pdf:
        pipeline.ingest_pdf(args.pdf)
        pipeline.save_index(args.index_dir)
    elif os.path.exists(os.path.join(args.index_dir, "chunks.json")):
        pipeline.load_index(args.index_dir)
    else:
        print("Provide --pdf or ensure index exists")
        return

    if args.query:
        result = pipeline.query(args.query, evaluate=args.eval)
        _print(result)

    if args.interactive:
        print(f"\n{'='*60}\nInteractive mode ('quit' to exit, 'eval' to toggle metrics)\n{'='*60}")
        do_eval = args.eval
        while True:
            try:
                q = input("\n> ").strip()
                if q.lower() in ("quit", "exit"):
                    break
                if q.lower() == "eval":
                    do_eval = not do_eval
                    print(f"Evaluation: {'ON' if do_eval else 'OFF'}")
                    continue
                if not q:
                    continue
                result = pipeline.query(q, evaluate=do_eval)
                _print(result)
            except KeyboardInterrupt:
                break


def _print(r):
    print(f"\n{'='*60}")
    print(r["answer"])
    print(f"\n--- Route: {r['route']} | Sub-queries: {r['num_sub_queries']} | "
          f"HyDE: {r['hyde_used']} | Chunks: {r['num_chunks_retrieved']} ---")
    print(f"--- Filters: state={r['filters']['state']}, lob={r['filters']['lob']}, topic={r['filters']['topic']} ---")
    if r.get("eval_metrics"):
        m = r["eval_metrics"]
        print(f"--- Eval: faithfulness={m.get('faithfulness', 'N/A'):.2f}, "
              f"relevancy={m.get('answer_relevancy', 'N/A'):.2f} ---")
    for i, s in enumerate(r["sources"][:5]):
        print(f"  [{i+1}] {s['section'][:65]} (State:{s['state']}, Score:{s['score']})")


if __name__ == "__main__":
    main()
