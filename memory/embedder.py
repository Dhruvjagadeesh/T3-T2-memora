import numpy as np
import torch
from dataclasses import dataclass
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import pipeline


EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"

EMOTION_VALENCE = {
    "joy":      0.85,
    "surprise": 0.10,
    "neutral":  0.00,
    "fear":    -0.65,
    "sadness": -0.70,
    "disgust": -0.75,
    "anger":   -0.80,
}

EMOTION_AROUSAL = {
    "joy":      0.65,
    "surprise": 0.70,
    "neutral":  0.10,
    "fear":     0.75,
    "sadness":  0.30,
    "disgust":  0.55,
    "anger":    0.85,
}


@dataclass
class EmotionResult:
    label:    str
    score:    float
    valence:  float
    arousal:  float
    saliency: float


@dataclass
class EmbeddingResult:
    vector:    np.ndarray
    dimension: int


class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        self._loaded = True

    def encode(self, text: str) -> EmbeddingResult:
        self._load()
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return EmbeddingResult(vector=vec.astype(np.float32), dimension=len(vec))

    def encode_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        self._load()
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
        )
        return [EmbeddingResult(vector=v.astype(np.float32), dimension=len(v)) for v in vecs]


class EmotionTagger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        self._pipe = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
        )
        self._loaded = True

    def tag(self, text: str) -> EmotionResult:
        self._load()
        results = self._pipe(text[:512])[0]
        top = max(results, key=lambda x: x["score"])
        label   = top["label"].lower()
        score   = top["score"]
        valence = EMOTION_VALENCE.get(label, 0.0)
        arousal = EMOTION_AROUSAL.get(label, 0.0)
        saliency = self._compute_saliency(valence, arousal)
        return EmotionResult(
            label=label,
            score=score,
            valence=valence,
            arousal=arousal,
            saliency=saliency,
        )

    def _compute_saliency(self, valence: float, arousal: float) -> float:
        if abs(valence) > 0.6 and arousal > 0.5:
            return (abs(valence) + arousal) / 2.0
        return abs(valence) * 0.3 + arousal * 0.2


_embedder     = None
_emotion_tagger = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_emotion_tagger() -> EmotionTagger:
    global _emotion_tagger
    if _emotion_tagger is None:
        _emotion_tagger = EmotionTagger()
    return _emotion_tagger
