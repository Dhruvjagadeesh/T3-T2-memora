"""
data/empathetic_loader.py
=========================
Loads and preprocesses the EmpatheticDialogues dataset (Rashkin et al., 2019)
for Trust Engine experiments.

Dataset structure
-----------------
EmpatheticDialogues has 25k conversations, each anchored to one of 32
emotion labels.  Each conversation: conv_id, emotion_label, prompt,
utterances (speaker A / B alternating).

Output per conversation: a list of TurnSample dataclasses ready to feed
into TrustEngine.update().

Usage
-----
    from data.empathetic_loader import EmpatheticLoader

    loader = EmpatheticLoader(split="train", max_convs=500)
    convs   = loader.load()          # list[Conversation]
    samples = loader.flatten(convs)  # list[TurnSample]
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from features.emotion_map import get_valence_arousal
from features.behavior_features import BehaviorFeatureExtractor
from features.ic_simulator import ICSimulator


@dataclass
class TurnSample:
    """One processed turn, ready for TrustEngine.update()."""
    conv_id:       str
    turn_index:    int
    speaker_id:    str          # 'A' or 'B' within conv; prefixed with conv_id
    text:          str
    emotion_label: str          # ground-truth label from dataset
    valence:       float        # mapped from emotion_label
    arousal:       float
    behavior_vec:  np.ndarray   # shape (16,)
    ic_score:      float
    session_boundary: bool      # True on first turn of a new conversation
    is_last_turn:  bool

@dataclass
class Conversation:
    conv_id:       str
    emotion_label: str
    turns:         list[TurnSample] = field(default_factory=list)


class EmpatheticLoader:
    """
    Loads EmpatheticDialogues via HuggingFace datasets.
    Falls back to a small synthetic stub if the dataset cannot be
    downloaded (offline environments).
    """

    DATASET_NAME = "empathetic_dialogues"

    def __init__(self,
                 split: str = "train",
                 max_convs: int = 500,
                 ic_mode: str = "stochastic",
                 seed: int = 42):
        self.split     = split
        self.max_convs = max_convs
        self.ic_mode   = ic_mode
        self._feat_ext  = BehaviorFeatureExtractor()
        self._ic_sim    = ICSimulator(seed=seed)

    # -----------------------------------------------------------------------
    def load(self) -> list[Conversation]:
        try:
            return self._load_hf()
        except Exception as e:
            print(f"[EmpatheticLoader] HF download failed ({e}). Using stub.")
            return self._load_stub()

    def flatten(self, convs: list[Conversation]) -> list[TurnSample]:
        out = []
        for c in convs:
            out.extend(c.turns)
        return out

    # -----------------------------------------------------------------------
    def _load_hf(self) -> list[Conversation]:
        from datasets import load_dataset
        ds = load_dataset(self.DATASET_NAME, split=self.split,
                          trust_remote_code=True)

        # Group rows by conv_id
        conv_dict: dict[str, list] = {}
        for row in ds:
            cid = row["conv_id"]
            conv_dict.setdefault(cid, []).append(row)

        convs = []
        for cid, rows in list(conv_dict.items())[:self.max_convs]:
            conv = self._build_conversation(cid, rows)
            convs.append(conv)
        return convs

    def _build_conversation(self, conv_id: str, rows: list) -> Conversation:
        emotion_label = rows[0].get("context", "neutral")
        conv = Conversation(conv_id=conv_id, emotion_label=emotion_label)

        utterances = [r["utterance"] for r in rows]
        n = len(utterances)

        for i, (row, utt) in enumerate(zip(rows, utterances)):
            # Each row alternates speaker A / B
            raw_speaker = row.get("selfeval", str(i % 2))
            speaker_id  = f"{conv_id}_{i % 2}"          # stable per-conv speaker

            valence, arousal = get_valence_arousal(emotion_label)
            # Add slight per-turn noise to simulate dynamic emotion
            rng = np.random.default_rng(hash(f"{conv_id}_{i}") % (2**31))
            valence = float(np.clip(valence + rng.normal(0, 0.05), -1, 1))
            arousal = float(np.clip(arousal + rng.normal(0, 0.03),  0, 1))

            bvec = self._feat_ext.extract_turn(
                text=utt,
                session_turn_index=i,
                total_session_turns=n
            )

            ic = self._ic_sim.get_ic(
                speaker_id=speaker_id,
                turn_index=i,
                mode=self.ic_mode
            )

            turn = TurnSample(
                conv_id=conv_id,
                turn_index=i,
                speaker_id=speaker_id,
                text=utt,
                emotion_label=emotion_label,
                valence=valence,
                arousal=arousal,
                behavior_vec=bvec,
                ic_score=ic,
                session_boundary=(i == 0),
                is_last_turn=(i == n - 1),
            )
            conv.turns.append(turn)
        return conv

    # -----------------------------------------------------------------------
    # Offline fallback stub (32 emotions × 3 turns)
    # -----------------------------------------------------------------------
    def _load_stub(self) -> list[Conversation]:
        STUB_EMOTIONS = [
            ("joyful",    "I'm so happy today, everything is going well!",
                          "That's wonderful to hear, tell me more.",
                          "Life feels really good right now."),
            ("anxious",   "I keep worrying about my health exam results.",
                          "I understand, uncertainty is really hard.",
                          "I just can't stop thinking about it."),
            ("angry",     "My colleague never respects my boundaries!",
                          "That sounds really frustrating.",
                          "I don't know how much longer I can take it."),
            ("sad",       "My dog passed away yesterday.",
                          "I'm so sorry for your loss.",
                          "I miss him so much already."),
            ("excited",   "I just got accepted into my dream university!",
                          "Congratulations, that is amazing news!",
                          "I can't wait for this new chapter."),
            ("lonely",    "I feel like nobody really understands me.",
                          "I'm here to listen, please share more.",
                          "It helps a little to talk about it."),
            ("guilty",    "I forgot my friend's birthday and feel terrible.",
                          "It happens, but it's good that you care.",
                          "I need to make it up to them."),
            ("grateful",  "My team supported me through a really tough time.",
                          "Having that support makes a real difference.",
                          "I feel so fortunate to have them."),
        ]

        convs = []
        for i, (emotion, u0, u1, u2) in enumerate(STUB_EMOTIONS):
            cid = f"stub_{i:03d}"
            rows = [
                {"conv_id": cid, "context": emotion, "utterance": u0},
                {"conv_id": cid, "context": emotion, "utterance": u1},
                {"conv_id": cid, "context": emotion, "utterance": u2},
            ]
            conv = self._build_conversation(cid, rows)
            convs.append(conv)
        return convs


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = EmpatheticLoader(split="train", max_convs=5)
    convs  = loader.load()
    print(f"Loaded {len(convs)} conversations")
    for turn in convs[0].turns:
        print(f"  [{turn.turn_index}] {turn.emotion_label:15s} "
              f"V={turn.valence:+.2f} A={turn.arousal:.2f} "
              f"IC={turn.ic_score:.2f}  '{turn.text[:50]}'")
