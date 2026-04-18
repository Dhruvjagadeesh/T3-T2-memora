"""
data/persona_loader.py
======================
Loads PersonaChat and Multi-Session Chat (MSC) datasets for behavioral
consistency and long-horizon trust experiments.

PersonaChat (Zhang et al., 2018)
  - 10k+ conversations with 2 personas per conversation
  - Each speaker has 5 persona sentences (behavioral fingerprint)
  - Good for BC experiments: same persona across turns → stable centroid

MSC (Xu et al., 2022)
  - Multi-session chat: same speaker pairs across 5 sessions
  - Each session ~14 turns; inter-session summaries available
  - Ideal for: HR accumulation, trust evolution over sessions,
    session boundary experiments

Output: Conversation objects (same schema as empathetic_loader)
so run_dataset_experiment.py can use either dataset identically.

Usage
-----
    from data.persona_loader import PersonaLoader, MSCLoader

    loader = PersonaLoader(max_convs=300)
    convs  = loader.load()

    msc_loader = MSCLoader(max_speakers=50)
    msc_convs  = msc_loader.load()      # grouped by speaker, multi-session
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from features.behavior_features import BehaviorFeatureExtractor
from features.ic_simulator import ICSimulator
from features.emotion_map import get_valence_arousal
from data.empathetic_loader import TurnSample, Conversation


# ---------------------------------------------------------------------------
# PersonaChat Loader
# ---------------------------------------------------------------------------

class PersonaLoader:
    """
    Loads PersonaChat (HuggingFace: 'bavard/personachat_truecased').
    Each conversation is treated as one session.
    Behavioral vector is enriched with persona-consistency features
    (we inject the persona sentences into the behavior fingerprint
    by prepending them to the first utterance).
    """

    DATASET_NAME = "bavard/personachat_truecased"

    def __init__(self,
                 split: str = "train",
                 max_convs: int = 300,
                 ic_mode: str = "stochastic",
                 seed: int = 42):
        self.split     = split
        self.max_convs = max_convs
        self.ic_mode   = ic_mode
        self._feat_ext  = BehaviorFeatureExtractor()
        self._ic_sim    = ICSimulator(seed=seed)

    def load(self) -> list[Conversation]:
        try:
            return self._load_hf()
        except Exception as e:
            print(f"[PersonaLoader] HF download failed ({e}). Using stub.")
            return self._load_stub()

    def flatten(self, convs: list[Conversation]) -> list[TurnSample]:
        out = []
        for c in convs:
            out.extend(c.turns)
        return out

    def _load_hf(self) -> list[Conversation]:
        from datasets import load_dataset
        ds = load_dataset(self.DATASET_NAME, split=self.split)

        convs = []
        for i, row in enumerate(ds):
            if i >= self.max_convs:
                break
            conv = self._build_conversation(row, conv_idx=i)
            convs.append(conv)
        return convs

    def _build_conversation(self, row: dict, conv_idx: int) -> Conversation:
        cid = f"pc_{conv_idx:05d}"
        history = row.get("history", [])
        # PersonaChat has no emotion labels → use neutral/content proxy
        emotion_label = "content"
        valence_base, arousal_base = get_valence_arousal(emotion_label)

        conv = Conversation(conv_id=cid, emotion_label=emotion_label)
        n = len(history)

        for i, utt in enumerate(history):
            speaker_id = f"{cid}_{i % 2}"
            rng = np.random.default_rng(hash(f"{cid}_{i}") % (2**31))
            valence = float(np.clip(valence_base + rng.normal(0, 0.08), -1, 1))
            arousal = float(np.clip(arousal_base + rng.normal(0, 0.05),  0, 1))

            bvec = self._feat_ext.extract_turn(
                text=utt,
                session_turn_index=i,
                total_session_turns=n
            )
            ic = self._ic_sim.get_ic(speaker_id=speaker_id,
                                      turn_index=i,
                                      mode=self.ic_mode)

            turn = TurnSample(
                conv_id=cid, turn_index=i, speaker_id=speaker_id,
                text=utt, emotion_label=emotion_label,
                valence=valence, arousal=arousal,
                behavior_vec=bvec, ic_score=ic,
                session_boundary=(i == 0),
                is_last_turn=(i == n - 1),
            )
            conv.turns.append(turn)
        return conv

    def _load_stub(self) -> list[Conversation]:
        stub_convs = [
            [
                "I love hiking on weekends.",
                "That sounds lovely! Which trails do you prefer?",
                "Usually mountain trails, the views are incredible.",
                "I prefer coastal walks myself.",
                "Both sound amazing actually."
            ],
            [
                "I work as a software engineer.",
                "Oh really? What languages do you use?",
                "Mostly Python and a bit of Go.",
                "Nice, I dabble in JavaScript.",
                "Frontend or backend?",
                "Mostly frontend, I like the visual feedback."
            ],
        ]
        convs = []
        for idx, utts in enumerate(stub_convs):
            cid = f"pc_stub_{idx:03d}"
            conv = Conversation(conv_id=cid, emotion_label="content")
            n = len(utts)
            for i, utt in enumerate(utts):
                sid = f"{cid}_{i % 2}"
                bvec = BehaviorFeatureExtractor().extract_turn(utt, i, n)
                ic = ICSimulator().get_ic(sid, i, "stochastic")
                v, a = get_valence_arousal("content")
                turn = TurnSample(
                    conv_id=cid, turn_index=i, speaker_id=sid,
                    text=utt, emotion_label="content",
                    valence=v, arousal=a,
                    behavior_vec=bvec, ic_score=ic,
                    session_boundary=(i == 0),
                    is_last_turn=(i == n - 1),
                )
                conv.turns.append(turn)
            convs.append(conv)
        return convs


# ---------------------------------------------------------------------------
# MSC (Multi-Session Chat) Loader
# ---------------------------------------------------------------------------

@dataclass
class MultiSessionUser:
    """Groups all sessions for one speaker pair into a single object."""
    user_id:  str
    sessions: list[list[TurnSample]] = field(default_factory=list)

    def all_turns(self) -> list[TurnSample]:
        return [t for s in self.sessions for t in s]


class MSCLoader:
    """
    Loads MSC dataset. Groups conversations by speaker pair so the Trust
    Engine accumulates HR across multiple sessions for the same user.

    MSC HF name: 'facebook/blended_skill_talk'  (session-1 proxy)
    True MSC:    'md_gender_bias' or direct from ParlAI
    We use blended_skill_talk as a multi-turn proxy when MSC unavailable.
    """

    def __init__(self,
                 max_speakers: int = 50,
                 sessions_per_speaker: int = 3,
                 ic_mode: str = "stochastic",
                 seed: int = 42):
        self.max_speakers = max_speakers
        self.sessions_per_speaker = sessions_per_speaker
        self.ic_mode   = ic_mode
        self._feat_ext  = BehaviorFeatureExtractor()
        self._ic_sim    = ICSimulator(seed=seed)

    def load(self) -> list[MultiSessionUser]:
        try:
            return self._load_hf()
        except Exception as e:
            print(f"[MSCLoader] HF download failed ({e}). Using stub.")
            return self._load_stub()

    def _load_hf(self) -> list[MultiSessionUser]:
        from datasets import load_dataset
        # Use blended_skill_talk as MSC proxy (available without auth)
        ds = load_dataset("blended_skill_talk", split="train",
                          trust_remote_code=True)

        users: list[MultiSessionUser] = []
        for speaker_idx in range(min(self.max_speakers, len(ds))):
            row = ds[speaker_idx]
            uid = f"msc_user_{speaker_idx:04d}"
            user = MultiSessionUser(user_id=uid)

            # Simulate multiple sessions by splitting the conversation
            dialogue = row.get("free_messages", []) + row.get("guided_messages", [])
            if len(dialogue) < 2:
                dialogue = ["Hello.", "Hi there!"]

            chunk_size = max(len(dialogue) // self.sessions_per_speaker, 2)
            for sess_idx in range(self.sessions_per_speaker):
                start = sess_idx * chunk_size
                end   = start + chunk_size
                sess_utts = dialogue[start:end]
                if not sess_utts:
                    break
                session = self._build_session(uid, sess_idx, sess_utts)
                user.sessions.append(session)
            users.append(user)
        return users

    def _build_session(self, uid: str, sess_idx: int,
                       utterances: list[str]) -> list[TurnSample]:
        n = len(utterances)
        session: list[TurnSample] = []
        for i, utt in enumerate(utterances):
            sid = f"{uid}_sess{sess_idx}"
            rng = np.random.default_rng(hash(f"{uid}_{sess_idx}_{i}") % (2**31))
            valence = float(np.clip(0.2 + rng.normal(0, 0.15), -1, 1))
            arousal = float(np.clip(0.4 + rng.normal(0, 0.10),  0, 1))
            bvec = self._feat_ext.extract_turn(utt, i, n)
            ic = self._ic_sim.get_ic(uid, turn_index=i + sess_idx * n,
                                     mode=self.ic_mode)
            turn = TurnSample(
                conv_id=f"{uid}_s{sess_idx}", turn_index=i,
                speaker_id=uid, text=utt,
                emotion_label="neutral", valence=valence, arousal=arousal,
                behavior_vec=bvec, ic_score=ic,
                session_boundary=(i == 0),
                is_last_turn=(i == n - 1),
            )
            session.append(turn)
        return session

    def _load_stub(self) -> list[MultiSessionUser]:
        stub = [
            ("alice", [
                ["I've been coming here for a while now.", "Yes, I remember you!"],
                ["Good to be back. Same as usual.", "Of course, I'll set it up."],
                ["Thanks, really appreciate the consistency.", "Always happy to help."],
            ]),
            ("bob", [
                ["First time here, not sure what to expect.", "Welcome! Let me explain."],
                ["Second visit, feels more familiar already.", "Glad to hear it."],
                ["I think I'm getting the hang of it.", "You're doing great."],
            ]),
        ]
        users = []
        for uid, sessions_raw in stub:
            user = MultiSessionUser(user_id=uid)
            for sess_idx, utts in enumerate(sessions_raw):
                session = self._build_session(uid, sess_idx, utts)
                user.sessions.append(session)
            users.append(user)
        return users


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PersonaChat ===")
    loader = PersonaLoader(max_convs=3)
    convs = loader.load()
    for c in convs:
        print(f"  {c.conv_id}: {len(c.turns)} turns, emotion={c.emotion_label}")

    print("\n=== MSC ===")
    msc = MSCLoader(max_speakers=2, sessions_per_speaker=3)
    users = msc.load()
    for u in users:
        print(f"  {u.user_id}: {len(u.sessions)} sessions, "
              f"{sum(len(s) for s in u.sessions)} total turns")
