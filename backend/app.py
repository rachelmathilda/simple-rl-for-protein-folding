import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ─────────────────────────────────────────────
# CONSTANTS (must match training code)
# ─────────────────────────────────────────────

DIRECTION_DELTAS = {
    "E": ( 1,  0),
    "N": ( 0,  1),
    "W": (-1,  0),
    "S": ( 0, -1),
}

TURN_MAP = {
    "E": {"F": "E", "L": "N", "R": "S"},
    "N": {"F": "N", "L": "W", "R": "E"},
    "W": {"F": "W", "L": "S", "R": "N"},
    "S": {"F": "S", "L": "E", "R": "W"},
}

IDX2ACTION = {0: "F", 1: "L", 2: "R"}

BENCHMARKS = {
    "HPHPPHHPHP"                                      : -4,
    "HPHPPHHPHPPHPHHPPHPH"                            : -9,
    "HHHPPHPHPHPPHPHPHPPH"                            : -10,
    "HHPPHPPHPPHPPHPPHPPHPPHH"                        : -9,
    "PPHPPHHPPPPHHPPPPHHPPPPHH"                       : -8,
    "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP"            : -14,
    "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH": -23,
}

# ─────────────────────────────────────────────
# ACTOR-CRITIC NETWORK (must match training)
# ─────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, n_actions: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


# ─────────────────────────────────────────────
# HP ENVIRONMENT (self-contained, no gym dep)
# ─────────────────────────────────────────────

class HPFoldingEnv:
    def __init__(self, sequence: str):
        assert all(c in "HP" for c in sequence)
        self.sequence  = sequence
        self.n         = len(sequence)
        self.origin    = (self.n, self.n)
        self.reset()

    def reset(self):
        p0 = self.origin
        p1 = (self.origin[0] + 1, self.origin[1])
        self.positions  = [p0, p1]
        self.occupied   = {p0, p1}
        self.facing     = "E"
        self.step_idx   = 2
        self.done       = False
        self.rel_moves  = []
        return self._encode()

    def _encode(self) -> np.ndarray:
        vec = np.zeros((self.n - 2) * 4, dtype=np.float32)
        for t, act in enumerate(self.rel_moves):
            base         = t * 4
            vec[base+act] = 1.0
            res_idx = t + 2
            if res_idx < self.n and self.sequence[res_idx] == "H":
                vec[base+3] = 1.0
        return vec

    def get_valid_actions(self) -> list[int]:
        if self.done:
            return []
        tip   = self.positions[-1]
        valid = []
        for idx, act in IDX2ACTION.items():
            nf     = TURN_MAP[self.facing][act]
            dx, dy = DIRECTION_DELTAS[nf]
            if (tip[0]+dx, tip[1]+dy) not in self.occupied:
                valid.append(idx)
        return valid

    def step(self, action: int):
        act_str    = IDX2ACTION[action]
        new_facing = TURN_MAP[self.facing][act_str]
        dx, dy     = DIRECTION_DELTAS[new_facing]
        tip        = self.positions[-1]
        new_pos    = (tip[0]+dx, tip[1]+dy)

        if new_pos in self.occupied:
            self.done = True
            return self._encode(), True

        self.positions.append(new_pos)
        self.occupied.add(new_pos)
        self.facing   = new_facing
        self.rel_moves.append(action)
        self.step_idx += 1

        if self.step_idx == self.n:
            self.done = True
        elif not self.get_valid_actions():
            self.done = True

        return self._encode(), self.done

    def compute_energy(self) -> int:
        pos_to_idx = {pos: i for i, pos in enumerate(self.positions)}
        contacts   = 0
        for i, pos in enumerate(self.positions):
            if self.sequence[i] != "H":
                continue
            x, y = pos
            for dx, dy in DIRECTION_DELTAS.values():
                nb = (x+dx, y+dy)
                j  = pos_to_idx.get(nb)
                if j is not None and abs(i-j) > 1 and self.sequence[j] == "H":
                    contacts += 1
        return -(contacts // 2)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

_model_cache: dict[int, ActorCritic] = {}

def load_model(sequence_length: int) -> ActorCritic:
    if sequence_length in _model_cache:
        return _model_cache[sequence_length]

    state_dim  = (sequence_length - 2) * 4
    model      = ActorCritic(state_dim=state_dim)
    model_path = "model.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "model.pt not found. Train the PPO agent from the notebook "
            "and upload model.pt to this Space."
        )

    checkpoint = torch.load(model_path, map_location="cpu")

    # Support both raw state_dict and wrapped checkpoint
    if isinstance(checkpoint, dict) and "ac_state_dict" in checkpoint:
        state_dict    = checkpoint["ac_state_dict"]
        saved_seq_len = checkpoint.get("sequence_length")
        if saved_seq_len is not None and saved_seq_len != sequence_length:
            raise ValueError(
                f"model.pt was trained on sequence length {saved_seq_len}, "
                f"but request is for length {sequence_length}."
            )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    _model_cache[sequence_length] = model
    return model


# ─────────────────────────────────────────────
# GREEDY ROLLOUT
# ─────────────────────────────────────────────

def greedy_rollout(sequence: str, n_rollouts: int = 64) -> dict:
    """
    Run n_rollouts greedy episodes, return the best conformation found.
    Uses stochastic sampling (temperature=1) rather than pure argmax
    to get diversity across rollouts while still being fast.
    """
    model     = load_model(len(sequence))
    best_energy    = 1
    best_positions = None
    all_energies   = []

    for _ in range(n_rollouts):
        env   = HPFoldingEnv(sequence)
        state = env.reset()
        done  = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break

            with torch.no_grad():
                s    = torch.FloatTensor(state).unsqueeze(0)
                logits, _ = model(s)
                logits = logits.squeeze(0)

            # Mask invalid actions
            mask = torch.full((3,), -1e9)
            for a in valid:
                mask[a] = 0.0
            masked_logits = logits + mask

            # Greedy argmax
            action = masked_logits.argmax().item()
            state, done = env.step(action)

        energy = env.compute_energy()
        all_energies.append(energy)

        if energy < best_energy:
            best_energy    = energy
            best_positions = list(env.positions)

    if best_positions is None:
        env   = HPFoldingEnv(sequence)
        state = env.reset()
        best_positions = list(env.positions)
        best_energy    = env.compute_energy()

    # Normalize positions to start near (0,0) for cleaner frontend rendering
    if best_positions:
        min_x = min(p[0] for p in best_positions)
        min_y = min(p[1] for p in best_positions)
        best_positions = [(p[0]-min_x, p[1]-min_y) for p in best_positions]

    # Compute H-H contacts for frontend highlighting
    pos_to_idx = {pos: i for i, pos in enumerate(best_positions)}
    contacts   = []
    for i, pos in enumerate(best_positions):
        if sequence[i] != "H":
            continue
        x, y = pos
        for dx, dy in DIRECTION_DELTAS.values():
            nb = (x+dx, y+dy)
            j  = pos_to_idx.get(nb)
            if j is not None and abs(i-j) > 1 and sequence[j] == "H":
                pair = (min(i,j), max(i,j))
                if pair not in contacts:
                    contacts.append(pair)

    return {
        "sequence"     : sequence,
        "energy"       : best_energy,
        "optimal_energy": BENCHMARKS.get(sequence),
        "positions"    : best_positions,
        "contacts"     : contacts,
        "n_rollouts"   : n_rollouts,
        "energy_history": all_energies,
    }


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="HP Model Protein Folding API",
    description="PPO-based HP lattice protein folding inference. Deploy model.pt from notebook training.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SCHEMAS ────────────────────────────────────

class FoldRequest(BaseModel):
    sequence  : str
    n_rollouts: int = 64

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("Sequence cannot be empty.")
        if not all(c in "HP" for c in v):
            raise ValueError("Sequence must contain only H and P characters.")
        if len(v) < 6:
            raise ValueError("Sequence must be at least 6 residues long.")
        if len(v) > 64:
            raise ValueError("Sequence length must be 64 or fewer for this deployment.")
        return v

    @field_validator("n_rollouts")
    @classmethod
    def validate_rollouts(cls, v: int) -> int:
        if v < 1 or v > 256:
            raise ValueError("n_rollouts must be between 1 and 256.")
        return v


class FoldResponse(BaseModel):
    sequence       : str
    energy         : int
    optimal_energy : int | None
    positions      : list[list[int]]
    contacts       : list[list[int]]
    n_rollouts     : int
    energy_history : list[int]


# ── ENDPOINTS ──────────────────────────────────

@app.get("/")
def root():
    return {
        "message"   : "HP Model Protein Folding API",
        "algorithm" : "PPO (Proximal Policy Optimization)",
        "endpoints" : ["/fold", "/benchmarks", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/benchmarks")
def get_benchmarks():
    return {
        "sequences": [
            {
                "sequence"      : seq,
                "length"        : len(seq),
                "optimal_energy": e,
            }
            for seq, e in BENCHMARKS.items()
        ]
    }


@app.post("/fold", response_model=FoldResponse)
def fold(req: FoldRequest):
    try:
        result = greedy_rollout(req.sequence, req.n_rollouts)
        return FoldResponse(
            sequence       = result["sequence"],
            energy         = result["energy"],
            optimal_energy = result["optimal_energy"],
            positions      = [list(p) for p in result["positions"]],
            contacts       = [list(c) for c in result["contacts"]],
            n_rollouts     = result["n_rollouts"],
            energy_history = result["energy_history"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    