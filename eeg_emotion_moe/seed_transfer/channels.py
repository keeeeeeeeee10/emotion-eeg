from __future__ import annotations

from dataclasses import dataclass


SEED_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4",
    "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1",
    "O1", "OZ", "O2", "CB2",
]

CONTEST_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]

# The contest uses older 10-20 names for temporal electrodes.
CONTEST_TO_SEED_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


@dataclass(frozen=True)
class ChannelMapping:
    contest_channels: list[str]
    seed_indices: list[int]
    seed_names: list[str]


def seed_to_contest_mapping() -> ChannelMapping:
    seed_index = {name: idx for idx, name in enumerate(SEED_CHANNELS)}
    seed_indices: list[int] = []
    seed_names: list[str] = []
    for contest_name in CONTEST_CHANNELS:
        seed_name = CONTEST_TO_SEED_ALIASES.get(contest_name, contest_name)
        if seed_name not in seed_index:
            raise KeyError(f"Missing SEED channel for contest channel {contest_name}")
        seed_indices.append(seed_index[seed_name])
        seed_names.append(seed_name)
    return ChannelMapping(
        contest_channels=list(CONTEST_CHANNELS),
        seed_indices=seed_indices,
        seed_names=seed_names,
    )

