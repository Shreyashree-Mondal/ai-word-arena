from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Puzzle:
    word: str
    sentence: str
    theme: str
    difficulty: str
    options: List[str] = field(default_factory=list)
    explanation: str = ""
    hint: str = ""
    source: str = ""  # "ai" | "bank" | "local"


@dataclass
class Player:
    name: str
    total_score: int = 0
    current_streak: int = 0
    best_streak: int = 0
    favorite_theme: str = ""
    games_played: int = 0
    correct_answers: int = 0
    hints_used: int = 0
    # theme -> {"played": int, "correct": int}
    theme_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class Theme:
    name: str
    times_played: int = 0
    success_rate: float = 0.0


@dataclass
class GameResult:
    correct: bool
    points: int
    message: str
