import random

from database.models import Puzzle


PUZZLES = {
    "Technology": [
        Puzzle(
            word="PYTHON",
            sentence="Many developers use _____ because it has simple syntax and powerful libraries.",
            theme="Technology",
            difficulty="Easy",
            options=[
                "PYTHON",
                "DATABASE",
                "SERVER",
                "ROUTER"
            ],
            explanation="Python is a popular programming language used for software development, AI, and automation."
        ),

        Puzzle(
            word="CLOUD",
            sentence="Companies store their applications and data on the _____ instead of only local servers.",
            theme="Technology",
            difficulty="Easy",
            options=[
                "CLOUD",
                "DESKTOP",
                "KEYBOARD",
                "MONITOR"
            ],
            explanation="Cloud computing provides remote access to computing resources."
        ),

        Puzzle(
            word="DATABASE",
            sentence="Applications store structured information inside a _____.",
            theme="Technology",
            difficulty="Medium",
            options=[
                "DATABASE",
                "KEYBOARD",
                "MONITOR",
                "ROUTER"
            ],
            explanation="A database stores and organizes data."
        ),

        Puzzle(
            word="ALGORITHM",
            sentence="A step-by-step procedure for solving a problem is called an _____.",
            theme="Technology",
            difficulty="Medium",
            options=[
                "ALGORITHM",
                "COMPILER",
                "SERVER",
                "PROCESSOR"
            ],
            explanation="An algorithm is a sequence of instructions for solving a problem."
        ),

        Puzzle(
            word="ENCRYPTION",
            sentence="Sensitive information is protected using _____.",
            theme="Technology",
            difficulty="Hard",
            options=[
                "ENCRYPTION",
                "DEBUGGING",
                "COMPILING",
                "INDEXING"
            ],
            explanation="Encryption secures information by converting it into an unreadable format."
        )
    ],

    "Science": [
        Puzzle(
            word="ATOM",
            sentence="Everything around us is made of tiny particles called _____.",
            theme="Science",
            difficulty="Easy",
            options=[
                "ATOM",
                "CELL",
                "ENERGY",
                "FORCE"
            ],
            explanation="Atoms are the basic units of matter."
        ),

        Puzzle(
            word="CELL",
            sentence="The basic unit of life in living organisms is called a _____.",
            theme="Science",
            difficulty="Easy",
            options=[
                "CELL",
                "ATOM",
                "TISSUE",
                "ORGAN"
            ],
            explanation="Cells are the basic structural and functional units of life."
        ),

        Puzzle(
            word="GRAVITY",
            sentence="The force that attracts objects toward Earth is called _____.",
            theme="Science",
            difficulty="Easy",
            options=[
                "GRAVITY",
                "ENERGY",
                "PRESSURE",
                "MOTION"
            ],
            explanation="Gravity is the force that pulls objects toward each other."
        ),

        Puzzle(
            word="MITOSIS",
            sentence="The process by which a cell divides into two identical cells is called _____.",
            theme="Science",
            difficulty="Medium",
            options=[
                "MITOSIS",
                "EVAPORATION",
                "OXIDATION",
                "FERMENTATION"
            ],
            explanation="Mitosis is responsible for growth and tissue repair."
        ),

        Puzzle(
            word="PHOTOSYNTHESIS",
            sentence="Plants produce food through the process of _____.",
            theme="Science",
            difficulty="Medium",
            options=[
                "PHOTOSYNTHESIS",
                "RESPIRATION",
                "DIGESTION",
                "DIFFUSION"
            ],
            explanation="Photosynthesis uses sunlight to make food."
        ),

        Puzzle(
            word="THERMODYNAMICS",
            sentence="The study of heat, energy, and work is known as _____.",
            theme="Science",
            difficulty="Hard",
            options=[
                "THERMODYNAMICS",
                "ECOLOGY",
                "GENETICS",
                "ASTRONOMY"
            ],
            explanation="Thermodynamics explains the relationship between heat and energy."
        )
    ]
}


def generate_local_puzzle(theme, difficulty, mode):
    """
    Generate a puzzle based on selected theme, difficulty, and game mode.
    """

    if theme not in PUZZLES:
        theme = "Technology"

    puzzles = PUZZLES[theme]

    matching_puzzles = [
        puzzle
        for puzzle in puzzles
        if puzzle.difficulty == difficulty
    ]

    if mode == "Wordle":
        matching_puzzles = [
            puzzle
            for puzzle in matching_puzzles
            if len(puzzle.word) == 6

        ]

    if not matching_puzzles:
        if mode == "Wordle":
            matching_puzzles = [
                puzzle
                for puzzle in puzzles
                if len(puzzle.word) == 6
            ]

        else:
            matching_puzzles = puzzles

    if not matching_puzzles:
        return None

    selected_puzzle = random.choice(matching_puzzles)

    # Currently both game modes use the same puzzle object.
    # Later, when the AI generator is implemented, this section
    # can generate different puzzle formats for MCQ and Wordle.
    if mode == "MCQ":
        return selected_puzzle

    elif mode == "Wordle":
        return selected_puzzle

    # Fallback
    return selected_puzzle