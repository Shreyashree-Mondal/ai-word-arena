class DifficultyManager:

    def __init__(self):
        self.level = 1
        self.correct_streak = 0
        self.wrong_count = 0


    def record_result(self, correct: bool):
        """
        Update difficulty based on player's performance.
        """

        if correct:
            self.correct_streak += 1
            self.wrong_count = 0

            # Increase difficulty after 3 consecutive correct answers
            if self.correct_streak >= 3:
                self.increase_difficulty()
                self.correct_streak = 0

        else:
            self.wrong_count += 1
            self.correct_streak = 0

            # Reduce difficulty after 2 mistakes
            if self.wrong_count >= 2:
                self.decrease_difficulty()
                self.wrong_count = 0


    def increase_difficulty(self):
        if self.level < 3:
            self.level += 1


    def decrease_difficulty(self):
        if self.level > 1:
            self.level -= 1


    def get_level(self):
        return self.level