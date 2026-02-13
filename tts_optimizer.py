"""
TTS parameter optimization for different content types.

Adjusts XTTS v2 generation parameters based on text characteristics
(dialogue, questions, lists) for more natural-sounding speech.
"""


class TTSOptimizer:
    """Optimize TTS parameters based on text characteristics."""

    @staticmethod
    def get_optimal_params(text: str, base_speed: float = 1.0) -> dict:
        """
        Get optimal TTS parameters for given text.

        Adjusts temperature, top_p, and speed based on whether the text
        contains dialogue, questions, or list-like content.

        Args:
            text: The text chunk to analyze
            base_speed: Base speech speed multiplier

        Returns:
            Dict with TTS generation parameters
        """
        params = {
            'temperature': 0.55,
            'top_p': 0.85,
            'repetition_penalty': 1.5,
            'speed': base_speed,
        }

        # Dialogue: allow more expressive variation
        if '"' in text or '\u201e' in text or '\u201c' in text:
            params['temperature'] = 0.60
            params['top_p'] = 0.90

        # Questions: slightly slower for emphasis
        if '?' in text:
            params['speed'] = base_speed * 0.95

        # Lists/enumerations: more deliberate pacing
        if text.count('\n') > 2 or text.count(',') > 5:
            params['speed'] = base_speed * 0.90
            params['temperature'] = 0.60

        return params
