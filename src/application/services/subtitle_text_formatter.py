class SubtitleTextFormatter:
    @staticmethod
    def wrap_lines(text: str, max_chars_per_line: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        lines = []
        current = []
        current_len = 0
        for word in words:
            add_len = len(word) + (1 if current else 0)
            if current and current_len + add_len > max_chars_per_line:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += add_len
        if current:
            lines.append(" ".join(current))
        return lines

    @staticmethod
    def format_text(text: str, max_chars_per_line: int, max_lines_per_subtitle: int | None = None) -> str:
        lines = SubtitleTextFormatter.wrap_lines(text, max_chars_per_line)
        return "\n".join(lines)
