"""Spoken lines share a short batch. One failed batch is retried line by line."""

from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter


class _Scheduler:
    def __init__(self) -> None:
        self.max_retries = 3


class _Client:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._timeout = 180.0

    def generate_content(self, request):
        self.prompts.append(request.contents[0]["parts"][0]["text"])

        class _Response:
            text = "0: Xin chào\n1: Xin chào\n2: Xin chào"

        return _Response()


class _Runtime:
    def __init__(self, client: _Client) -> None:
        self._client = client
        self._scheduler = _Scheduler()

    def get_client(self, model=None):
        return self._client


def test_spoken_lines_share_one_short_call():
    client = _Client()
    runtime = _Runtime(client)
    adapter = GeminiTranslationAdapter(runtime=runtime, purpose="spoken")
    result = adapter.translate_texts(["你好", "再见", "谢谢"], target_language="vi")
    assert result == ["Xin chào", "Xin chào", "Xin chào"]
    assert len(client.prompts) == 1
    assert client._timeout == 70.0
    assert runtime._scheduler.max_retries == 1
    assert "Translation guidelines" not in client.prompts[0]
    assert "你好" in client.prompts[0]
    assert "再见" in client.prompts[0]
    assert "谢谢" in client.prompts[0]


def test_one_spoken_timeout_does_not_drop_the_other_lines():
    class _Client:
        def __init__(self) -> None:
            self.n = 0
            self._timeout = 180.0

        def generate_content(self, request):
            text = request.contents[0]["parts"][0]["text"]
            if "你好" in text:
                raise RuntimeError("The read operation timed out")

            class _Response:
                text = "0: Tạm biệt"

            return _Response()

    adapter = GeminiTranslationAdapter(runtime=_Runtime(_Client()), purpose="spoken")
    assert adapter.translate_texts(["你好", "再见"], target_language="vi") == ["", "Tạm biệt"]


def test_one_spoken_line_without_an_index_is_still_the_translation():
    class _Client:
        def __init__(self) -> None:
            self._timeout = 180.0

        def generate_content(self, request):
            class _Response:
                text = "Xin chào bạn"

            return _Response()

    adapter = GeminiTranslationAdapter(runtime=_Runtime(_Client()), purpose="spoken")
    assert adapter.translate_texts(["hello"], target_language="vi") == ["Xin chào bạn"]


def test_spoken_reasoning_response_is_not_dubbed():
    # The bare-line salvage must accept a plain answer only. Multi-line
    # chain-of-thought output has no translation to keep.
    class _Client:
        def __init__(self) -> None:
            self._timeout = 180.0

        def generate_content(self, request):
            class _Response:
                text = (
                    "*   Constraint 3: Fluent, concise.\n"
                    "*   Rule 7: Format `i: translated_text`.\n"
                    "*   Is the format correct? Yes."
                )

            return _Response()

    adapter = GeminiTranslationAdapter(runtime=_Runtime(_Client()), purpose="spoken")
    assert adapter.translate_texts(["hello"], target_language="vi") == [""]


def test_router_repairs_suspicious_lines_in_one_call():
    from src.infrastructure.translation.router import FallbackTranslationAdapter

    class _Provider:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            self.calls.append(list(texts))
            if len(texts) == 3:
                return ["Xin chào", "Preserve line numbers (`i: translated_text`).", "Tạm biệt"]
            return ["Dòng giữa"]

    provider = _Provider()
    result = FallbackTranslationAdapter([("gemini", provider)]).translate_texts(
        ["你好", "中间", "再见"], target_language="vi"
    )
    assert result == ["Xin chào", "Dòng giữa", "Tạm biệt"]
    assert provider.calls == [["你好", "中间", "再见"], ["中间"]]


def test_router_keeps_a_partial_spoken_batch():
    from src.infrastructure.translation.router import FallbackTranslationAdapter

    class _Provider:
        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            return ["Xin chào", ""]

    result = FallbackTranslationAdapter([("gemini", _Provider())]).translate_texts(
        ["你好", "再见"], target_language="vi"
    )
    assert result == ["Xin chào", ""]

