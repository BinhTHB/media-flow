"""Regression tests: numbered-line parsing must survive chatty model output.

Gemma-style responses broke the old parser three ways: unparsed lines fell
back to the SOURCE text (rendering original-language captions with literal
"\\n"), index prefixes leaked into translations ("3: Đôi tay nhẹ nhàng"), and
"N." bullets were not recognized at all."""

from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter

parse = GeminiTranslationAdapter._parse_numbered_lines


def test_unparsed_lines_return_empty_not_source():
    # A chatty response with no numbered lines must NOT yield the source text
    # as a "translation" — the fail-safe pass suppresses those pixels instead.
    out = parse("I need to transcribe the text from the image.\nThe image has two lines of text.", ["你好", "世界"])
    assert out == ["", ""]


def test_index_prefix_never_leaks_into_translation():
    content = "0: 0: Đôi tay nhẹ nhàng\n1: 1: Ánh mắt dịu dàng"
    out = parse(content, ["双手轻柔", "soft eyes"])
    assert out[0] == "Đôi tay nhẹ nhàng"
    assert out[1] == "Ánh mắt dịu dàng"


def test_dot_bullets_and_dash_bullets_are_accepted():
    content = "- 0: Một: Hãy dịu dàng\n1. Hai: Đoan trang"
    out = parse(content, ["第一条：要温柔", "第二条：要端庄"])
    assert out[0] == "Một: Hãy dịu dàng"
    assert out[1] == "Hai: Đoan trang"


def test_well_formed_output_still_parses():
    out = parse("0: Mẹ luôn dạy chúng con\n1: Điều thứ nhất", ["妈妈总是教导我们", "第一条"])
    assert out == ["Mẹ luôn dạy chúng con", "Điều thứ nhất"]


def test_backtick_source_echo_with_arrow_keeps_first_option():
    # Real gemma-4-26b output: `source` -> "option A" or "option B".
    content = '0: `第一条: 要温柔\\nOne: be gentle` -> "Phải thật dịu dàng" or "Hãy nhẹ nhàng".'
    out = parse(content, ["第一条: 要温柔\nOne: be gentle"])
    assert out[0] == "Phải thật dịu dàng"


def test_arrow_echo_with_same_as_note_keeps_quoted_translation():
    content = '0: `轻声细语\\nsoft voice` -> (Same as 0) "Phải thật dịu dàng".'
    # The sentence period lives outside the quotes; unwrapping drops it.
    out = parse(content, ["轻声细语\nsoft voice"])
    assert out[0] == "Phải thật dịu dàng"


def test_plain_english_echo_is_stripped_via_backtick():
    content = '0: `soft eyes` -> "Ánh mắt dịu dàng".'
    out = parse(content, ["soft eyes"])
    assert out[0] == "Ánh mắt dịu dàng"


def test_quoted_speech_inside_sentence_is_preserved():
    content = '0: Anh nói "đi" rồi bỏ đi.'
    out = parse(content, ['他说完"走"就离开了'])
    # Decorative trailing punctuation is stripped for captions; inner quotes stay.
    assert out[0] == 'Anh nói "đi" rồi bỏ đi'


def test_caption_with_literal_arrow_is_preserved():
    # No backtick and no CJK before the arrow -> not an echo; keep the caption.
    content = "0: A -> B"
    out = parse(content, ["A -> B"])
    assert out[0] == "A -> B"


def test_entries_glued_by_backtick_on_one_line_are_split():
    # Real gemma-4-26b output: a re-echo of the entry is glued to the answer
    # with a backtick on the same physical line.
    content = "0: Một quý cô chuẩn mực phải tuân thủ những quy tắc này`0: Một quý cô chu"
    out = parse(content, ["一位合格的淑女必须遵循以下规则"])
    assert out == ["Một quý cô chuẩn mực phải tuân thủ những quy tắc này"]


def test_entries_glued_by_period_on_one_line_are_split():
    content = "0: Một quý cô chuẩn mực phải tuân thủ các quy tắc sau đây.0: Một quý cô chuẩn"
    out = parse(content, ["一位合格的淑女必须遵循以下规则"])
    assert out == ["Một quý cô chuẩn mực phải tuân thủ các quy tắc sau đây"]


def test_timestamp_value_is_not_split_as_entry():
    # "0:00" has a digit right after the separator, so it is a time, not an
    # embedded entry marker; the value must survive intact.
    content = "0: Từ 0:00 đến 5:00 là mở đầu"
    out = parse(content, ["开场从0:00到5:00"])
    assert out == ["Từ 0:00 đến 5:00 là mở đầu"]


def test_salvage_extracts_answer_glued_to_prose_line():
    # Real gemma-4-26b single-item response (abbreviated): the chain-of-thought
    # ends with the answer glued to a prose line, unreachable by the anchored
    # regex. The line-anchored echo ("Source: `0: ...`") also appears but the
    # LAST candidate per index must win.
    content = (
        "*   Source: `0: 眼神柔和`\n"
        "    *   Context: single line, gaze/soft.\n"
        "    *   \"Ánh mắt hiền từ\" (possible).\n"
        "    *   Is the format correct? Yes.0: Ánh mắt dịu dàng"
    )
    out = parse(content, ["眼神柔和"])
    assert out == ["Ánh mắt dịu dàng"]


def test_salvage_ignores_out_of_range_indices():
    # CoT bullets reference rule numbers ("Constraint 3: ...") that must not be
    # invented into translations for indices the batch does not have.
    content = (
        "*   Constraint 3: Fluent, concise.\n"
        "*   Rule 7: Format `i: translated_text`.\n"
        "*   Is the format correct? Yes."
    )
    out = parse(content, ["眼神柔和"])
    assert out == [""]


def test_bullet_backtick_numbered_line_parses():
    content = "*   `0: Ánh mắt dịu dàng`"
    out = parse(content, ["眼神柔和"])
    assert out == ["Ánh mắt dịu dàng"]