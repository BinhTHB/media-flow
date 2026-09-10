from scripts.run_final_dubbing import _source_gate, _target_audio_gate


def test_source_gate_rejects_unexpected_latin_token_inside_cjk():
    gate = _source_gate(
        "有个外卖我送错",
        "有个 WIFI 我送错了",
    )
    assert gate["passed"] is False
    assert gate["unexpected_latin_tokens"] == ["wifi"]


def test_source_gate_accepts_clean_cjk_transcript():
    gate = _source_gate(
        "你好 请问这里是咖啡店吗 是的欢迎光临",
        "你好，请问这里是咖啡店吗？是的，欢迎光临。",
    )
    assert gate["passed"] is True
    assert gate["unexpected_latin_tokens"] == []


def test_source_gate_accepts_corrected_keyterm_even_when_raw_script_similarity_is_low():
    gate = _source_gate(
        "好 謝謝 拜拜 謝謝你了 不客氣 對不起 有个外卖我送错",
        "我有喝。谢谢你，谢谢你。谢谢你了。拜拜。不客气。对不起，对不起，有个外卖我送错了。",
        ["有个外卖我送错"],
    )
    assert gate["passed"] is True
    assert gate["critical_keyterm_coverage"] == 1.0
    assert gate["unexpected_latin_tokens"] == []


def test_target_audio_gate_accepts_close_reasr():
    gate = _target_audio_gate(
        "Tôi có uống. Cảm ơn nha. Bye bye. Không có gì. Xin lỗi. Tôi giao nhầm một đơn đồ ăn rồi.",
        "Tôi có uống cảm ơn nha bye bye không có gì xin lỗi tôi giao nhầm một đơn đồ ăn rồi",
    )
    assert gate["passed"] is True
    assert gate["similarity"] >= 0.80
