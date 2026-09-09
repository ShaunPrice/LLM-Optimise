from llm_optimise.context import select_context


def test_adjacent_chunks_keep_original_span_and_cross_boundary_fact():
    result = select_context(
        "alpha beta",
        {"notes.txt": "alpha start\nbeta finish\n"},
        chunk_lines=1,
        required_facts=["alpha start\nbeta finish"],
    )
    assert result["required_fact_gate"]
    assert result["selected"] == [
        {
            **result["selected"][0],
            "path": "notes.txt",
            "start_line": 1,
            "end_line": 2,
            "content": "alpha start\nbeta finish\n",
        }
    ]
    assert "alpha start\nbeta finish" in result["text"]


def test_gaps_and_source_boundaries_cannot_manufacture_facts():
    result = select_context(
        "alpha beta",
        {"a.txt": "alpha\nignored\nbeta\n", "b.txt": "beta\n"},
        chunk_lines=1,
        required_facts=["alpha\nbeta"],
    )
    assert not result["required_fact_gate"]
    assert result["required_facts_absent_from_sources"] == ["alpha\nbeta"]
    assert all(not (s["start_line"] == 1 and s["end_line"] == 3) for s in result["selected"])


def test_labels_do_not_count_as_required_facts():
    result = select_context(
        "useful", {"invented_fact.txt": "useful source\n"}, required_facts=["invented_fact"]
    )
    assert result["required_facts_missing"] == ["invented_fact"]


def test_zero_relevance_chunks_are_not_padding():
    result = select_context("needle", {"a": "unrelated\ncontent\n"}, chunk_lines=1)
    assert result["text"] == "" and result["selected"] == []
    assert result["zero_relevance_chunks_skipped"] == 2
    assert result["output_chars"] == 0


def test_relevant_but_oversize_chunk_is_distinct_from_absent_fact():
    text = "needle " * 100 + "known required fact"
    result = select_context(
        "needle", {"a": text}, max_chars=256, required_facts=["known required fact", "absent"]
    )
    assert result["required_facts_missing"] == ["known required fact", "absent"]
    assert result["required_facts_absent_from_sources"] == ["absent"]
    assert result["unselected_relevant_chunks"] == 1


def test_selection_hashes_and_budgets_remain_reproducible():
    files = {"a": "beta\nalpha\nother\n", "b": "alpha\n"}
    first = select_context("alpha beta", files, chunk_lines=1, max_chars=256)
    second = select_context("alpha beta", files, chunk_lines=1, max_chars=256)
    assert first["text"] == second["text"]
    assert first["source_hashes"] == second["source_hashes"]
    assert first["output_chars"] <= 256
