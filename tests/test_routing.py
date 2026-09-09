"""Tests for the deterministic routing engine (stdlib only, no network)."""

import math

import pytest

from llm_optimise.routing import ProviderModel, RoutePolicy, choose_route


def make_model(**overrides):
    base = dict(
        id="local-code",
        model="qwen2.5-coder-14b",
        provider="openai",
        location="local",
        base_url="http://127.0.0.1:11434/v1",
        context_window=32_768,
        max_output_tokens=8_192,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        latency_ms=900.0,
        quality=0.6,
    )
    base.update(overrides)
    return ProviderModel(**base)


def cloud_model(**overrides):
    base = dict(
        id="cloud-sonnet",
        model="claude-sonnet-5",
        provider="anthropic",
        location="cloud",
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        context_window=200_000,
        max_output_tokens=64_000,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        latency_ms=1_500.0,
        quality=0.9,
    )
    base.update(overrides)
    return ProviderModel(**base)


# ---------------------------------------------------------------- validation


class TestProviderModelValidation:
    def test_valid_models_construct(self):
        assert make_model().capabilities == ("chat", "code")
        assert cloud_model().api_key_env == "ANTHROPIC_API_KEY"

    @pytest.mark.parametrize("bad", ["", "   ", None, 7])
    def test_rejects_bad_id(self, bad):
        with pytest.raises(ValueError):
            make_model(id=bad)

    def test_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="provider"):
            make_model(provider="mistral")

    def test_rejects_unknown_location(self):
        with pytest.raises(ValueError, match="location"):
            make_model(location="edge")

    @pytest.mark.parametrize("field", ["context_window", "max_output_tokens"])
    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, None])
    def test_rejects_non_positive_int_windows(self, field, bad):
        with pytest.raises(ValueError):
            make_model(**{field: bad})

    def test_rejects_output_exceeding_context(self):
        with pytest.raises(ValueError, match="max_output_tokens"):
            make_model(context_window=1000, max_output_tokens=2000)

    @pytest.mark.parametrize(
        "field", ["input_cost_per_million", "output_cost_per_million", "latency_ms"]
    )
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, "3"])
    def test_rejects_non_finite_or_negative_metrics(self, field, bad):
        with pytest.raises(ValueError):
            make_model(**{field: bad})

    def test_rejects_zero_latency(self):
        with pytest.raises(ValueError, match="latency_ms"):
            make_model(latency_ms=0.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True])
    def test_rejects_quality_out_of_range(self, bad):
        with pytest.raises(ValueError):
            make_model(quality=bad)

    def test_zero_cost_local_is_a_valid_estimate(self):
        m = make_model(input_cost_per_million=0.0, output_cost_per_million=0.0)
        assert m.estimated_cost_usd(1000, 1000) == 0.0

    @pytest.mark.parametrize(
        "bad_url",
        [
            "ftp://host/api",
            "not a url",
            "http://",
            "https://user:secret@api.example.com",
            "https://user@api.example.com",
        ],
    )
    def test_rejects_bad_urls(self, bad_url):
        with pytest.raises(ValueError):
            make_model(base_url=bad_url)

    def test_rejects_plaintext_cloud_url(self):
        with pytest.raises(ValueError, match="https"):
            cloud_model(base_url="http://api.example.com/v1")

    @pytest.mark.parametrize(
        "url",
        [
            "https://localhost:8443/v1",
            "https://127.0.0.1:8443/v1",
            "https://myserver.local/v1",
        ],
    )
    def test_rejects_localhost_labelled_cloud(self, url):
        with pytest.raises(ValueError, match="mislabelled"):
            cloud_model(base_url=url)

    def test_local_http_loopback_is_allowed(self):
        assert make_model(base_url="http://localhost:1234/v1").location == "local"

    @pytest.mark.parametrize("bad", [(), ("",), ["chat"], ("chat", 3)])
    def test_rejects_bad_capabilities(self, bad):
        with pytest.raises(ValueError):
            make_model(capabilities=bad)

    def test_public_dict_has_no_credential_material(self):
        d = cloud_model().to_public_dict()
        assert "api_key_env" not in d
        assert "@" not in d["base_url"]
        assert d["model_id"] if "model_id" in d else d["id"] == "cloud-sonnet"

    def test_partial_price_means_unknown_cost(self):
        m = make_model(input_cost_per_million=1.0, output_cost_per_million=None)
        assert m.estimated_cost_usd(1000, 1000) is None


class TestRoutePolicyValidation:
    def test_defaults(self):
        p = RoutePolicy()
        assert (p.placement, p.objective) == ("mixed", "balanced")

    def test_rejects_bad_placement_and_objective(self):
        with pytest.raises(ValueError, match="placement"):
            RoutePolicy(placement="hybrid")
        with pytest.raises(ValueError, match="objective"):
            RoutePolicy(objective="speed")

    @pytest.mark.parametrize("field", ["max_cost_usd", "max_latency_ms"])
    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True])
    def test_rejects_bad_budgets(self, field, bad):
        with pytest.raises(ValueError):
            RoutePolicy(**{field: bad})

    def test_zero_budget_only_allows_known_free_models(self):
        assert RoutePolicy(max_cost_usd=0).max_cost_usd == 0
        with pytest.raises(ValueError):
            RoutePolicy(max_latency_ms=0)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
    def test_rejects_bad_min_quality(self, bad):
        with pytest.raises(ValueError):
            RoutePolicy(min_quality=bad)


# ------------------------------------------------------------------ routing


class TestChooseRouteInputs:
    @pytest.mark.parametrize("bad", [0, -5, 1.5, True, None])
    def test_rejects_bad_token_counts(self, bad):
        with pytest.raises(ValueError):
            choose_route([make_model()], RoutePolicy(), bad, 100)
        with pytest.raises(ValueError):
            choose_route([make_model()], RoutePolicy(), 100, bad)

    def test_rejects_empty_model_list(self):
        with pytest.raises(ValueError, match="no models"):
            choose_route([], RoutePolicy(), 100, 100)

    def test_rejects_duplicate_ids(self):
        with pytest.raises(ValueError, match="duplicate"):
            choose_route([make_model(), make_model()], RoutePolicy(), 100, 100)

    def test_rejects_non_policy(self):
        with pytest.raises(ValueError, match="RoutePolicy"):
            choose_route([make_model()], {"objective": "cost"}, 100, 100)


class TestCostObjective:
    def test_picks_cheapest_known_cost(self):
        cheap = cloud_model(id="cheap", input_cost_per_million=0.5, output_cost_per_million=1.0)
        dear = cloud_model(id="dear")
        result = choose_route([dear, cheap], RoutePolicy(objective="cost"), 1000, 500)
        assert result["model_id"] == "cheap"
        assert result["estimated_cost_usd"] == pytest.approx((1000 * 0.5 + 500 * 1.0) / 1_000_000)
        assert [a["model_id"] for a in result["alternatives"]] == ["dear"]

    def test_unknown_price_is_not_free(self):
        unknown = cloud_model(
            id="unknown", input_cost_per_million=None, output_cost_per_million=None
        )
        priced = cloud_model(id="priced")
        result = choose_route([unknown, priced], RoutePolicy(objective="cost"), 1000, 500)
        assert result["model_id"] == "priced"
        assert all(a["model_id"] != "unknown" for a in result["alternatives"])

    def test_only_unknown_prices_raises(self):
        unknown = cloud_model(
            id="unknown", input_cost_per_million=None, output_cost_per_million=None
        )
        with pytest.raises(ValueError, match="not treated as free"):
            choose_route([unknown], RoutePolicy(objective="cost"), 1000, 500)

    def test_explicit_zero_local_cost_beats_paid_cloud_and_is_labelled(self):
        result = choose_route(
            [cloud_model(), make_model()], RoutePolicy(objective="cost"), 1000, 500
        )
        assert result["model_id"] == "local-code"
        assert result["estimated_cost_usd"] == 0.0
        assert any("electricity" in r for r in result["reasons"])

    def test_cost_estimate_arithmetic(self):
        m = cloud_model(input_cost_per_million=3.0, output_cost_per_million=15.0)
        result = choose_route([m], RoutePolicy(objective="cost"), 200_000 - 64_000, 64_000)
        expected = (136_000 * 3.0 + 64_000 * 15.0) / 1_000_000
        assert result["estimated_cost_usd"] == pytest.approx(expected)


class TestBudgets:
    def test_unknown_cost_fails_cost_budget(self):
        unknown = cloud_model(
            id="unknown", input_cost_per_million=None, output_cost_per_million=None
        )
        with pytest.raises(ValueError, match="unknown price is not free"):
            choose_route(
                [unknown], RoutePolicy(objective="performance", max_cost_usd=1.0), 1000, 500
            )

    def test_over_budget_excluded(self):
        dear = cloud_model(id="dear", input_cost_per_million=100.0, output_cost_per_million=500.0)
        cheap = cloud_model(id="cheap")
        result = choose_route(
            [dear, cheap], RoutePolicy(objective="cost", max_cost_usd=0.05), 1000, 1000
        )
        assert result["model_id"] == "cheap"
        assert result["alternatives"] == []

    def test_unknown_latency_fails_latency_cap(self):
        m = cloud_model(latency_ms=None)
        with pytest.raises(ValueError, match="latency is unknown"):
            choose_route([m], RoutePolicy(objective="cost", max_latency_ms=2000), 1000, 500)

    def test_unknown_quality_fails_min_quality(self):
        m = cloud_model(quality=None)
        with pytest.raises(ValueError, match="quality is unknown"):
            choose_route([m], RoutePolicy(objective="cost", min_quality=0.5), 1000, 500)

    def test_quality_below_threshold_excluded(self):
        low = make_model(id="low", quality=0.3)
        high = make_model(id="high", quality=0.8, base_url="http://127.0.0.1:8080/v1")
        result = choose_route(
            [low, high], RoutePolicy(objective="cost", min_quality=0.5), 1000, 500
        )
        assert result["model_id"] == "high"


class TestPerformanceObjective:
    def test_picks_lowest_known_latency(self):
        slow = cloud_model(id="slow", latency_ms=3000.0)
        fast = cloud_model(id="fast", latency_ms=400.0)
        nolat = cloud_model(id="nolat", latency_ms=None)
        result = choose_route([slow, fast, nolat], RoutePolicy(objective="performance"), 1000, 500)
        assert result["model_id"] == "fast"
        assert result["estimated_latency_ms"] == 400.0
        assert [a["model_id"] for a in result["alternatives"]] == ["slow"]

    def test_all_unknown_latency_raises(self):
        with pytest.raises(ValueError, match="latency estimate is unknown"):
            choose_route(
                [cloud_model(latency_ms=None)], RoutePolicy(objective="performance"), 1000, 500
            )


class TestBalancedObjective:
    def test_requires_all_comparable_metrics(self):
        incomplete = cloud_model(id="incomplete", quality=None)
        complete = cloud_model(id="complete")
        result = choose_route([incomplete, complete], RoutePolicy(objective="balanced"), 1000, 500)
        assert result["model_id"] == "complete"

    def test_prefers_dominating_model(self):
        worse = cloud_model(id="worse", latency_ms=3000.0, quality=0.5)
        better = cloud_model(id="better", latency_ms=1000.0, quality=0.9)
        result = choose_route([worse, better], RoutePolicy(objective="balanced"), 1000, 500)
        assert result["model_id"] == "better"

    def test_all_missing_metrics_raises(self):
        m = cloud_model(input_cost_per_million=None, output_cost_per_million=None)
        with pytest.raises(ValueError, match="missing comparable metric"):
            choose_route([m], RoutePolicy(objective="balanced"), 1000, 500)


class TestPlacement:
    def test_local_placement_never_falls_back_to_cloud(self):
        with pytest.raises(ValueError, match="placement"):
            choose_route(
                [cloud_model()], RoutePolicy(placement="local", objective="cost"), 100, 100
            )

    def test_local_privacy_constraint_selects_local_only(self):
        result = choose_route(
            [cloud_model(), make_model()],
            RoutePolicy(placement="local", objective="cost"),
            1000,
            500,
        )
        assert result["model_id"] == "local-code"
        assert result["alternatives"] == []

    def test_cloud_placement_excludes_local(self):
        result = choose_route(
            [cloud_model(), make_model()],
            RoutePolicy(placement="cloud", objective="cost"),
            1000,
            500,
        )
        assert result["model_id"] == "cloud-sonnet"

    def test_mixed_considers_both(self):
        result = choose_route(
            [cloud_model(), make_model()],
            RoutePolicy(placement="mixed", objective="cost"),
            1000,
            500,
        )
        assert result["model_id"] == "local-code"
        assert [a["model_id"] for a in result["alternatives"]] == ["cloud-sonnet"]


class TestContextAndCapability:
    def test_context_window_excludes_small_model(self):
        small = make_model(id="small", context_window=4096, max_output_tokens=1024)
        with pytest.raises(ValueError, match="context window"):
            choose_route([small], RoutePolicy(objective="cost"), 8000, 500)

    def test_max_output_excludes(self):
        m = make_model(max_output_tokens=256)
        with pytest.raises(ValueError, match="max_output_tokens"):
            choose_route([m], RoutePolicy(objective="cost"), 100, 512)

    def test_capability_filter(self):
        chat_only = make_model(id="chat-only", capabilities=("chat",))
        coder = make_model(id="coder", base_url="http://127.0.0.1:8080/v1")
        result = choose_route(
            [chat_only, coder], RoutePolicy(objective="cost"), 100, 100, capability="code"
        )
        assert result["model_id"] == "coder"
        with pytest.raises(ValueError, match="capability"):
            choose_route([chat_only], RoutePolicy(objective="cost"), 100, 100, capability="code")


class TestPinnedModel:
    def test_pinned_model_wins_over_ranking(self):
        result = choose_route(
            [make_model(), cloud_model()],
            RoutePolicy(objective="cost"),
            1000,
            500,
            selected_model="cloud-sonnet",
        )
        assert result["model_id"] == "cloud-sonnet"
        assert any("user-pinned" in r for r in result["reasons"])
        assert [a["model_id"] for a in result["alternatives"]] == ["local-code"]

    def test_pinned_model_with_unknown_price_is_allowed_without_budget(self):
        m = cloud_model(input_cost_per_million=None, output_cost_per_million=None)
        result = choose_route(
            [m], RoutePolicy(objective="cost"), 1000, 500, selected_model="cloud-sonnet"
        )
        assert result["model_id"] == "cloud-sonnet"
        assert result["estimated_cost_usd"] is None

    def test_pinned_model_must_pass_placement(self):
        with pytest.raises(ValueError, match="selected_model"):
            choose_route(
                [cloud_model(), make_model()],
                RoutePolicy(placement="local"),
                100,
                100,
                selected_model="cloud-sonnet",
            )

    def test_pinned_model_must_pass_budget(self):
        with pytest.raises(ValueError, match="selected_model"):
            choose_route(
                [cloud_model()],
                RoutePolicy(objective="cost", max_cost_usd=0.000001),
                100_000,
                10_000,
                selected_model="cloud-sonnet",
            )

    def test_pinned_model_must_pass_capability(self):
        m = cloud_model(capabilities=("chat",))
        with pytest.raises(ValueError, match="selected_model"):
            choose_route(
                [m], RoutePolicy(), 100, 100, capability="code", selected_model="cloud-sonnet"
            )

    def test_unknown_pinned_id_raises(self):
        with pytest.raises(ValueError, match="not among configured"):
            choose_route([make_model()], RoutePolicy(), 100, 100, selected_model="ghost")


class TestDeterminism:
    def test_tie_broken_by_id(self):
        a = cloud_model(id="alpha")
        b = cloud_model(id="beta")
        for order in ([a, b], [b, a]):
            result = choose_route(order, RoutePolicy(objective="cost"), 1000, 500)
            assert result["model_id"] == "alpha"
            assert [x["model_id"] for x in result["alternatives"]] == ["beta"]

    def test_same_inputs_same_output(self):
        models = [make_model(), cloud_model()]
        policy = RoutePolicy(objective="balanced")
        first = choose_route(models, policy, 2000, 1000)
        second = choose_route(models, policy, 2000, 1000)
        assert first == second


class TestResultShape:
    def test_result_contract(self):
        result = choose_route(
            [make_model(), cloud_model()], RoutePolicy(objective="cost"), 1000, 500
        )
        assert set(result) == {
            "model_id",
            "estimated_cost_usd",
            "estimated_latency_ms",
            "reasons",
            "alternatives",
            "rejections",
        }
        assert isinstance(result["reasons"], list) and result["reasons"]
        assert all(isinstance(r, str) for r in result["reasons"])
        for alt in result["alternatives"]:
            assert set(alt) == {"model_id", "estimated_cost_usd", "estimated_latency_ms"}

    def test_no_credentials_in_result(self):
        result = choose_route([cloud_model()], RoutePolicy(objective="cost"), 1000, 500)
        flat = repr(result)
        assert "ANTHROPIC_API_KEY" not in flat
        assert "api_key" not in flat

    def test_no_feasible_error_lists_every_model(self):
        a = make_model(id="a", capabilities=("chat",))
        b = cloud_model(id="b", latency_ms=None)
        with pytest.raises(ValueError) as exc:
            choose_route(
                [a, b],
                RoutePolicy(objective="cost", max_latency_ms=100),
                100,
                100,
                capability="code",
            )
        message = str(exc.value)
        assert "a:" in message and "b:" in message

    def test_infinity_never_leaks_into_estimates(self):
        result = choose_route(
            [cloud_model(latency_ms=None, quality=None)], RoutePolicy(objective="cost"), 1000, 500
        )
        assert result["estimated_latency_ms"] is None
        assert result["estimated_cost_usd"] is not None
        assert math.isfinite(result["estimated_cost_usd"])
