"""Deterministic agent routing engine for LLM-Optimise.

Selects a provider model for a task from user-supplied estimates only
(prices, latency, quality are labels the user configured or calibrated;
nothing here is an observed measurement and nothing is invented).

Design rules enforced here:

* Pure stdlib, no network calls, no environment reads. ``api_key_env`` is
  only the *name* of an environment variable; credential values are never
  read, stored, or emitted in any dict this module produces.
* Unknown metrics never default. A model with no price is excluded from
  the ``cost`` objective and from any ``max_cost_usd`` budget check rather
  than being treated as free; the same applies to latency and quality.
* A local model with an explicitly zero price is a legitimate estimate --
  it is labelled as excluding electricity/hardware amortisation.
* Routing is deterministic: equal scores are broken by lexicographic
  model id.
* There is no silent fallback. If no model is feasible a ``ValueError``
  is raised listing why each candidate was rejected -- a ``local``
  placement never quietly routes to the cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from urllib.parse import urlsplit

__all__ = ["ProviderModel", "RoutePolicy", "choose_route"]

_PROVIDERS = frozenset({"openai", "anthropic"})
_LOCATIONS = frozenset({"local", "cloud"})
_PLACEMENTS = frozenset({"local", "cloud", "mixed"})
_OBJECTIVES = frozenset({"cost", "performance", "balanced"})

# Hostnames that are unambiguously loopback/local. A "cloud" model pointing
# at one of these is almost certainly mislabelled and is rejected.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"})

# Balanced-objective weights over normalised cost, normalised latency and
# quality deficit. Fixed constants keep the ranking deterministic and
# explainable; they are not tunable per call by design (keep the policy
# surface small).
_BALANCED_WEIGHTS = (0.4, 0.4, 0.2)


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_positive_int(value: object, name: str) -> int:
    # bool is a subclass of int and must not sneak through as 1/0.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _check_optional_number(
    value: object, name: str, *, minimum: float, allow_equal: bool = True
) -> float | None:
    """Validate an optional finite number; bool, NaN and infinities are rejected."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or None, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < minimum or (not allow_equal and value == minimum):
        bound = f">= {minimum}" if allow_equal else f"> {minimum}"
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return value


def _validate_base_url(url: str, location: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"base_url must use http or https, got {url!r}")
    if parts.query or parts.fragment:
        raise ValueError("base_url must not have a query or fragment")
    if not parts.hostname:
        raise ValueError(f"base_url has no host: {url!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("base_url must not embed credentials (user:pass@host)")
    host = parts.hostname.lower()
    is_loopback = (
        host in _LOOPBACK_HOSTS
        or host.startswith("127.")
        or host.endswith(".local")
        or host.endswith(".localhost")
    )
    if location == "cloud":
        if parts.scheme != "https":
            raise ValueError(f"cloud base_url must use https (plaintext rejected): {url!r}")
        if is_loopback:
            raise ValueError(
                f"base_url {url!r} points at a local host but location is 'cloud'; "
                "this looks mislabelled -- use location='local'"
            )


@dataclass(frozen=True)
class ProviderModel:
    """A routable model endpoint described by user-supplied estimates.

    Costs are USD per million tokens. ``latency_ms`` is an estimated
    end-to-end request latency and ``quality`` a subjective 0..1 rating.
    All three are optional: absent means unknown, never zero.
    """

    id: str
    model: str
    provider: str
    location: str
    base_url: str
    context_window: int
    max_output_tokens: int
    api_key_env: str | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    latency_ms: float | None = None
    quality: float | None = None
    ram_gib: float | None = None
    gpu_gib: float | None = None
    capabilities: tuple[str, ...] = ("chat", "code")
    supports_json_schema: bool = False

    def __post_init__(self) -> None:
        if type(self.supports_json_schema) is not bool:
            raise ValueError("supports_json_schema must be a boolean")
        _require_str(self.id, "id")
        _require_str(self.model, "model")
        if self.provider not in _PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(_PROVIDERS)}, got {self.provider!r}")
        if self.location not in _LOCATIONS:
            raise ValueError(f"location must be one of {sorted(_LOCATIONS)}, got {self.location!r}")
        _require_str(self.base_url, "base_url")
        _validate_base_url(self.base_url, self.location)
        _require_positive_int(self.context_window, "context_window")
        _require_positive_int(self.max_output_tokens, "max_output_tokens")
        if self.max_output_tokens > self.context_window:
            raise ValueError("max_output_tokens cannot exceed context_window")
        if self.api_key_env is not None:
            _require_str(self.api_key_env, "api_key_env")
        object.__setattr__(
            self,
            "input_cost_per_million",
            _check_optional_number(
                self.input_cost_per_million, "input_cost_per_million", minimum=0.0
            ),
        )
        object.__setattr__(
            self,
            "output_cost_per_million",
            _check_optional_number(
                self.output_cost_per_million, "output_cost_per_million", minimum=0.0
            ),
        )
        object.__setattr__(
            self,
            "latency_ms",
            _check_optional_number(self.latency_ms, "latency_ms", minimum=0.0, allow_equal=False),
        )
        quality = _check_optional_number(self.quality, "quality", minimum=0.0)
        if quality is not None and quality > 1.0:
            raise ValueError(f"quality must be within 0..1, got {quality!r}")
        object.__setattr__(self, "quality", quality)
        for key in ("ram_gib", "gpu_gib"):
            object.__setattr__(
                self, key, _check_optional_number(getattr(self, key), key, minimum=0.0)
            )
        if not isinstance(self.capabilities, tuple) or not self.capabilities:
            raise ValueError("capabilities must be a non-empty tuple of strings")
        for cap in self.capabilities:
            _require_str(cap, "capability")

    def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        """USD estimate for a request, or None when either price is unknown.

        A partial price (only input or only output known) is treated as
        unknown so a missing component can never undercount the total.
        """
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000.0

    def to_public_dict(self) -> dict:
        """A persistable description with no credential material.

        ``api_key_env`` is deliberately omitted; even the variable *name*
        stays out of persisted routing artefacts.
        """
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "api_key_env"}
        data["capabilities"] = list(self.capabilities)
        return data


@dataclass(frozen=True)
class RoutePolicy:
    """Routing preferences: where to run and what to optimise for.

    Budgets/thresholds are hard constraints. A model whose corresponding
    metric is unknown cannot prove it satisfies a set constraint and is
    excluded rather than assumed compliant.
    """

    placement: str = "mixed"
    objective: str = "balanced"
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    min_quality: float | None = None
    max_ram_gib: float | None = None
    max_gpu_gib: float | None = None

    def __post_init__(self) -> None:
        if self.placement not in _PLACEMENTS:
            raise ValueError(
                f"placement must be one of {sorted(_PLACEMENTS)}, got {self.placement!r}"
            )
        if self.objective not in _OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(_OBJECTIVES)}, got {self.objective!r}"
            )
        object.__setattr__(
            self,
            "max_cost_usd",
            _check_optional_number(
                self.max_cost_usd, "max_cost_usd", minimum=0.0, allow_equal=False
            ),
        )
        object.__setattr__(
            self,
            "max_latency_ms",
            _check_optional_number(
                self.max_latency_ms, "max_latency_ms", minimum=0.0, allow_equal=False
            ),
        )
        for key in ("max_ram_gib", "max_gpu_gib"):
            object.__setattr__(
                self, key, _check_optional_number(getattr(self, key), key, minimum=0.0)
            )
        min_quality = _check_optional_number(self.min_quality, "min_quality", minimum=0.0)
        if min_quality is not None and min_quality > 1.0:
            raise ValueError(f"min_quality must be within 0..1, got {min_quality!r}")
        object.__setattr__(self, "min_quality", min_quality)


@dataclass
class _Candidate:
    model: ProviderModel
    cost: float | None
    reasons: list[str] = field(default_factory=list)


def _hard_constraint_failures(
    m: ProviderModel,
    policy: RoutePolicy,
    input_tokens: int,
    output_tokens: int,
    capability: str,
    cost: float | None,
) -> list[str]:
    """Constraints every model must pass, including a user-pinned one."""
    failures: list[str] = []
    if capability not in m.capabilities:
        failures.append(f"lacks capability {capability!r} (has {', '.join(m.capabilities)})")
    if policy.placement != "mixed" and m.location != policy.placement:
        failures.append(f"placement policy is {policy.placement!r} but model is {m.location!r}")
    if input_tokens + output_tokens > m.context_window:
        failures.append(
            f"context window {m.context_window} cannot fit "
            f"{input_tokens} input + {output_tokens} output tokens"
        )
    if output_tokens > m.max_output_tokens:
        failures.append(
            f"max_output_tokens {m.max_output_tokens} is below requested {output_tokens}"
        )
    if policy.max_cost_usd is not None:
        if cost is None:
            failures.append(
                "cost is unknown and cannot be verified against the "
                f"max_cost_usd {policy.max_cost_usd} budget (unknown price is not free)"
            )
        elif cost > policy.max_cost_usd:
            failures.append(f"estimated cost {cost:.6f} USD exceeds budget {policy.max_cost_usd}")
    if policy.max_latency_ms is not None:
        if m.latency_ms is None:
            failures.append(
                f"latency is unknown and cannot be verified against max_latency_ms "
                f"{policy.max_latency_ms}"
            )
        elif m.latency_ms > policy.max_latency_ms:
            failures.append(
                f"estimated latency {m.latency_ms} ms exceeds cap {policy.max_latency_ms} ms"
            )
    if policy.min_quality is not None:
        if m.quality is None:
            failures.append(
                f"quality is unknown and cannot be verified against min_quality "
                f"{policy.min_quality}"
            )
        elif m.quality < policy.min_quality:
            failures.append(f"quality {m.quality} is below min_quality {policy.min_quality}")
    if m.location == "local":
        for metric, limit in (("ram_gib", "max_ram_gib"), ("gpu_gib", "max_gpu_gib")):
            value, bound = getattr(m, metric), getattr(policy, limit)
            if bound is not None and (value is None or value > bound):
                failures.append(f"{metric} unknown or exceeds local {limit} budget")
    return failures


def _objective_metric_failures(c: _Candidate, objective: str) -> list[str]:
    """Metrics the ranking objective needs; unknowns exclude, never default."""
    failures: list[str] = []
    m = c.model
    if objective == "cost" and c.cost is None:
        failures.append("excluded from cost objective: price is unknown (not treated as free)")
    if objective == "performance" and m.latency_ms is None:
        failures.append("excluded from performance objective: latency estimate is unknown")
    if objective == "balanced":
        missing = []
        if c.cost is None:
            missing.append("cost")
        if m.latency_ms is None:
            missing.append("latency")
        if m.quality is None:
            missing.append("quality")
        if missing:
            failures.append(
                "excluded from balanced objective: missing comparable metric(s): "
                + ", ".join(missing)
            )
    return failures


def _rank(candidates: list[_Candidate], objective: str) -> list[_Candidate]:
    """Deterministic ordering, best first; ties broken by model id."""
    if objective == "cost":
        return sorted(candidates, key=lambda c: (c.cost, c.model.id))
    if objective == "performance":
        return sorted(candidates, key=lambda c: (c.model.latency_ms, c.model.id))
    # balanced: all metrics are known here (enforced by the objective filter).
    max_cost = max(c.cost for c in candidates)
    max_latency = max(c.model.latency_ms for c in candidates)
    w_cost, w_latency, w_quality = _BALANCED_WEIGHTS

    def score(c: _Candidate) -> float:
        norm_cost = c.cost / max_cost if max_cost > 0 else 0.0
        norm_latency = c.model.latency_ms / max_latency if max_latency > 0 else 0.0
        return w_cost * norm_cost + w_latency * norm_latency + w_quality * (1.0 - c.model.quality)

    return sorted(candidates, key=lambda c: (score(c), c.model.id))


def _entry(c: _Candidate) -> dict:
    return {
        "model_id": c.model.id,
        "estimated_cost_usd": c.cost,
        "estimated_latency_ms": c.model.latency_ms,
    }


def choose_route(
    models: list[ProviderModel],
    policy: RoutePolicy,
    input_tokens: int,
    output_tokens: int,
    capability: str = "code",
    selected_model: str | None = None,
) -> dict:
    """Pick the best feasible model for a request.

    Returns ``{model_id, estimated_cost_usd, estimated_latency_ms,
    reasons, alternatives}``. ``estimated_*`` are None when the underlying
    user-supplied estimate is unknown. ``alternatives`` lists the other
    feasible models in objective rank order.

    ``selected_model`` pins a model by id: it wins regardless of ranking
    but must still satisfy capability, placement, context/output limits
    and every explicit budget -- pinning never bypasses constraints.

    Raises ``ValueError`` (never falls back silently) when inputs are
    invalid or no model is feasible, with per-model rejection reasons.
    """
    if not isinstance(policy, RoutePolicy):
        raise ValueError(f"policy must be a RoutePolicy, got {type(policy).__name__}")
    _require_positive_int(input_tokens, "input_tokens")
    _require_positive_int(output_tokens, "output_tokens")
    _require_str(capability, "capability")
    if selected_model is not None:
        _require_str(selected_model, "selected_model")
    if not models:
        raise ValueError("no models configured: cannot route")
    for m in models:
        if not isinstance(m, ProviderModel):
            raise ValueError(f"models must all be ProviderModel instances, got {type(m).__name__}")
    ids = [m.id for m in models]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate model ids: {', '.join(dupes)}")

    candidates = [
        _Candidate(model=m, cost=m.estimated_cost_usd(input_tokens, output_tokens)) for m in models
    ]
    rejections: dict[str, list[str]] = {}
    feasible: list[_Candidate] = []
    for c in candidates:
        failures = _hard_constraint_failures(
            c.model, policy, input_tokens, output_tokens, capability, c.cost
        )
        if failures:
            rejections[c.model.id] = failures
        else:
            feasible.append(c)

    pinned: _Candidate | None = None
    if selected_model is not None:
        if selected_model not in ids:
            raise ValueError(
                f"selected_model {selected_model!r} is not among configured models: "
                f"{', '.join(sorted(ids))}"
            )
        if selected_model in rejections:
            raise ValueError(
                f"selected_model {selected_model!r} does not satisfy the policy: "
                + "; ".join(rejections[selected_model])
            )
        pinned = next(c for c in feasible if c.model.id == selected_model)

    # Objective-metric availability only gates ranked (non-pinned) candidates.
    rankable: list[_Candidate] = []
    for c in feasible:
        if pinned is not None and c is pinned:
            continue
        failures = _objective_metric_failures(c, policy.objective)
        if failures:
            rejections.setdefault(c.model.id, []).extend(failures)
        else:
            rankable.append(c)

    ranked = _rank(rankable, policy.objective) if rankable else []

    if pinned is not None:
        chosen = pinned
        chosen.reasons.append(
            f"user-pinned selected_model {chosen.model.id!r}; passed placement, capability, "
            "context and budget checks"
        )
        alternatives = ranked
    elif ranked:
        chosen = ranked[0]
        chosen.reasons.append(
            f"best of {len(ranked)} feasible model(s) for objective {policy.objective!r} "
            f"under placement {policy.placement!r}"
        )
        alternatives = ranked[1:]
    else:
        detail = "; ".join(
            f"{mid}: {' | '.join(reasons)}" for mid, reasons in sorted(rejections.items())
        )
        raise ValueError(
            f"no feasible model for placement={policy.placement!r} "
            f"objective={policy.objective!r} capability={capability!r} -- {detail}"
        )

    m = chosen.model
    if chosen.cost is not None:
        chosen.reasons.append(
            f"estimated cost {chosen.cost:.6f} USD for {input_tokens} input + "
            f"{output_tokens} output tokens (user-supplied per-token prices)"
        )
        if chosen.cost == 0.0 and m.location == "local":
            chosen.reasons.append(
                "zero-price local estimate excludes electricity and hardware amortisation"
            )
    else:
        chosen.reasons.append("cost estimate unavailable: model pricing is unknown")
    if m.latency_ms is not None:
        chosen.reasons.append(f"estimated latency {m.latency_ms:g} ms (user-supplied estimate)")
    else:
        chosen.reasons.append("latency estimate unavailable")
    chosen.reasons.append(f"runs {m.location}: {m.provider} model {m.model}")

    return {
        "model_id": m.id,
        "estimated_cost_usd": chosen.cost,
        "estimated_latency_ms": m.latency_ms,
        "reasons": list(chosen.reasons),
        "alternatives": [_entry(c) for c in alternatives],
        "rejections": rejections,
    }
