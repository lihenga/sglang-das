"""Negotiate optional Store APIs without depending on a wheel version string.

This describes the local package. A standalone client daemon must come from
the same Mooncake package as its Python client; it is not probed by this module.
"""

import inspect
import logging
import re

logger = logging.getLogger(__name__)

_FEATURE_METHODS = {
    "dfs_health": ("probe_dfs_read", "probe_dfs_write", "set_dfs_access_policy"),
    "dfs_prefetch": (
        "batch_get_session_start_with_sources",
        "batch_get_session_prefetch",
        "batch_get_session_refresh",
        "dfs_prefetch_arena_available",
        "disable_dfs_prefetch",
    ),
    "read_plan": ("create_read_plan",),
    "read_layout": ("batch_get_into_layouts",),
    "service_metrics": ("set_service_endpoint",),
}
_PREFETCH_METHODS = (
    "batch_get_session_prefetch",
    "dfs_prefetch_arena_available",
    "disable_dfs_prefetch",
)
_DIAGNOSTIC_SETUP_PARAMETERS = {"enable_client_http_server", "client_http_port"}


def _declared_capabilities(store):
    query = getattr(store, "get_capabilities", None)
    if not callable(query):
        return {}
    result = query()
    return result if isinstance(result, dict) else {}


def supports_feature(store, feature: str) -> bool:
    """Use explicit declarations when available, then check callable APIs."""
    declared = _declared_capabilities(store).get("features", {})
    if isinstance(declared, dict) and declared.get(feature) is False:
        return False
    return all(
        callable(getattr(store, name, None)) for name in _FEATURE_METHODS[feature]
    )


def _has_prefetch_api(store) -> bool:
    return any(callable(getattr(store, name, None)) for name in _PREFETCH_METHODS)


def disable_dfs_prefetch(store) -> None:
    """Release an L4 arena; community/L3 packages have no such arena/API."""
    declared = _declared_capabilities(store)
    if (
        declared.get("client_mode") == "standalone"
        and declared.get("features", {}).get("dfs_prefetch") is False
    ):
        # The native DummyClient inherits the method but owns no local arena.
        return
    disable = getattr(store, "disable_dfs_prefetch", None)
    if callable(disable):
        if not disable():
            raise RuntimeError(
                "Mooncake cannot disable DFS prefetch after requests have started."
            )
    elif _has_prefetch_api(store):
        raise RuntimeError(
            "Mooncake exposes DFS prefetch but cannot release its arena; "
            "install a package with disable_dfs_prefetch support."
        )


def _documented_parameters(text):
    # pybind11 prints all supported overloads followed by the *invoked* args.
    # Never infer support from the latter: they contain every rejected keyword.
    declarations = text.split("Invoked with:", 1)[0]
    # __doc__ overloads say "1. setup(self: ...)"; pybind error messages
    # usually say "1. (self: ...)". Ignore the untyped "setup(): ..." header.
    signatures = re.findall(
        r"(?:\bsetup|^\s*\d+\.\s*)\(([^)\n]*)\)", declarations, re.MULTILINE
    )
    names = set()
    for signature in signatures:
        names.update(re.findall(r"\b(\w+)\s*:", signature))
    return names or None


def _setup_parameters(store):
    declared = _declared_capabilities(store).get("setup_parameters")
    if isinstance(declared, (list, tuple)) and all(
        isinstance(name, str) for name in declared
    ):
        return set(declared)
    try:
        parameters = inspect.signature(store.setup).parameters
    except (TypeError, ValueError):
        return _documented_parameters(getattr(store.setup, "__doc__", "") or "")
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return set(parameters)


def _unsupported_parameters(error, kwargs):
    message = str(error)
    explicit = re.search(
        r"(?:unexpected keyword argument|invalid keyword argument)\s+['\"](\w+)['\"]",
        message,
    ) or re.search(r"['\"]?(\w+)['\"]? is an invalid keyword argument", message)
    if explicit:
        # An inner helper's TypeError can mention the same keyword after setup
        # has already changed state. Only retry setup binding errors.
        if re.search(r"\w+\(", message) and not re.search(r"\bsetup\(", message):
            return set()
        return {explicit.group(1)} & kwargs.keys()
    if "incompatible function arguments" in message:
        supported = _documented_parameters(message)
        if supported is not None:
            return kwargs.keys() - supported
    return set()


def setup_store(store, args, kwargs):
    """Call setup with only supported optional arguments.

    Diagnostic endpoints may be omitted. Storage placement and tenant options
    must never be silently dropped. A missing prefetch gate is unnecessary on
    community/L3, or is replaced by an immediate startup-only arena release.
    Unrelated TypeErrors propagate without retrying a partially executed setup.
    """
    kwargs = dict(kwargs)
    disable_after_setup = False
    if (
        kwargs.get("enable_client_http_server")
        and kwargs.get("client_http_port") == 0
        and _declared_capabilities(store)
        .get("features", {})
        .get("client_http_auto_port")
        is not True
    ):
        # L3 rejects port zero after allocating client state; do not retry setup.
        kwargs["enable_client_http_server"] = False
        kwargs.pop("client_http_port")
        logger.warning(
            "Mooncake package does not declare automatic client HTTP ports; "
            "disabling its optional client HTTP metrics endpoint. "
            "Rebuild/install a package with automatic client HTTP port support."
        )

    def omit(unsupported):
        nonlocal disable_after_setup
        for name in sorted(unsupported):
            if name == "enable_dfs_prefetch" and kwargs[name] is False:
                if _has_prefetch_api(store):
                    if not callable(getattr(store, "disable_dfs_prefetch", None)):
                        raise RuntimeError(
                            "The installed Mooncake package cannot disable DFS "
                            "prefetch per client. Install a package with "
                            "disable_dfs_prefetch support."
                        )
                    disable_after_setup = True
            elif name not in _DIAGNOSTIC_SETUP_PARAMETERS:
                raise RuntimeError(
                    f"The installed Mooncake package does not support {name} "
                    "in MooncakeDistributedStore.setup(); this storage option "
                    "cannot be safely omitted."
                )
            logger.warning(
                "Mooncake setup omits unsupported optional parameter %s", name
            )
            kwargs.pop(name)

    supported = _setup_parameters(store)
    if (
        kwargs.get("enable_dfs_prefetch") is False
        and not _has_prefetch_api(store)
        and (supported is None or "enable_dfs_prefetch" not in supported)
    ):
        # The specified community and L3 baselines predate DFS prefetch arenas.
        kwargs.pop("enable_dfs_prefetch")
    if supported is not None:
        omit(kwargs.keys() - supported)
    while True:
        try:
            result = store.setup(*args, **kwargs)
            break
        except TypeError as error:
            unsupported = _unsupported_parameters(error, kwargs)
            if not unsupported:
                raise
            omit(unsupported)
    if not result and disable_after_setup:
        disable_dfs_prefetch(store)
    return result


def resolve_read_plan_options(store, enabled: bool, reuse: bool) -> tuple[bool, bool]:
    if enabled and not supports_feature(store, "read_plan"):
        logger.warning(
            "Mooncake package lacks ReadPlan support; using compatible session reads "
            "and disabling ReadPlan address reuse."
        )
        return False, False
    return enabled, reuse
