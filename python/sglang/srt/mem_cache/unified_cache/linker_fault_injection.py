from __future__ import annotations

import random
from typing import Callable

from sglang.srt.environ import envs


class InjectedLinkerLoadFailure(RuntimeError):
    pass


def _rank_is_selected(tp_rank: int) -> bool:
    selected = envs.SGLANG_TEST_LINKER_LOAD_FAILURE_RANKS.get()
    if selected is None or selected.strip().lower() in ("", "all"):
        return True
    return tp_rank in {
        int(part.strip()) for part in selected.split(",") if part.strip()
    }


def _never_fails(pool: str, where: str) -> None:
    return None


def arm_load_failure_injection(
    tp_rank: int,
    *,
    rng: random.Random | None = None,
) -> Callable[[str, str], None]:
    probability = envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.get()
    if probability <= 0 or not _rank_is_selected(tp_rank):
        return _never_fails

    chooser = rng if rng is not None else random
    if chooser.random() >= probability:
        return _never_fails

    def check(pool: str, where: str) -> None:
        raise InjectedLinkerLoadFailure(
            f"Injected external-linker load failure for pool={pool}, {where}, "
            f"tp_rank={tp_rank}"
        )

    return check
