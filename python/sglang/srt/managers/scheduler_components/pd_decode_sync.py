"""Hierarchical CPU exchange for the PD decode scheduler's full state.

Every rank still joins every scheduling epoch. Only the transport topology
changes: local gather, exchange between node leaders, then local broadcast.
No model or GPU collective uses these process groups.
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist

_HIERARCHIES = {}


class PDDecodeSyncHierarchy:
    def __init__(self, full_group, local_group, leader_group, node_rank, node_count):
        self.full_group = full_group
        self.local_group = local_group
        self.leader_group = leader_group
        self.rank = dist.get_rank(full_group)
        self.world_size = dist.get_world_size(full_group)
        self.node_count = node_count
        self.ranks_per_node = self.world_size // node_count
        self.leader = node_rank * self.ranks_per_node
        self._buffers = None

    def all_gather(self, local_info):
        if local_info.device.type != "cpu" or local_info.dtype != torch.int64:
            raise RuntimeError("PD scheduler hierarchy requires CPU INT64 state")
        width = local_info.numel()
        if self._buffers is None or self._buffers[0].shape[1] != width:
            local = torch.empty((self.ranks_per_node, width), dtype=local_info.dtype)
            full = torch.empty((self.world_size, width), dtype=local_info.dtype)
            local_views = list(local.unbind())
            node_views = list(full.view(self.node_count, -1).unbind())
            self._buffers = local, full, local_views, node_views
        local, full, local_views, node_views = self._buffers
        dist.gather(
            local_info,
            gather_list=local_views if self.rank == self.leader else None,
            dst=self.leader,
            group=self.local_group,
        )
        if self.rank == self.leader:
            dist.all_gather(node_views, local.flatten(), group=self.leader_group)
        dist.broadcast(full, src=self.leader, group=self.local_group)
        # Keep metadata from an earlier epoch independent of reused buffers.
        return full.clone()


def init_pd_decode_sync_hierarchy(full_group, ranks, node_rank, node_count):
    """Validate all ranks' configuration before creating groups in fixed order."""
    enabled = os.getenv("SGLANG_PD_DECODE_SYNC_HIERARCHY", "0").lower() in {
        "1", "true", "yes"
    }
    world_size = dist.get_world_size(full_group)
    rank = dist.get_rank(full_group)
    local = torch.tensor([int(enabled), node_rank, node_count], dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=full_group)
    values = [row.tolist() for row in gathered]
    if len({row[0] for row in values}) != 1:
        raise RuntimeError("PD scheduler hierarchy setting differs across ranks")
    if not enabled:
        return False
    if (list(ranks) != list(range(world_size)) or node_count < 2
            or world_size % node_count or any(row[2] != node_count for row in values)):
        raise RuntimeError("Unsupported PD scheduler hierarchy topology")
    ranks_per_node = world_size // node_count
    if any(row[1] != index // ranks_per_node for index, row in enumerate(values)):
        raise RuntimeError("PD scheduler rank ordering differs from node ordering")
    timeout = timedelta(seconds=60)
    local_groups = []
    for start in range(0, world_size, ranks_per_node):
        local_groups.append(dist.new_group(
            ranks=list(range(start, start + ranks_per_node)),
            backend="gloo", timeout=timeout,
        ))
    leader_group = dist.new_group(
        ranks=list(range(0, world_size, ranks_per_node)),
        backend="gloo", timeout=timeout,
    )
    _HIERARCHIES[full_group] = PDDecodeSyncHierarchy(
        full_group, local_groups[node_rank], leader_group, node_rank, node_count
    )
    return True


def gather_pd_decode_step_info(local_info, full_group):
    hierarchy = _HIERARCHIES.get(full_group)
    return None if hierarchy is None else hierarchy.all_gather(local_info)
