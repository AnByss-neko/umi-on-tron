from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RateLimitedJointPositionAction(JointPositionAction):
    """Joint-position action with a per-policy-step target slew limit."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.max_leg_step <= 0.0:
            raise ValueError("max_leg_step must be greater than zero")

        leg_indices, leg_names = string_utils.resolve_matching_names(
            cfg.leg_joint_names, self._joint_names, preserve_order=False
        )
        self._max_step = torch.full(
            (1, self.action_dim), float("inf"), device=self.device
        )
        self._max_step[:, leg_indices] = cfg.max_leg_step

        self._previous_targets = self._asset.data.default_joint_pos[
            :, self._joint_ids
        ].clone()
        self._applied_actions = torch.zeros_like(self._raw_actions)

        print(
            f"[RateLimitedJointPositionAction] max_leg_step={cfg.max_leg_step:g} rad/step, "
            f"leg_joints={leg_names}"
        )

    @property
    def applied_actions(self) -> torch.Tensor:
        """Action after target slew limiting, expressed relative to its offset."""
        return self._applied_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)

        target_delta = self._processed_actions - self._previous_targets
        target_delta = torch.clamp(
            target_delta, min=-self._max_step, max=self._max_step
        )
        self._processed_actions = self._previous_targets + target_delta
        self._previous_targets.copy_(self._processed_actions)
        self._applied_actions.copy_(
            (self._processed_actions - self._offset) / self._scale
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)

        # Start limiting from the randomized reset pose. This prevents the
        # first policy command from jumping from the nominal default pose.
        current_positions = self._asset.data.joint_pos[env_ids][
            :, self._joint_ids
        ]
        self._previous_targets[env_ids] = current_positions
        self._processed_actions[env_ids] = current_positions
        self._applied_actions[env_ids] = 0.0


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`RateLimitedJointPositionAction`."""

    class_type: type = RateLimitedJointPositionAction

    leg_joint_names: list[str] = MISSING
    """Leg-joint names or regular expressions to which the limit is applied."""

    max_leg_step: float = 0.4
    """Maximum leg target change in radians per policy step."""
