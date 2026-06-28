# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class SupportedEnvType(Enum):
    # The franka branch keeps only the real-robot stack plus the FrankaSim
    # smoke-test env. Other sim/world-model envs were removed; re-add their
    # enum member and a ``get_env_cls`` branch to bring one back.
    REALWORLD = "realworld"
    FRANKASIM = "frankasim"


def get_env_cls(env_type: str, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment ("realworld" or "frankasim").
        env_cfg: Optional environment configuration.

    Returns:
        Environment class corresponding to the environment type.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.REALWORLD:
        from rlinf.envs.realworld import RealWorldEnv

        return RealWorldEnv
    elif env_type == SupportedEnvType.FRANKASIM:
        from rlinf.envs.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
