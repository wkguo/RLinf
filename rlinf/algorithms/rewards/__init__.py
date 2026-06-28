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

# The rule-based reward classes (math / code / rstar2 / searchr1 / vqa) belong
# to the LLM reasoning/agent subsystem, which was removed on the franka branch
# along with its heavy verifier dependencies. The registry API is kept so the
# (now dormant) rule-based RewardWorker still imports; the embodied
# EmbodiedRewardWorker does not use it. Re-register classes here to restore a
# rule-based reward.

reward_registry = {}


def register_reward(name: str, reward_class: type):
    assert name not in reward_registry, f"Reward {name} already registered"
    reward_registry[name] = reward_class


def get_rule_based_reward_class(name: str):
    assert name in reward_registry, (
        f"Reward {name} not found. Rule-based rewards were removed on the "
        "franka branch; only embodied reward models are supported."
    )
    return reward_registry[name]
