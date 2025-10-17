# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Partition utility functions.
"""

from typing import Any, List


def partition_by_size(data: List[Any], size: int) -> List[List[Any]]:
    """
    Partition a list by size.
    When indivisible, the last group contains fewer items than the target size.

    Examples:
        - data: [1,2,3,4,5]
        - size: 2
        - return: [[1,2], [3,4], [5]]
    """
    assert size > 0
    return [data[i : (i + size)] for i in range(0, len(data), size)]


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    """
    Partition a list by groups.
    When indivisible, some groups may have more items than others.

    Examples:
        - data: [1,2,3,4,5]
        - groups: 2
        - return: [[1,3,5], [2,4]]
    """
    assert groups > 0
    return [data[i::groups] for i in range(groups)]


def shift_list(data: List[Any], n: int) -> List[Any]:
    """
    Rotate a list by n elements.

    Examples:
        - data: [1,2,3,4,5]
        - n: 3
        - return: [4,5,1,2,3]
    """
    return data[(n % len(data)) :] + data[: (n % len(data))]
