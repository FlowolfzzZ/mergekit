# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import override

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.sparsify import magnitude_outliers
from mergekit.merge_methods.generalized_task_arithmetic import get_task_vectors


class PCBMergeTask(Task[torch.Tensor]):
    output_weight: WeightInfo
    tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    density: float
    weight: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tv_info, base = get_task_vectors(
            self.output_weight,
            self.base_model,
            tensors,
            tensor_parameters=ImmutableMap({model: {} for model in tensors}),
        )
        if not tv_info:
            return base

        n = len(tv_info)
        tvs = torch.stack([tv["delta"] for tv in tv_info], dim=0)
        tvs_flat = tvs.view(n, -1)
        eps=1e-12

        # $b_i = b_{intra, i} \odot b_{inter, i}$
        # $b_{intra, i} = Softmax(N \cdot Norm(\delta_i \odot \delta_i))$
        # norm_tvs_sqr = F.normalize(tvs_flat * tvs_flat, p=1, dim=1, eps=eps)
        def minmax_normalize(x, dim=0, eps=eps):
            min_vals, _ = torch.min(x, dim=dim, keepdim=True)
            max_vals, _ = torch.max(x, dim=dim, keepdim=True)
            y = (x - min_vals) / (max_vals - min_vals + eps)
            return y
        norm_tvs_sqr = minmax_normalize(tvs_flat * tvs_flat, dim=1)

        # 向量维度太大，softmax的分母很大，最终结果是0
        # b_intra = F.softmax(n * norm_tvs_sqr, dim=1)
        b_intra = torch.exp(n * norm_tvs_sqr)

        # $b_{inter, i} = \sum_{j = 1}^{n} Softmax(Norm(\delta_i \odot \delta_j))$
        # b_inter = torch.zeros_like(tvs_flat)
        # for i in range(n):
        #     inter_prod = tvs_flat[i] * tvs_flat
        #     # inter_norm = F.normalize(inter_prod, dim=1, eps=eps)
        #     inter_norm = minmax_normalize(inter_prod, dim=1)
        #     b_inter[i] = F.softmax(inter_norm, dim=1).sum(dim=0)
        inter_prod = tvs_flat.unsqueeze(1) * tvs_flat.unsqueeze(0)
        inter_norm = minmax_normalize(inter_prod, dim=-1).sum(dim=0)
        b_inter = torch.tanh(inter_norm)

        b = b_intra * b_inter
        b_hat = magnitude_outliers(
            b, density=self.density, gamma=(1 - self.density) / 2
        )

        sum_b_hat = torch.sum(b_hat, dim=0, keepdim=True)
        sum_b_hat = torch.where(sum_b_hat == 0, torch.ones_like(sum_b_hat), sum_b_hat)
        weights = b_hat / sum_b_hat
        final_delta = torch.sum(tvs_flat * weights, dim=0).view(tvs.shape[1:])
        return base + self.weight * final_delta


class PCBMerge(MergeMethod):
    def name(self) -> str:
        return "pcb_merging"

    @override
    def pretty_name(self) -> Optional[str]:
        return "PCB Merging"
    
    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/2410.02396"
    
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="density", required=True),
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        **kwargs,
    ) -> Task[torch.Tensor]:
        return PCBMergeTask(
            output_weight=output_weight,
            tensors=tensors,
            base_model=base_model,
            density=parameters["density"],
            weight=parameters["weight"],
        )