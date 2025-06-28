from typing import List, Tuple, Optional, Any, Union, Literal

import torch
from torch import Tensor

from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.functional.classification.auroc import _reduce_auroc
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
)


class AUPRC(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]
    confmat: Tensor

    def __init__(
        self,
        num_classes: int = 2,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_precision_recall_curve_arg_validation(
                num_classes, thresholds, ignore_index
            )

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.average = average

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.add_state(
                "confmat",
                default=torch.zeros(
                    len(thresholds), num_classes, 2, 2, dtype=torch.long
                ),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multiclass_precision_recall_curve_tensor_validation(
                preds, target, self.num_classes, self.ignore_index
            )
        preds, target, _ = _multiclass_precision_recall_curve_format(
            preds, target, self.num_classes, self.thresholds, self.ignore_index
        )
        state = _multiclass_precision_recall_curve_update(
            preds, target, self.num_classes, self.thresholds
        )
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(
        self,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]
    ]:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        curve = _multiclass_precision_recall_curve_compute(
            state, self.num_classes, self.thresholds
        )

        return _reduce_auroc(curve[0], curve[1], average=self.average)
