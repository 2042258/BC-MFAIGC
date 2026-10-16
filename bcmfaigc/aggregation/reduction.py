"""Ordered weighted tensor sums, preserving first nonfloating buffers."""

import copy
import torch


def sum_weighted_parameters(ordered_updates):
    sum_parameters = None
    total_weight = sum((weight for _, weight in ordered_updates))
    for local_updates_params, aggregation_weight in ordered_updates:
        if sum_parameters is None:
            sum_parameters = copy.deepcopy(local_updates_params)
            for var in sum_parameters:
                if torch.is_floating_point(sum_parameters[var]):
                    sum_parameters[var] *= aggregation_weight
        else:
            for var in sum_parameters:
                if torch.is_floating_point(sum_parameters[var]):
                    sum_parameters[var] += (
                        local_updates_params[var] * aggregation_weight
                    )
    return (sum_parameters, total_weight)
