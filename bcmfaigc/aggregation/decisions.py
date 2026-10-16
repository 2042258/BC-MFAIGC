"""Distinct direct/block vote and epoch reductions, without node/database/clock state."""

import copy
from .trust_weighting import median_trust_weight as combine_validator_weights


def collect_direct_decisions(validated_transactions, aggregation_method, weight_of):
    per_epoch_votes = {}
    for transaction in validated_transactions:
        if not transaction or "worker_device_idx" not in transaction:
            continue
        worker_idx = transaction["worker_device_idx"]
        epoch_seq = transaction["local_total_accumulated_epochs_this_round"]
        key = (worker_idx, epoch_seq)
        votes = per_epoch_votes.setdefault(
            key,
            {
                "positive": 0,
                "negative": 0,
                "positive_weights": [],
                "negative_weights": [],
                "aggregation_transactions": [],
                "transaction": transaction,
            },
        )
        if transaction.get("update_direction") is True:
            votes["positive"] += 1
            votes["positive_weights"].append(weight_of(transaction))
        else:
            votes["negative"] += 1
            votes["negative_weights"].append(weight_of(transaction))
        if transaction.get(
            "aggregation_eligible", transaction.get("update_direction") is True
        ):
            votes["aggregation_transactions"].append(transaction)
    positive_epochs = {}
    negative_epochs = {}
    finally_used_params = {}
    finally_used_weights = {}
    for (worker_idx, epoch_seq), votes in per_epoch_votes.items():
        if aggregation_method == "trust_weighted" and votes["aggregation_transactions"]:
            aggregation_transaction = votes["aggregation_transactions"][0]
            positive_epochs.setdefault(worker_idx, set()).add(epoch_seq)
            if epoch_seq == max(positive_epochs[worker_idx]):
                finally_used_params[worker_idx] = aggregation_transaction[
                    "local_updates_params"
                ]
                finally_used_weights[worker_idx] = combine_validator_weights(
                    (weight_of(item) for item in votes["aggregation_transactions"])
                )
        elif votes["positive"] >= votes["negative"]:
            positive_epochs.setdefault(worker_idx, set()).add(epoch_seq)
            if epoch_seq == max(positive_epochs[worker_idx]):
                finally_used_params[worker_idx] = votes["transaction"][
                    "local_updates_params"
                ]
                finally_used_weights[worker_idx] = combine_validator_weights(
                    votes["positive_weights"]
                )
        else:
            negative_epochs.setdefault(worker_idx, set()).add(epoch_seq)
    for worker_idx in {key[0] for key in per_epoch_votes}:
        positive_count = sum(
            (
                votes["positive"]
                for (candidate, _), votes in per_epoch_votes.items()
                if candidate == worker_idx
            )
        )
        negative_count = sum(
            (
                votes["negative"]
                for (candidate, _), votes in per_epoch_votes.items()
                if candidate == worker_idx
            )
        )
        if negative_count > positive_count and aggregation_method != "trust_weighted":
            finally_used_params.pop(worker_idx, None)
            finally_used_weights.pop(worker_idx, None)
    return (per_epoch_votes, finally_used_params, finally_used_weights)


def accumulate_block_decision(
    records,
    transaction,
    worker_id,
    aggregation_method,
    discount,
    observer_id,
    reward_total,
    weight_of,
):
    if not worker_id in records.keys():
        records[worker_id] = {}
        records[worker_id]["positive_epochs"] = set()
        records[worker_id]["negative_epochs"] = set()
        records[worker_id]["positive_votes"] = 0
        records[worker_id]["negative_votes"] = 0
        records[worker_id]["all_valid_epochs"] = set()
        records[worker_id]["finally_used_params"] = None
        records[worker_id]["finally_used_weight"] = 1.0
    local_epoch_seq = transaction["local_total_accumulated_epochs_this_round"]
    positive_direction_validators = transaction["positive_direction_validators"]
    negative_direction_validators = transaction["negative_direction_validators"]
    aggregation_validators = transaction.get(
        "aggregation_validators", positive_direction_validators
    )
    if aggregation_method == "trust_weighted" and aggregation_validators:
        records[worker_id]["positive_votes"] += len(positive_direction_validators)
        records[worker_id]["negative_votes"] += len(negative_direction_validators)
        records[worker_id]["all_valid_epochs"].add(local_epoch_seq)
        if local_epoch_seq == max(records[worker_id]["all_valid_epochs"]):
            records[worker_id]["finally_used_params"] = transaction[
                "local_updates_params"
            ]
            records[worker_id]["finally_used_weight"] = combine_validator_weights(
                (
                    validator_record.get("aggregation_weight", 1.0)
                    for validator_record in aggregation_validators
                )
            )
        if observer_id == worker_id:
            reward_total += transaction["local_updates_rewards"]
    elif len(positive_direction_validators) >= len(negative_direction_validators):
        records[worker_id]["positive_votes"] += len(positive_direction_validators)
        records[worker_id]["negative_votes"] += len(negative_direction_validators)
        records[worker_id]["positive_epochs"].add(local_epoch_seq)
        records[worker_id]["all_valid_epochs"].add(local_epoch_seq)
        if local_epoch_seq == max(records[worker_id]["all_valid_epochs"]):
            records[worker_id]["finally_used_params"] = transaction[
                "local_updates_params"
            ]
            records[worker_id]["finally_used_weight"] = combine_validator_weights(
                (
                    validator_record.get("aggregation_weight", 1.0)
                    for validator_record in positive_direction_validators
                )
            )
        if observer_id == worker_id:
            reward_total += transaction["local_updates_rewards"]
    else:
        records[worker_id]["positive_votes"] += len(positive_direction_validators)
        records[worker_id]["negative_votes"] += len(negative_direction_validators)
        if discount:
            records[worker_id]["negative_epochs"].add(local_epoch_seq)
            records[worker_id]["all_valid_epochs"].add(local_epoch_seq)
            if local_epoch_seq == max(records[worker_id]["all_valid_epochs"]):
                discounted_valid_validator_sig_worker_transaciton_local_updates_params = copy.deepcopy(
                    transaction["local_updates_params"]
                )
                for var in discounted_valid_validator_sig_worker_transaciton_local_updates_params:
                    discounted_valid_validator_sig_worker_transaciton_local_updates_params[
                        var
                    ] *= discount
                records[worker_id]["finally_used_params"] = (
                    discounted_valid_validator_sig_worker_transaciton_local_updates_params
                )
                records[worker_id]["finally_used_weight"] = (
                    weight_of(transaction) * discount
                )
            if observer_id == worker_id:
                reward_total += transaction["local_updates_rewards"] * discount
        else:
            records[worker_id]["negative_epochs"].add(local_epoch_seq)
    return (reward_total, positive_direction_validators, negative_direction_validators)
