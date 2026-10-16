"""Model-update evaluation and scoring, without transaction or ledger state.

Device retains signature checks, timers, voting, rewards and logging. These
functions own only reference-model preparation, temporary parameter evaluation
and the signed loss difference used by the existing aggregation metrics.
"""

import copy

from bcmfaigc.training.local import create_local_optimizer


def prepare_reference_model(model, global_parameters, optimizer_name, learning_rate):
    """Clone the model and construct one optimizer, before caller-owned timing."""
    reference_model = copy.deepcopy(model)
    reference_model.load_state_dict(global_parameters, strict=True)
    optimizer = create_local_optimizer(reference_model, optimizer_name, learning_rate)
    return reference_model, optimizer


def evaluate_model_parameters(model, parameters, loss_evaluator):
    """Temporarily load parameters, preserving the original restoration order."""
    original_parameters = copy.deepcopy(model.state_dict())
    was_training = model.training
    try:
        model.load_state_dict(parameters, strict=True)
        return loss_evaluator(model)
    finally:
        model.load_state_dict(original_parameters, strict=True)
        model.train(was_training)


def score_model_update(worker_loss, reference_loss, aggregation_metrics):
    """Return signed loss increase, normalized score and aggregation weight.

    A larger DDPM loss is worse. This calculation does not decide the vote or
    eligibility, which may differ after a malicious-validator vote flip.
    """
    validation_score = worker_loss - reference_loss
    normalized_score, aggregation_weight = aggregation_metrics(validation_score)
    return validation_score, normalized_score, aggregation_weight
