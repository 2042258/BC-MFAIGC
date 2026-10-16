"""Legacy callable signatures and the few real experiment compatibility policies."""

from bcmfaigc.variants import CLASSIC, EXTENDED, KRUM
from .node import FederatedNode
from .network import FederatedNetwork


class ClassicDevice(FederatedNode):
    variant = CLASSIC

    def __init__(
        self,
        idx,
        assigned_train_ds,
        assigned_test_dl,
        local_batch_size,
        learning_rate,
        loss_func,
        opti,
        network_stability,
        net,
        dev,
        miner_acception_wait_time,
        miner_accepted_transactions_size_limit,
        validator_threshold,
        pow_difficulty,
        even_link_speed_strength,
        base_data_transmission_speed,
        even_computation_power,
        is_malicious,
        noise_variance,
        check_signature,
        not_resync_chain,
        malicious_updates_discount,
        knock_out_rounds,
        lazy_worker_knock_out_rounds,
        validation_seed,
        defense_on=1,
    ):
        super().__init__(
            idx=idx,
            assigned_train_ds=assigned_train_ds,
            assigned_test_dl=assigned_test_dl,
            local_batch_size=local_batch_size,
            learning_rate=learning_rate,
            loss_func=loss_func,
            opti=opti,
            network_stability=network_stability,
            net=net,
            dev=dev,
            miner_acception_wait_time=miner_acception_wait_time,
            miner_accepted_transactions_size_limit=miner_accepted_transactions_size_limit,
            validator_threshold=validator_threshold,
            pow_difficulty=pow_difficulty,
            even_link_speed_strength=even_link_speed_strength,
            base_data_transmission_speed=base_data_transmission_speed,
            even_computation_power=even_computation_power,
            is_malicious=is_malicious,
            noise_variance=noise_variance,
            check_signature=check_signature,
            not_resync_chain=not_resync_chain,
            malicious_updates_discount=malicious_updates_discount,
            knock_out_rounds=knock_out_rounds,
            lazy_worker_knock_out_rounds=lazy_worker_knock_out_rounds,
            validation_seed=validation_seed,
            defense_on=defense_on,
        )

    def set_aggregation_config(
        self,
        aggregation_method="hard_filter",
        trust_center=0.0,
        trust_scale=1.0,
        trust_soft_threshold=0.0,
        trust_beta=1.0,
        trust_min_weight=0.05,
        trust_hard_reject_score=5.0,
    ):
        """Configure validation-guided aggregation without changing DDPM training."""
        if aggregation_method not in {"hard_filter", "trust_weighted"}:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        self.aggregation_method = aggregation_method
        self.trust_center = float(trust_center)
        self.trust_scale = float(trust_scale)
        self.trust_soft_threshold = float(trust_soft_threshold)
        self.trust_beta = float(trust_beta)
        self.trust_min_weight = float(trust_min_weight)
        self.trust_hard_reject_score = float(trust_hard_reject_score)


class ExtendedDevice(FederatedNode):
    variant = EXTENDED


class KrumDevice(FederatedNode):
    variant = KRUM


class ClassicNetwork(FederatedNetwork):
    variant = CLASSIC
    node_type = ClassicDevice

    def __init__(
        self,
        data_set_name,
        is_iid,
        batch_size,
        learning_rate,
        loss_func,
        opti,
        num_devices,
        network_stability,
        net,
        dev,
        knock_out_rounds,
        lazy_worker_knock_out_rounds,
        shard_test_data,
        miner_acception_wait_time,
        miner_accepted_transactions_size_limit,
        validator_threshold,
        pow_difficulty,
        even_link_speed_strength,
        base_data_transmission_speed,
        even_computation_power,
        malicious_updates_discount,
        num_malicious,
        noise_variance,
        check_signature,
        not_resync_chain,
        validation_seed,
        test_batch_size=256,
        defense_on=1,
    ):
        super().__init__(
            data_set_name=data_set_name,
            is_iid=is_iid,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_func=loss_func,
            opti=opti,
            num_devices=num_devices,
            network_stability=network_stability,
            net=net,
            dev=dev,
            knock_out_rounds=knock_out_rounds,
            lazy_worker_knock_out_rounds=lazy_worker_knock_out_rounds,
            shard_test_data=shard_test_data,
            miner_acception_wait_time=miner_acception_wait_time,
            miner_accepted_transactions_size_limit=miner_accepted_transactions_size_limit,
            validator_threshold=validator_threshold,
            pow_difficulty=pow_difficulty,
            even_link_speed_strength=even_link_speed_strength,
            base_data_transmission_speed=base_data_transmission_speed,
            even_computation_power=even_computation_power,
            malicious_updates_discount=malicious_updates_discount,
            num_malicious=num_malicious,
            noise_variance=noise_variance,
            check_signature=check_signature,
            not_resync_chain=not_resync_chain,
            validation_seed=validation_seed,
            test_batch_size=test_batch_size,
            defense_on=defense_on,
        )


class ExtendedNetwork(FederatedNetwork):
    node_type = ExtendedDevice


class KrumNetwork(FederatedNetwork):
    variant = KRUM
    node_type = KrumDevice
