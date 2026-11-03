"""Portable entry to the unchanged reconstruction training recipe."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Subset
from src.models.halo import HALO
from src.models.tvt_control_chain import SEMANTICS
from src.models.tvt_online import make_window_independent
from scripts.tvt_guarded_training import CleanForecastTargetTrainer

def _build_model(cfg, channel_names, n_features):
    mcfg=cfg["model"]
    window=int(cfg["data"]["window"])
    return HALO(
        channel_names=channel_names,
        window=window,
        d_model=int(mcfg.get("d_model", 128)),
        n_heads=int(mcfg.get("n_heads", 4)),
        n_encoder_blocks=int(mcfg.get("n_encoder_blocks", 2)),
        n_decoder_blocks=int(mcfg.get("n_decoder_blocks", 2)),
        dropout=float(mcfg.get("dropout", 0.1)),
        layer_dropout=float(mcfg.get("layer_dropout", 0.0)),
        actuator_aug_rate=float(mcfg.get("actuator_aug_rate", 0.0)),
        actuator_aug_max_channels=int(mcfg.get("actuator_aug_max_channels", 2)),
        layer_score_weights=mcfg.get("layer_score_weights", None),
        use_multi_period=bool(mcfg.get("use_multi_period", False)),
        multi_period_topk=int(mcfg.get("multi_period_topk", 3)),
        multi_period_inner=int(mcfg.get("multi_period_inner", 32)),
        multi_period_kernels=mcfg.get("multi_period_kernels", None),
        multi_period_gate=bool(mcfg.get("multi_period_gate", False)),
        multi_period_gate_init=float(mcfg.get("multi_period_gate_init", 0.0)),
        pred_horizon=int(mcfg.get("pred_horizon", 0)),
        pred_score_weight=float(mcfg.get("pred_score_weight", 0.0)),
        use_tn_stem=bool(mcfg.get("use_tn_stem", False)),
        tn_stem_kernels=mcfg.get("tn_stem_kernels", None),
        tn_stem_inner=int(mcfg.get("tn_stem_inner", 16)),
        tn_stem_gate=bool(mcfg.get("tn_stem_gate", False)),
        tn_stem_gate_init=float(mcfg.get("tn_stem_gate_init", 0.1)),
        use_tn_branch=bool(mcfg.get("use_tn_branch", False)),
        tn_branch_layers=int(mcfg.get("tn_branch_layers", 3)),
        tn_branch_topk=int(mcfg.get("tn_branch_topk", 3)),
        tn_branch_hidden=int(mcfg.get("tn_branch_hidden", 32)),
        tn_branch_gate=bool(mcfg.get("tn_branch_gate", True)),
        tn_branch_gate_init=float(mcfg.get("tn_branch_gate_init", 0.3)),
        tn_branch_gate_data_dependent=bool(mcfg.get("tn_branch_gate_data_dependent", False)),
        tn_branch_gate_mlp_hidden=int(mcfg.get("tn_branch_gate_mlp_hidden", 32)),
        tn_branch_gate_bias_init=float(mcfg.get("tn_branch_gate_bias_init", -2.2)),
        tn_branch_gate_hard_input=bool(mcfg.get("tn_branch_gate_hard_input", False)),
        tn_branch_gate_hard_min_active=float(mcfg.get("tn_branch_gate_hard_min_active", 0.05)),
        tn_branch_gate_warmup_start=int(mcfg.get("tn_branch_gate_warmup_start", 0)),
        tn_branch_gate_warmup_full=int(mcfg.get("tn_branch_gate_warmup_full", 0)),
        use_tn_only_encoder=bool(mcfg.get("use_tn_only_encoder", False)),
        tn_encoder_layers=int(mcfg.get("tn_encoder_layers", 3)),
        tn_encoder_topk=int(mcfg.get("tn_encoder_topk", 3)),
        tn_encoder_hidden=int(mcfg.get("tn_encoder_hidden", 64)),
        use_lstm_forecast=bool(mcfg.get("use_lstm_forecast", False)),
        lstm_forecast_horizon=int(mcfg.get("lstm_forecast_horizon", 20)),
        lstm_forecast_hidden=int(mcfg.get("lstm_forecast_hidden", 64)),
        lstm_forecast_layers=int(mcfg.get("lstm_forecast_layers", 2)),
        lstm_score_weight=float(mcfg.get("lstm_score_weight", 0.0)),
        lstm_loss_weight=float(mcfg.get("lstm_loss_weight", 1.0)),
        use_channel_token_branch=bool(mcfg.get("use_channel_token_branch", False)),
        channel_token_layers=int(mcfg.get("channel_token_layers", 3)),
        channel_token_heads=int(mcfg.get("channel_token_heads", 4)),
        channel_token_alpha_init=float(mcfg.get("channel_token_alpha_init", 0.3)),
    )


class Channels(Dataset):
    def __init__(self, dataset, channels):self.dataset=dataset;self.channels=list(channels)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,i):return self.dataset[i][:,self.channels]

class IndependentValidationTrainer(CleanForecastTargetTrainer):
    @torch.no_grad()
    def validate_nominal(self, epoch=-1):
        original=self.model
        self.model=make_window_independent(copy.deepcopy(original).eval())
        try:return super().validate_nominal(epoch)
        finally:self.model=original

def build_base(cfg, platform):
    s=SEMANTICS[platform]
    bb=json.loads(Path(cfg['backbone_config']).read_text())
    if platform=='alfa':names=[bb['channel_aliases'][i] for i in s.base]
    else:names=[f'motor/throttle_pwm_{i}' for i in range(4)]+['imu/gyro_x','imu/gyro_y','imu/gyro_z','pressure','state/roll','state/pitch','state/yaw','state/alt']
    model=_build_model(bb,names,len(names))
    # Old attention cannot reshape a zero-token command layer. Remove only
    # unobservable encoders; do not fabricate command channels or modify source.
    for layer,channels in enumerate(model.by_layer):
        if not channels:model.encoders[layer]=nn.Identity()
    return model,bb

def loader(ds, batch, seed=None):
    return DataLoader(ds,batch_size=batch,shuffle=seed is not None,
                      generator=None if seed is None else torch.Generator().manual_seed(seed),num_workers=0)

def reference_subset(ds, size, seed):
    ids=np.random.default_rng(seed).permutation(len(ds))[:size]
    return Subset(ds,ids.tolist()),ids
