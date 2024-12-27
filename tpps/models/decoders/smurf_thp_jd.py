import torch as th
import torch.nn as nn

from tpps.pytorch.activations import ParametricSoftplus

from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.base.monte_carlo import MCDecoder
from tpps.models.base.process import Events


from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor


class SmurfTHP_JD(MCDecoder):
    """A mlp decoder based on Monte Carlo estimations. See https://arxiv.org/abs/2002.09291.pdf

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either `None`, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        mc_prop_est: Proportion of numbers of samples for the MC method,
                     compared to the size of the input. (Default=1.).
        emb_dim: Size of the embeddings (default=2).
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the events: either times_only, or temporal.
            Defaults to times_only.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            # Other params
            mc_prop_est: Optional[float] = 1.,
            emb_dim: Optional[int] = 2,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            name:Optional[str] = 'smurf-thp-jd', 
            **kwargs):
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        super(SmurfTHP_JD, self).__init__(
            name=name,
            input_size=units_mlp[0],
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.h1 = nn.Linear(in_features=units_mlp[0], out_features=units_mlp[1])
        self.h2 = nn.Linear(in_features=units_mlp[0], out_features=units_mlp[1])
        self.h3 = nn.Linear(in_features=units_mlp[1], out_features=1)

        self.m1 = nn.Linear(in_features=units_mlp[0], out_features=units_mlp[1])
        self.m2 = nn.Linear(in_features=units_mlp[0], out_features=units_mlp[1])
        self.m3 = nn.Linear(in_features=units_mlp[1], out_features=marks)

        self.activation = ParametricSoftplus(units=1)
        
    
    def log_mark_pmf(
            self, 
            delta_t:th.Tensor, 
            history_representations:th.Tensor):
        
        inner_mark = th.tanh(self.m1(history_representations)) * delta_t + self.m2(history_representations)
        inner_mark = th.tanh(inner_mark)
        p_m = th.softmax(self.m3(inner_mark), dim=-1)        
        p_m = p_m + epsilon(dtype=p_m.dtype, device=p_m.device)
        return th.log(p_m)
    

    def log_ground_intensity(
            self,
            query: th.Tensor,
            prev_times: th.Tensor,
            history_representations: th.Tensor,
            intensity_mask:Optional[th.Tensor] = None):
        
        prev_times = prev_times + epsilon(dtype=prev_times.dtype, device=prev_times.device)
        delta_t = (query - prev_times)/prev_times
        check_tensor(delta_t)
        delta_t = delta_t.unsqueeze(-1)

        
        inner_time = th.tanh(self.h1(history_representations)) * delta_t + self.h2(history_representations)
        inner_time = th.tanh(inner_time)
        outputs = self.activation(self.h3(inner_time))
        
        outputs = outputs + epsilon(dtype=outputs.dtype, device=outputs.device)

        outputs = outputs.squeeze(-1)
    
        return th.log(outputs)

    
    def log_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        pass 
        
    

    def forward(self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None):
        
        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]
        
        prev_times = prev_times + epsilon(dtype=prev_times.dtype, device=prev_times.device)
        delta_t = query - prev_times
        check_tensor(delta_t)
        delta_t = delta_t.unsqueeze(-1)

        log_ground_intensity = self.log_ground_intensity(
                                        query=query,
                                        prev_times=prev_times,
                                        history_representations=history_representations
        )
        
        log_mark_pmf = self.log_mark_pmf(
                                delta_t=delta_t,
                                history_representations=history_representations)

        
        ground_intensity_integral = self.intensity_integral(
                                                query=query, 
                                                prev_times=prev_times,
                                                prev_times_idxs=prev_times_idxs,
                                                intensity_mask=intensity_mask,
                                                representations=representations
                                                )

        
        check_tensor(log_ground_intensity)
        check_tensor(log_mark_pmf)
        check_tensor(ground_intensity_integral * intensity_mask, positive=True)
        
        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        artifacts = {}
        artifacts['last_h'] = last_h.detach().cpu().numpy()
        
        return log_ground_intensity, log_mark_pmf, ground_intensity_integral, intensity_mask, artifacts 

