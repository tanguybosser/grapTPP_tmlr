import math
import torch as th

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.log_normal_mixture_jd import LogNormalMixture_JD

from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor

class LogNormalMixture_DD(LogNormalMixture_JD):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://arxiv.org/pdf/1909.12127.pdf.
    """
    def __init__(
            self,
            n_mixture: int,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            embedding_constraint: Optional[str] = None,
            emb_dim: Optional[int] = 2,
            mark_activation: Optional[str] = 'relu',
            cond_ind : Optional[bool] = False,
            **kwargs):
        super(LogNormalMixture_DD, self).__init__(
            name="log-normal-mixture-dd",
            n_mixture=n_mixture,
            units_mlp=units_mlp,
            multi_labels=multi_labels,
            marks=marks,
            encoding=encoding,
            embedding_constraint=embedding_constraint,
            emb_dim=emb_dim,
            mark_activation=mark_activation,
            cond_ind=cond_ind,
            **kwargs)
        

    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.LongTensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None, 
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        
        query.requires_grad = True
        
        batch_size = query.shape[0]

        (query_representations,
         intensity_mask) = self.get_query_representations(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations, 
            representations_mask=representations_mask)  # [B,T,enc_size], [B,T]

    
        batch_size = query.shape[0]
        representations_time = representations[0:batch_size,:,:]
        representations_mark = representations[batch_size:,:,:]
        
        history_representations_time = take_3_by_2(
            representations_time, index=prev_times_idxs)  # [B,T,D] 
        history_representations_mark = take_3_by_2(
            representations_mark, index=prev_times_idxs)
        
        delta_t = query - prev_times  # [B,T]
        delta_t = delta_t.unsqueeze(-1)  # [B,T,1]
        delta_t = th.relu(delta_t) 
        delta_t = delta_t + (delta_t == 0).float() * epsilon(
        dtype=delta_t.dtype, device=delta_t.device)
        delta_t = th.log(delta_t)
        
        mu = self.mu(history_representations_time)  # [B,T,K]
        std = th.exp(self.s(history_representations_time))
        w = th.softmax(self.w(history_representations_time), dim=-1)
        
        log_mark_pmf = self.log_mark_pmf(
                                query_representations=query_representations,
                                history_representations=history_representations_mark
                            )
        check_tensor(log_mark_pmf)

        cum_f = w * 0.5 * (
                1 + th.erf((delta_t - mu) / (std * math.sqrt(2))))
        #Ground cumulative density. 
        cum_f = th.clamp(th.sum(cum_f, dim=-1), max=1-1e-6)

        one_min_cum_f = 1. - cum_f
        one_min_cum_f = th.relu(one_min_cum_f) + epsilon(
            dtype=cum_f.dtype, device=cum_f.device)
        
        f = th.autograd.grad(
            outputs=cum_f,
            inputs=query,
            grad_outputs=th.ones_like(cum_f, requires_grad=True),
            retain_graph=True,
            create_graph=True)[0]
        query.requires_grad = False
    
        f = f + epsilon(dtype=f.dtype, device=f.device)

        log_ground_intensity = th.log(f / one_min_cum_f)
        check_tensor(log_ground_intensity)

        ground_intensity_integrals = - th.log(one_min_cum_f) +  epsilon(eps=1e-7,
            dtype=cum_f.dtype, device=cum_f.device)
        check_tensor(ground_intensity_integrals * intensity_mask, positive=True)
    
    
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h_t = history_representations_time[th.arange(batch_size), last_event_idx,:]
        last_h_m = history_representations_mark[th.arange(batch_size), last_event_idx,:] #[B,D]
        artifacts = {}
        artifacts['last_h_t'] = last_h_t.detach().cpu().numpy()
        artifacts['last_h_m'] = last_h_m.detach().cpu().numpy()

        return (log_ground_intensity,
                log_mark_pmf,
                ground_intensity_integrals,
                intensity_mask,
                artifacts)                      # [B,T,M], [B,T,M], [B,T], Dict
    

