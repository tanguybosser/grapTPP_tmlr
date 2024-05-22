import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.rmtpp_jd import RMTPP_JD
from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, subtract_exp, check_tensor
from tpps.utils.nnplus import non_neg_param

class RMTPP_DD(RMTPP_JD):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            mark_activation: Optional[str] = 'relu',
            hist_time_grouping: Optional[str] = 'summation',
            **kwargs):
        super(RMTPP_DD, self).__init__(
            name="rmtpp-dd",
            units_mlp=units_mlp,
            multi_labels=multi_labels,
            marks=marks,
            encoding=encoding,
            mark_activation=mark_activation,
            hist_time_grouping=hist_time_grouping,
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
            sampling: Optional[bool] = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the intensities for each query time given event
        representations.

        Args:
            events: [B,L] Times and labels of events.
            query: [B,T] Times to evaluate the intensity function.
            prev_times: [B,T] Times of events directly preceding queries.
            prev_times_idxs: [B,T] Indexes of times of events directly
                preceding queries. These indexes are of window-prepended
                events.
            pos_delta_mask: [B,T] A mask indicating if the time difference
                `query - prev_times` is strictly positive.
            is_event: [B,T] A mask indicating whether the time given by
                `prev_times_idxs` corresponds to an event or not (a 1 indicates
                an event and a 0 indicates a window boundary).
            representations: [B,L+1,D] Representations of window start and
                each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        

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

        b,l = query.shape
        representations_time = representations[0:b,:,:]
        representations_mark = representations[b:,:,:]
        history_representations_time = take_3_by_2(                            
            representations_time, index=prev_times_idxs)                   # [B,T,D]
        history_representations_mark = take_3_by_2(                          
            representations_mark, index=prev_times_idxs)

        self.mu.data = non_neg_param(self.mu.data)
        check_tensor(self.mu.data, positive=True)

        v_h_t = self.w_t(history_representations_time)                         #[B,T,1]
        v_h_t = v_h_t.squeeze(-1)                                         #[B,T]
        delta_t = query - prev_times
        w_delta_t = self.w * delta_t                     # [B,T]
    
        exp_term = th.clamp(w_delta_t + v_h_t, max=80) #Avoids infinity. 
        exp_term = th.exp(exp_term) 
        ground_intensity = self.mu + exp_term
        check_tensor(ground_intensity, positive=True)
        ground_intensity = ground_intensity + epsilon(dtype=ground_intensity.dtype, device=ground_intensity.device)
        log_ground_intensity = th.log(ground_intensity)
        check_tensor(log_ground_intensity)

        log_mark_pmf = self.log_mark_pmf(
                                query_representations=query_representations,
                                history_representations=history_representations_mark
                            )
        check_tensor(log_mark_pmf)

        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask
        
        exp_1, exp_2 = v_h_t + w_delta_t, v_h_t                         # [B,T]
        exp_1, exp_2 = exp_1 * intensity_mask, exp_2 * intensity_mask   # [B,T]
        
        poisson_integral = self.mu * delta_t                            # [B,T]
                
        ground_intensity_integrals = th.exp(exp_1) - th.exp(exp_2)
        ground_intensity_integrals = ground_intensity_integrals / self.w                # [B,T]
        
        ground_intensity_integrals = ground_intensity_integrals + poisson_integral 
        
        check_tensor(ground_intensity_integrals * intensity_mask,
                     positive=True)
        
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


