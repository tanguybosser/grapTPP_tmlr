import math
import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder
from tpps.pytorch.models import LAYER_CLASSES

from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor
from tpps.utils.encoding import encoding_size

class JointLogNormalMixtureDecoder(VariableHistoryDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://arxiv.org/pdf/2210.15294.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
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
            **kwargs):
        if encoding not in ["times_only", "learnable"]:
            raise ValueError("Invalid event encoding for LogNormMix decoder.")
        if encoding == 'learnable' and embedding_constraint is None:
            print("Warning, embedding are unconstrained for LongNormMix decoder, setting to 'nonneg'")
            embedding_constraint = 'nonneg'
        super(JointLogNormalMixtureDecoder, self).__init__(
            name="joint-log-normal-mixture",
            input_size=units_mlp[0],
            marks=marks,
            encoding=encoding, 
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,  
            **kwargs)
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        self.marks = marks
        self.n_mixture = n_mixture
        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        self.mu = nn.Linear(in_features=units_mlp[0], out_features=marks * n_mixture)
        self.s = nn.Linear(in_features=units_mlp[0], out_features=marks * n_mixture)
        self.w = nn.Linear(in_features=units_mlp[0], out_features=marks * n_mixture)
        if self.encoding == "learnable":
            w_t_layer = LAYER_CLASSES[self.embedding_constraint]
            self.w_t = w_t_layer(in_features=enc_size, out_features=1)
        self.marks1 = nn.Linear(
            in_features=units_mlp[0], out_features=units_mlp[1])
        self.marks2 = nn.Linear(
            in_features=units_mlp[1], out_features=marks)
        self.multi_labels = multi_labels
        self.mark_activation = self.get_mark_activation(mark_activation)

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

        
        b,l,k,c = query.shape[0], query.shape[1], self.marks, self.n_mixture
        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)  # [B,T,D] actually history 
        
        delta_t = query - prev_times  # [B,T]
        delta_t = delta_t.unsqueeze(-1).unsqueeze(-1)  # [B,T,1, 1]
        delta_t = th.relu(delta_t) #Just to ensure that the deltas are positive ? 
        delta_t = delta_t + (delta_t == 0).float() * epsilon(
        dtype=delta_t.dtype, device=delta_t.device)
        delta_t = th.log(delta_t)
    
        mu = self.mu(history_representations)  # [B,T,K*C]
        mu = mu.view(b,l,k,c) #[B,T,K,C]
        std = th.exp(self.s(history_representations))
        std = std.view(b,l,k,c)        
        
        w = self.w(history_representations)
        w = w.view(b,l,k,c)
        w = th.softmax(w, dim=-1)
        
        p_m = th.softmax(
            self.marks2(
                self.mark_activation(self.marks1(history_representations))), dim=-1)
        p_m = p_m + epsilon(dtype=p_m.dtype, device=p_m.device)
        
        cum_f_k = w * 0.5 * (
                1 + th.erf((delta_t - mu) / (std * math.sqrt(2))))
        cum_f_k = th.clamp(th.sum(cum_f_k, dim=-1), max=1-1e-6) #[B,T,K]
        
        grad_outputs = th.zeros_like(cum_f_k, requires_grad=True)
        grad_inputs = th.autograd.grad(
            outputs=cum_f_k,
            inputs=query,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True)[0]
        f_k = th.autograd.grad(
            outputs=grad_inputs,
            inputs=grad_outputs,
            grad_outputs=th.ones_like(grad_inputs),
            retain_graph=True,
            create_graph=True)[0]

        #conditional distribution
        f_k = f_k + epsilon(dtype=f_k.dtype, device=f_k.device) #[B,T,K]
        
        #Joint distribution
        f_joint = f_k * p_m #[B,T,K] 
        
        #Marginal of times 
        f = th.sum(f_joint, dim=-1)

        #Joint CDF
        cum_f_joint = cum_f_k * p_m 
        
        #Marginal CDF
        cum_f = th.sum(cum_f_joint, dim=-1) #[B,T]
        
        one_min_cum_f = 1. - cum_f
        one_min_cum_f = th.relu(one_min_cum_f) + epsilon(
            dtype=cum_f.dtype, device=cum_f.device)
        
        base_log_intensity = th.log(f / one_min_cum_f)
        base_intensity_itg = - th.log(one_min_cum_f)
        
        one_min_cum_f = one_min_cum_f.unsqueeze(-1) #[B,T,1]
        
        marked_log_intensity = th.log(f_joint / one_min_cum_f) #[B,T,K]
        #marked_log_intensity = marked_log_intensity + th.log(p_m)  # [B,T,K]

        marked_intensity_itg = base_intensity_itg.unsqueeze(dim=-1)  # [B,T,1]
        
        #Trick to have Lambda(t) = \sum_k(Lambda_k(t))
        #This does not return the true Lambda_k(t), but we don't need it later, only Lambda(t)
        ones = th.ones_like(p_m)
        marked_intensity_itg = (marked_intensity_itg / self.marks) * ones  

        #intensity_mask = pos_delta_mask  # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        artifacts_decoder = {
            "base_log_intensity": base_log_intensity,
            "base_intensity_integral": base_intensity_itg,
            "mark_probability": p_m}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            artifacts['decoder'] = artifacts_decoder

        check_tensor(marked_log_intensity * intensity_mask.unsqueeze(-1))
        check_tensor(marked_intensity_itg * intensity_mask.unsqueeze(-1),
                     positive=True)
        
        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        
        artifacts['last_h'] = last_h.detach().cpu().numpy()
        artifacts["mu"] = mu.detach()[:,-1,:].squeeze().cpu().numpy() #Take history representation of window, i.e. up to last observed event. 
        artifacts["sigma"] = std.detach()[:,-1,:].squeeze().cpu().numpy()
        artifacts["w"] = w.detach()[:,-1,:].squeeze().cpu().numpy()

        return (marked_log_intensity,
                marked_intensity_itg,
                intensity_mask,
                artifacts)                      # [B,T,M], [B,T,M], [B,T], Dict

    def get_mark_activation(self, mark_activation):
        if mark_activation == 'relu':
            mark_activation = th.relu
        elif mark_activation == 'tanh':
            mark_activation = th.tanh
        elif mark_activation == 'sigmoid':
            mark_activation = th.sigmoid
        return mark_activation