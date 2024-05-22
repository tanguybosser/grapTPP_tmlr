import torch as th
import torch.nn as nn

from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.base.cumulative import CumulativeDecoder
from tpps.models.base.process import Events

from tpps.pytorch.models import MLP
from tpps.pytorch.layers.log import Log

from tpps.utils.encoding import encoding_size
from tpps.utils.index import take_3_by_2
from tpps.utils.stability import epsilon, check_tensor
from tpps.utils.nnplus import non_neg_param


class MLPCmDecoder(CumulativeDecoder):
    """A mlp decoder based on the cumulative approach.

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either none, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        mc_prop_est: Proportion of numbers of samples for the MC method,
                     compared to the size of the input. (Default=1.).
        do_zero_subtraction: If `True` the class computes
            Lambda(tau) = Lambda'(tau) - Lambda'(0)
            in order to enforce Lambda(0) = 0. Defaults to `True`.
        emb_dim: Size of the embeddings (default=2).
        encoding: Way to encode the events: either times_only, or temporal.
            Defaults to times_only.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = "nonneg",
            activation_final_mlp: Optional[str] = "parametric_softplus",
            # Other params
            model_log_cm: Optional[bool] = False,
            do_zero_subtraction: Optional[bool] = True,
            emb_dim: Optional[int] = 2,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):

        if constraint_mlp is None:
            print("Warning! MLP decoder is unconstrained. Setting to `nonneg`")
            constraint_mlp = "nonneg"

        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        input_size = units_mlp[0] - enc_size
        super(MLPCmDecoder, self).__init__(
            name="mlp-cm",
            do_zero_subtraction=do_zero_subtraction,
            model_log_cm=model_log_cm,
            input_size=input_size,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mlp = MLP(
            units=units_mlp[1:],
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            # units_mlp in this class also provides the input dimensionality
            # of the mlp
            input_shape=units_mlp[0],
            activation_final=activation_final_mlp)
        self.mu = nn.Parameter(th.Tensor(self.marks))
        self.reset_parameters()

    def cum_intensity(
            self,
            query_representations:th.Tensor, 
            history_representations:th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the cumulative log intensity and a mask

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
            representations: [B,L+1,D] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.
            update_running_stats: whether running stats are updated or not.

        Returns:
            intensity_integral: [B,T,M] The cumulative intensities for each
                query time for each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: Some measures.
        """
        
        hidden = th.cat(
            [query_representations, history_representations],
            dim=-1)                                        # [B,T,units_mlp[0]]
        intensity_itg = self.mlp(hidden.float())                    # [B,T,output_size]

        return intensity_itg


    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[bool] = None, 
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
            representations: [B,L+1,D] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: Some measures

        """
        # Add grads for query to compute derivative
        query.requires_grad = True
        
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

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D] 
        
        self.mu.data = non_neg_param(self.mu.data)
        check_tensor(self.mu.data, positive=True)
        
        intensity_integrals_q = self.cum_intensity(
                                    query_representations=query_representations,
                                    history_representations=history_representations)

        delta_t = query - prev_times                                  # [B,T]
        delta_t = delta_t.unsqueeze(dim=-1)                           # [B,T,1]
        poisson_term = self.mu * delta_t                # [B,T,M]
        

        intensity_integrals_q = intensity_integrals_q + poisson_term

        # Remove masked values and add epsilon for stability
        intensity_integrals_q = \
            intensity_integrals_q * intensity_mask.unsqueeze(-1)

        # Optional zero substraction
        if self.do_zero_subtraction: #Computes Lambda(0)
            
            query_representations_z, intensity_mask_z = self.get_query_representations(
                                        events=events,
                                        query=prev_times,
                                        prev_times=prev_times,
                                        prev_times_idxs=prev_times_idxs,
                                        pos_delta_mask=pos_delta_mask,
                                        is_event=is_event,
                                        representations=representations,
                                        representations_mask=representations_mask)  # [B,T,enc_size], [B,T]

            history_representations_z = take_3_by_2(representations, index=prev_times_idxs)                   # [B,T,D] 
                        
            intensity_integrals_z = self.cum_intensity(
                                            query_representations=query_representations_z,
                                            history_representations=history_representations_z)

            intensity_integrals_z = intensity_integrals_z * intensity_mask_z.unsqueeze(-1)
             
            intensity_integrals_q = th.clamp(
                intensity_integrals_q - intensity_integrals_z, min=0.
            ) + intensity_integrals_z 
            intensity_integrals_q = intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=intensity_integrals_q.dtype,
                device=intensity_integrals_q.device) * query.unsqueeze(-1)
            
            intensity_integrals = intensity_integrals_q - intensity_integrals_z
            intensity_mask = intensity_mask * intensity_mask_z

        else:
            intensity_integrals_q = intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=intensity_integrals_q.dtype,
                device=intensity_integrals_q.device) * query.unsqueeze(-1)
            
            intensity_integrals = intensity_integrals_q

        check_tensor(intensity_integrals * intensity_mask.unsqueeze(-1),
                     positive=True)
        
        ground_intensity_integrals = th.sum(intensity_integrals, dim=-1)
        
        #intensity_integrals = th.clamp(intensity_integrals, min=1e-5, max=1e8) #min 1e-6, max 1e9
        
        # Compute derivative of the integral - Double backward trick ! 
        grad_outputs = th.zeros_like(intensity_integrals, requires_grad=True)
        grad_inputs = th.autograd.grad(
            outputs=intensity_integrals,
            inputs=query,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True)[0]
        marked_intensity = th.autograd.grad(
            outputs=grad_inputs,
            inputs=grad_outputs,
            grad_outputs=th.ones_like(grad_inputs, requires_grad=True),
            retain_graph=True,
            create_graph=True)[0]
        
        query.requires_grad = False
        #marked_intensity.data.clamp_(min=1e-5, max=1e8)

        check_tensor(marked_intensity, positive=True, strict=True)

        ground_intensity = th.sum(marked_intensity, dim=-1)
        log_ground_intensity = th.log(ground_intensity)

        mark_pmf = marked_intensity / ground_intensity.unsqueeze(-1)
        log_mark_pmf = th.log(mark_pmf)
        
        '''
        artifacts_decoder = {
            "intensity_integrals": intensity_integrals,
            "marked_intensity": marked_intensity,
            "marked_log_intensity": marked_log_intensity,
            "intensity_mask": intensity_mask}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            if 'decoder' in artifacts:
                if 'attention_weights' in artifacts['decoder']:
                    artifacts_decoder['attention_weights'] = \
                        artifacts['decoder']['attention_weights']
            artifacts['decoder'] = artifacts_decoder
        '''

        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        artifacts = {}
        artifacts['last_h'] = last_h.detach().cpu().numpy()
            
        return (log_ground_intensity,
                log_mark_pmf,
                ground_intensity_integrals, 
                intensity_mask,
                artifacts)  # [B,T,M], [B,T,M], [B,T], Dict

## ADD POISSON TERM DIRECTLY HERE 