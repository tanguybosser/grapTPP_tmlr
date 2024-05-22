import pdb

from numpy import rate

import torch as th
import torch.distributions as td
import math
import matplotlib
import matplotlib.pyplot as plt

from typing import Dict, Optional, Tuple

from tpps.models.decoders.base.decoder import Decoder
from tpps.models.encoders.base.encoder import Encoder
from tpps.models.base.process import Process
from tpps.utils.events import Events
from tpps.utils.history_bst import get_prev_times
from tpps.utils.index import take_2_by_1
from tpps.utils.logical import xor
from tpps.utils.stability import epsilon, check_tensor

class EncDecProcess(Process):
    """A parametric encoder decoder process.

    Args
        encoder: The encoder.
        decoder: The decoder.

    """
    def __init__(self,
                 encoder: Encoder,
                 encoder_time:Encoder, 
                 encoder_mark:Encoder,
                 decoder: Decoder,
                 multi_labels: Optional[bool] = False,
                 decoder_type: Optional[str] = 'joint',
                 **kwargs):
        # TODO: Fix this hack that allows modular to work.
        if encoder is not None:
            assert encoder.marks == decoder.marks
            name = '_'.join([encoder.name, decoder.name])
            marks = encoder.marks
            self.enc_dec_hidden_size = encoder.output_size
            if decoder.input_size is not None:
                assert encoder.output_size == decoder.input_size
        elif encoder_time is not None:
            assert encoder_time.marks == decoder.marks
            assert encoder_mark.marks == decoder.marks
            name = '_'.join([encoder_time.name, encoder_mark.name, decoder.name])
            marks = encoder_time.marks
            if decoder.input_size is not None:
                assert encoder_time.output_size == decoder.input_size
                assert encoder_mark.output_size == decoder.input_size
            self.enc_dec_hidden_size = encoder_time.output_size
        else:
            name = kwargs.pop("name")
            marks = kwargs.pop("marks")
        super(EncDecProcess, self).__init__(name=name, marks=marks, **kwargs)
        self.encoder = encoder
        self.encoder_time = encoder_time
        self.encoder_mark = encoder_mark
        self.decoder = decoder
        self.multi_labels = multi_labels
        self.decoder_type = decoder_type

    def intensity(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the intensities at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            intensity: [B,T,M] The intensities for each query time for each
                mark (class).
            intensity_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        log_intensity, _, intensity_mask, _ = self.artifacts(
            query=query, events=events)
        return th.exp(log_intensity), intensity_mask

    def log_density(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the log joint densities at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            log_density: [B,T,M] The densities for each query time for each
                mark (class).
            density_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        # TODO: Intensity integral should be summed over marks.
        log_ground_intensity, log_mark_density, ground_intensity_integral ,intensity_mask, _ = self.artifacts(
            query=query, events=events)
        
        log_ground_intensity = log_ground_intensity * intensity_mask
        log_mark_density = log_mark_density * intensity_mask.unsqueeze(-1)
        ground_intensity_integral = ground_intensity_integral * intensity_mask

        ground_intensity = th.exp(log_ground_intensity)
        ground_density = ground_intensity * th.exp(-ground_intensity_integral)

        mark_density = th.exp(log_mark_density)        
        joint_density = ground_density.unsqueeze(-1) * mark_density
        joint_density = joint_density + epsilon(dtype=joint_density.dtype)
        log_joint_density = th.log(joint_density)
        
        check_tensor(log_joint_density)

        return log_joint_density, log_mark_density, intensity_mask

    def neg_log_likelihood(
            self, events: Events, test=False, 
            loss='nll', training_type='full') -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the negative log likelihood of events.

        Args:
            events: [B,L] Times and labels of events.
        Returns:
            nll: [B] The negative log likelihoods for each sequence.
            nll_mask: [B] Which neg_log_likelihoods are valid for further
                computation based on e.g. at least one element in sequence has
                a contribution.
            artifacts: Other useful items, e.g. the relevant window of the
                sequence.

        """
        artifacts = {}
        events_times = events.get_times(postpend_window=True)         # [B,L+1] 
        
        log_ground_intensity, log_mark_density, ground_intensity_integral, intensity_mask, artifacts_modular = self.artifacts(
                query=events_times, events=events)  # [B,L+1], [B,L+1,M], [B,L+1]
        
        window_intensity_mask = intensity_mask[:, -1]          # [B]
        intensity_mask = intensity_mask[:, :-1]                # [B,L]

        log_ground_intensity = log_ground_intensity[:,:-1] #[B,L]
        log_ground_intensity = log_ground_intensity * intensity_mask
        
        labels = events.labels                                        # [B,L,M]
        log_mark_density = log_mark_density[:,:-1,:] #[B,L,M]
        true_log_mark_density = log_mark_density * labels 
        true_log_mark_density = th.sum(true_log_mark_density, dim=-1)   
        true_log_mark_density = true_log_mark_density * intensity_mask

        window_integral = ground_intensity_integral[:,-1]  #[B,1]
        window_integral = window_integral * window_intensity_mask
        
        ground_intensity_integral = ground_intensity_integral[:,:-1]        #[B,L]
        ground_intensity_integral = ground_intensity_integral * intensity_mask

        add_window_integral = 1 - events.final_event_on_window.type(
            intensity_mask.dtype)                                     # [B]
        window_integral = window_integral * add_window_integral       # [B]

        loss_t = -th.sum(log_ground_intensity, dim=-1) + th.sum(ground_intensity_integral, dim=-1)
        loss_m = -th.sum(true_log_mark_density, dim=-1)
        loss_w = window_integral

        loss_mask = th.sum(intensity_mask, dim=-1)                    # [B]
        loss_mask = (loss_mask > 0.).type(intensity_mask.dtype)                    # [B]
        defined_window_integral = window_intensity_mask * add_window_integral
        no_window_integral = 1 - add_window_integral                    # [B]
        window_mask = xor(defined_window_integral, no_window_integral)  # [B]
        loss_mask = loss_mask * window_mask

        #Cumulative density#    
        if test:
            cumulative_density = 1 - th.exp(-ground_intensity_integral) #[B, L]
            cumulative_density[~intensity_mask.bool()] = -1      #[B,L]
            artifacts["cumulative density"] = cumulative_density.detach()  
        
        if 'last_h' in artifacts_modular:
            artifacts['last_h'] = artifacts_modular['last_h']
        if 'last_h_t' in artifacts_modular:
            artifacts['last_h_t'] = artifacts_modular['last_h_t']
            artifacts['last_h_m'] = artifacts_modular['last_h_m']

        return loss_t, loss_m, loss_w, loss_mask, artifacts 
    
    def artifacts(
            self, query: th.Tensor, events: Events, time_prediction = False, sampling = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the (log) intensities and intensity integrals at query times
        given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            log_intensity: [B,T,M] The log intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        if self.encoder is not None:
            representations, representations_mask, artifacts = self.encode(
                events=events, encoding_type='encoder')                            # [B,L+1,D] [B,L+1], Dict
        elif self.encoder_time is not None:
            representations_time, representations_mask, artifacts = self.encode(
                events=events, encoding_type='encoder_time')
            representations_mark, representations_mask, artifacts = self.encode(
                events=events, encoding_type='encoder_mark')
            representations = th.cat((representations_time, representations_mark), 0) #[2B, L+1, D] [B, L+1]
        prev_times, is_event, pos_delta_mask = get_prev_times( #
            query=query,
            events=events,
            allow_window=True,
            time_prediction=time_prediction,
            sampling=sampling)# ([B,T],[B,T]), [B,T], [B,T]
        
        prev_times, prev_times_idxs = prev_times  # [B,T], [B,T]
        return self.decode(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            is_event=is_event,
            pos_delta_mask=pos_delta_mask,
            representations=representations,
            representations_mask=representations_mask,
            artifacts=artifacts,
            time_prediction=time_prediction,
            sampling=sampling)

    def encode(self, events: Events, encoding_type: str) -> Tuple[th.Tensor, th.Tensor, Dict]:
        if encoding_type == "encoder":
            return self.encoder(events=events) 
        elif encoding_type == "encoder_time":
            return self.encoder_time(events=events)
        else:
            return self.encoder_mark(events=events)

    def decode(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            is_event: th.Tensor,
            pos_delta_mask: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None,
            time_prediction = False, 
            sampling = False
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
        return self.decoder(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            is_event=is_event,
            pos_delta_mask=pos_delta_mask,
            representations=representations,
            representations_mask=representations_mask,
            artifacts=artifacts)

    def cdf(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the cdf at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            cdf: [B,T] The (ground) cdf evaluated at each query time.
            intensity_mask: [B,T] Which cdf are valid for further
                computation based on e.g. sufficient history available.

        """
        log_intensity, intensity_integral, intensity_mask, _ = self.artifacts(
            query=query, events=events)
        ground_intensity_integral = intensity_integral.sum(-1)
        cdf = 1 - th.exp(-ground_intensity_integral)
        return cdf, intensity_mask
    

    def one_minus_cdf(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the 1-cdf at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            one_minus_cdf: [B,T] The 1 - (ground) cdf evaluated at each query time.
            intensity_mask: [B,T] Which cdf are valid for further
                computation based on e.g. sufficient history available.

        """
        log_intensity, intensity_integral, intensity_mask, _ = self.artifacts(
            query=query, events=events)
        ground_intensity_integral = intensity_integral.sum(-1)
        one_minus_cdf = th.exp(-ground_intensity_integral)
        return one_minus_cdf, intensity_mask