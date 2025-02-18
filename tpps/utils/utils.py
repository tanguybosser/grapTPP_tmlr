import torch as th


def smallest_positive(inputs, dim):
    """
    Args
        inputs: 3d array [B,T,L].
        dim: dimension on which the largest tj lower than t is evaluated.
    Return
        (delta_t, idx_delta_t), is_candidate:
            delta_t: t - tj, where th is the largest value lower than t
            idx_delta_t: position of the largest tj lower than t
            is_candidate: mask to remove padded points
    """
    non_positives = inputs <= 0                                   # [B,T,L]
    non_positives = non_positives.float()                         # [B,T,L]
    min_inputs, max_inputs = th.min(inputs), th.max(inputs)       # 1,1
    shift_matrix = (max_inputs - min_inputs) * non_positives * 2  # [B,T,L]
    shifted_matrix = inputs + shift_matrix                        # [B,T,L]

    is_candidate = inputs > 0                                     # [B,T,L]
    is_candidate = th.sum(is_candidate, dim=2)                    # [B,T]
    is_candidate = (is_candidate > 0).type(inputs.dtype)          # [B,T]

    result = th.min(shifted_matrix, dim=dim)                      # [B,T]

    return result, is_candidate                                  # [B,T], [B,T]

def detach(x: th.Tensor):
    return x.cpu().detach().numpy()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_enc(model):
    enc, rest = 0,0
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'encoder' in n:
                enc += p.numel()
            else:
                rest += p.numel()
    enc_prop = enc/total
    rest_prop = rest/total 
    print(f'Encoder Total: {enc}')
    print(f'Rest Total: {rest}')
    print(f'Encoder Proportion: {enc_prop}')
    print(f'Rest Proportion: {rest_prop}')
