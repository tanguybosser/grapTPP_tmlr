def get_acronym(models):
    
    models_thp = {
        'gru_thp_temporal_with_labels_adjust_param': 'THP',
        'gru_thp-jd_temporal_with_labels_adjust_param': 'THP-JD',
        'gru_temporal_with_labels_gru_temporal_with_labels_thp-dd_separate': 'THP-DD',
    }

    models_sa_thp = {
        'selfattention_thp_temporal_with_labels_adjust_param': 'SA-THP',
        'selfattention_thp-jd_temporal_with_labels_adjust_param': 'SA-THP-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_thp-dd_separate': 'SA-THP-DD',
    }

    models_sahp = {
        'gru_sahp_temporal_with_labels_adjust_param': 'SAHP',
        'gru_sahp-jd_temporal_with_labels_adjust_param': 'SAHP-JD',
        'gru_temporal_with_labels_gru_temporal_with_labels_sahp-dd_separate': 'SAHP-DD'
    }
    
    models_sa_sahp = {
        'selfattention_sahp_temporal_with_labels_adjust_param': 'SA-SAHP',
        'selfattention_sahp-jd_temporal_with_labels_adjust_param': 'SA-SAHP-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_sahp-dd_separate': 'SA-SAHP-DD'
    }

    models_fnn = {
        'poisson_gru_mlp-cm_temporal_with_labels_adjust_param':'FNN',
        'poisson_gru_mlp-cm-jd_temporal_with_labels_adjust_param':'FNN-JD',
        'poisson_gru_temporal_with_labels_gru_temporal_with_labels_mlp-cm-dd_separate':'FNN-DD'
    }

    models_sa_fnn = {
        'poisson_selfattention_mlp-cm_temporal_with_labels_adjust_param':'SA-FNN',
        'poisson_selfattention_mlp-cm-jd_temporal_with_labels_adjust_param':'SA-FNN-JD',
        'poisson_selfattention_temporal_with_labels_selfattention_temporal_with_labels_mlp-cm-dd_separate':'SA-FNN-DD'
    }
    
    models_rmtpp = {
        'gru_rmtpp_temporal_with_labels_adjust_param':'RMTPP',
        'gru_rmtpp-jd_temporal_with_labels_adjust_param':'RMTPP-JD',
        'gru_temporal_with_labels_gru_temporal_with_labels_rmtpp-dd_separate':'RMTPP-DD',
    }

    models_sa_rmtpp = {
        'selfattention_rmtpp_temporal_with_labels_adjust_param':'SA-RMTPP',
        'selfattention_rmtpp-jd_temporal_with_labels_adjust_param':'SA-RMTPP-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_rmtpp-dd_separate':'SA-RMTPP-DD',
    }

    models_lnm = {
        'gru_log-normal-mixture_temporal_with_labels_adjust_param': 'LNM',
        'gru_log-normal-mixture-jd_temporal_with_labels_adjust_param': 'LNM-JD',
        'gru_temporal_with_labels_gru_temporal_with_labels_log-normal-mixture-dd_separate': 'LNM-DD',
    }

    models_sa_lnm = {
        'selfattention_log-normal-mixture_temporal_with_labels_adjust_param': 'SA-LNM',
        'selfattention_log-normal-mixture-jd_temporal_with_labels_adjust_param': 'SA-LNM-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_log-normal-mixture-dd_separate': 'SA-LNM-DD',
    }

    models_lnm_joint = {
        'gru_joint-log-normal-mixture_temporal_with_labels_adjust_param': 'Joint-LNM'
    }

    models_poisson = {
          'identity_poisson_times_only':'Poisson'
    }

    models_smurf_thp = {
         'gru_smurf-thp-jd_temporal_with_labels_adjust_param':'STHP-JD',
         'gru_temporal_with_labels_gru_temporal_with_labels_smurf-thp-dd_separate':'STHP-DD'
    }

    map = {}
    maps = [models_poisson, models_smurf_thp,
            models_thp, models_sahp, models_fnn, models_rmtpp, models_lnm, models_lnm_joint,
            models_sa_thp, models_sa_sahp, models_sa_fnn, models_sa_rmtpp, models_sa_lnm, models_lnm_joint] 
    for mapping in maps:    
        map.update(mapping)
    models = [model.replace('_evaluation', '') if 'evaluation' in model else model for model in models]
    new_models = [map[model] for model in models]
    return new_models


def map_dataset_name(dataset):
    mapping = {
        'lastfm_filtered': 'LastFM',
        'lastfm':'LastFM',
        'mooc_filtered': 'MOOC', 
        'mooc': 'MOOC', 
        'mimic2_filtered': 'MIMIC2',
        'github_filtered': 'Github',
        'wikipedia_filtered':'Wikipedia', 
        'stack_overflow_filtered': 'Stack Overflow',
        'stack_overflow': 'Stack Overflow',
        'reddit_filtered_short':'Reddit',
        'reddit':'Reddit',
        'retweets_filtered_short': 'Retweets'
    }
    return mapping[dataset]


def map_model_name_cal(model):
    if 'gru_log-normal' in model:
            model_name = 'LNM'
    elif 'gru_cond-log-normal' in model:
            model_name = 'CLNM'
    elif 'sep-cond-log-normal' in model:
        if 'separate' in model:
            model_name = 'LNM-DSHS'
        else:
             model_name = 'DCLNM'
    elif 'selfattention' in model:
            model_name = 'SA/MC'
    elif 'hawkes' in model:
            model_name = 'Hawkes'
    elif 'mlp-mc' in model:
        model_name = 'MLP'
    elif 'mlp-cm' in model:
        if 'separate' in model:
            model_name = 'FNN-DSHS'
        else: 
            model_name = 'FNN'
    elif 'conditional-poisson' in model:
         model_name = 'EC'
    elif 'thp' in model:
        if 'sep-thp-mix_separate' in model:
            model_name = 'THP-DSHS'
        elif 'thp-mix' in model:
            model_name = 'THP-DCH'
        else:
            model_name = 'THP'
    elif 'sahp' in model:
        if 'separate' in model:
            model_name = 'SAHP-DSHS'
        elif 'sep-sahp' in model:
            model_name = 'SEP-SAHP'
        else:
            model_name = 'SAHP'
    elif 'rmtpp' in model:
        if 'separate' in model:
              model_name = 'RMTPP-DSHS'
        else:
             model_name = 'RMTPP'
    else:
        print(model)
        raise ValueError('Model not understood')
    return model_name
