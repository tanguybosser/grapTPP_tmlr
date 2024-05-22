def get_acronym(models):
    
    models_thp = {
        'gru_thp_temporal_with_labels': 'THP',
        'gru_thp-jd_temporal_with_labels': 'THP+',
        'gru_temporal_with_labels_gru_temporal_with_labels_thp-dd': 'THP++',
    }

    models_sa_thp = {
        'selfattention_thp_temporal_with_labels': 'SA-THP',
        'selfattention_thp-jd_temporal_with_labels': 'SA-THP-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_thp-dd': 'SA-THP-DD',
    }

    models_sahp = {
        'gru_sahp_temporal_with_labels': 'SAHP',
        'gru_sahp-jd_temporal_with_labels': 'SAHP+',
        'gru_temporal_with_labels_gru_temporal_with_labels_sahp-dd': 'SAHP++'
    }
    
    models_sa_sahp = {
        'selfattention_sahp_temporal_with_labels': 'SA-SAHP',
        'selfattention_sahp-jd_temporal_with_labels': 'SA-SAHP-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_sahp-dd': 'SA-SAHP-DD'
    }

    models_fnn = {
        'gru_mlp-cm_temporal_with_labels':'FNN',
        'gru_mlp-cm-jd_temporal_with_labels':'FNN+',
        'gru_temporal_with_labels_gru_temporal_with_labels_mlp-cm-dd':'FNN++'
    }

    models_sa_fnn = {
        'selfattention_mlp-cm_temporal_with_labels':'SA-FNN',
        'selfattention_mlp-cm-jd_temporal_with_labels':'SA-FNN-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_mlp-cm-dd':'SA-FNN-DD'
    }
    
    models_rmtpp = {
        'gru_rmtpp_temporal_with_labels':'RMTPP',
        'gru_rmtpp-jd_temporal_with_labels':'RMTPP+',
        'gru_temporal_with_labels_gru_temporal_with_labels_rmtpp-dd':'RMTPP++',
    }

    models_sa_rmtpp = {
        'selfattention_rmtpp_temporal_with_labels':'SA-RMTPP',
        'selfattention_rmtpp-jd_temporal_with_labels':'SA-RMTPP-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_rmtpp-dd':'SA-RMTPP-DD',
    }

    models_lnm = {
        'gru_log-normal-mixture_temporal_with_labels': 'LNM',
        'gru_log-normal-mixture-jd_temporal_with_labels': 'LNM+',
        'gru_temporal_with_labels_gru_temporal_with_labels_log-normal-mixture-dd': 'LNM++',
    }

    models_sa_lnm = {
        'selfattention_log-normal-mixture_temporal_with_labels': 'SA-LNM',
        'selfattention_log-normal-mixture-jd_temporal_with_labels': 'SA-LNM-JD',
        'selfattention_temporal_with_labels_selfattention_temporal_with_labels_log-normal-mixture-dd': 'SA-LNM-DD',
    }

    models_lnm_joint = {
        'gru_joint-log-normal-mixture_temporal_with_labels': 'Joint-LNM'
    }

    models_poisson = {
          'identity_poisson_times_only':'Poisson'
    }

    models_smurf_thp = {
         'gru_smurf-thp-jd_temporal_with_labels':'STHP+',
         'gru_temporal_with_labels_gru_temporal_with_labels_smurf-thp-dd':'STHP++'
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
        if 'sep-thp-mix' in model:
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
