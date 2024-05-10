## Files to run 

- CLNM
    - 'log-normal-mixture'
    - 'cond-log-normal-mixture'
    - 'sep-cond-log-normal-mixture'

- RMTPP 
    - 'rmtpp'
    - 'sep-rmtpp-ch'
    - 'sep-rmtpp'

- FNN 
    - 'mlp-cm'
    - 'mlp-cm-mix'
    - 'sep-mlp-cm-mix'

- THP 
    - 'thp'
    - 'thp-mix'
    - 'sep-thp-mix'

- SAHP 
    - 'sahp'
    - 'sahp-mix'
    - 'sep-sahp-mix'


## To check 

- Check if LNM-Joint runs (NLL-T, NLL-M) are ok on LastFM. 
    - Looks good so far, wait for runs on other datasets. 

- Launch all models with SA mechanisms.  
    - Check init if it requires changes. 

- Simulations on Hawkes 
    - Start with 5 marks.
        - On one split for now. 
        - Check the intensity modeled by all setups compared to the true intensity.
            - Can we use tick, on do we need to use the oracle ?   


- FNN results (Base, JD, DD) are not very good on the mark.
    - Check if Poisson term helps. 


## TO DO for Reviews 

- Implement and launch LNM-Joint 
    - **done**
    - launch on remaining runs (3,4) **Ongoing**

- Run experiments on simulate Hawkes datasets 
    - Generate scripts for visualisation. **done**
    - Compute average distance from true indentity for all models. 
    - Run hawkes model. Do we get close ? **done** Yes, it matches almost perfectly 
    - Run on Hawkes large, with max-epochs at 1000 --> Are we still far ? 

- Run all model with SA encoders. 
    - **Ongoing**

- Implement and launch baselines from Lin and SMURF-THP. 
    - Smurf-THP-JD, Smurf-THP-DD, all datasets **Ongoing**

## Parameters on LastFM for SA runs 
- THP
    - Base 9794
    - JD 9978 **this may need to be rerun because it was actually 11746** (Check if results are ok first)
    - DD 10010

- SAHP
    - Base 11652
    - JD 11952
    - DD 12058

- RMTPP
    - Base 10568
    - JD 10510
    - DD 10238

- LNM
    - Base 9122
    - JD 9138
    - DD 9098

- FNN
    - Base 7823
    - JD 7857
    - DD 7908


## Proportions param encoder/rest LastFM

## Runs must be adapted as such. 

RMTPP
    - Base (9384/5936= 15320) 0.61/0.39
    - JD (9384/5934=15318) **something weird here** 0.61/0.39
    - DD (8900/5934=14834) 0.6/0.4

LNM
    - Base (9384/4546=13930) 0.67/0.33
    - JD (9384/4562=13946) 0.67/0.33
    - DD (8672/4562=13234) 0.65/0.35

FNN
    - Base (9384/3471=12855) 0.73/0.27
    - JD (9384/3653=13037) 0.72/0.28
    - DD (8900/3888=12788) 0.7/0.3

THP
    - Base (9384/4754=14138) 0.66/0.34
    - JD (9384/5402=14786) 0.63/0.37
    - DD (8900/5654=14554) 0.61/0.39

SAHP 
    - Base (9384/6204=15588) 0.6/0.4
    - JD (9384/6178=15562) 0.6/0.4
    - DD (8900/6614=15514) 0.57/0.43