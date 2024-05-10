# CNTPP Extension for Neurips

## Changes made since CNTPP paper.

### Datasets 

- All marks have been considered for all datasets (LastFM, MOOC, Reddit, Stack Overflow), and not only the 50 most represented marks. This considerably increases the datasets sizes for LastFM and Reddit, while Stack Overflow remains untouched. 

- For LastFM, the split proportions have been adapted to 40/10/40/10 for train/val/cal/test instead of 60/15/15/10 so that enough observations are present in the calibration set. This partially solves the problem of very high prediction regions for C-HDR-T. 

### C-HDR-T 

- Cumulative:
    - Working with the CDF for inverse sampling turned out to be troublesome as it would quickly saturate at 1 due to precision issues (especially for the Poisson model), which led to too narrow prediction regions on LastFM. Working instead with 1-CDF for the bineary search alleviated the issue and led to prediction regions of correct size.   

- Beta distribution:
    - To sample arrival-times, we first sampled $\alpha$ from a uniform $U[0,1]$, and then performed inverse sampling using these $\alpha$'s. However, this led to some issues given that not enough points were being sampled at 'extreme' values of the distribution, i.e. near the 0-quantile and near the 0.99-quantile, which in turn led to prediction regions of too narrow size on MOOC for Poisson. Using a Beta(0.5, 0.5) allows to sample more at the extreme values of the distribution, leading to more accurate prediction regions. 
    
    - The Beta distribution has been employed at two stages: (1) during the sampling of the $Z$ samples, and (2) when building candidate prediction regions. While this does not cause an issue for (2), we need to check that (1) is still valid. 

- Icrease number of samples:
    Whenever sampling was required, the number of samples has been increased substantially, leading to more stable predictions. 

- Removal of $\epsilon$ for $\alpha$
    An $\epsilon = $1e^{-4}$ we added to the $\alpha's$ during the sampling of the candidate prediction regions, which forbid to sample very close to 0. Removing this term enables to reach the desired coverage guarantees (the problem was present for Poisson on LastFM). 

### C-QR

- Bound switch
    - With CQR, it sometimes occur (espcially on Stack Overflow) that the lower bounds of the prediction region becomes greater than the upper bound, due to $\hat{q}$ being large and negative. Following recommendation from the literature, I switch the lower and upper bounds whenever the situation arises. 


### C-QRL
-  

