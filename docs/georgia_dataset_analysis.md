# Georgia Tech Dataset Analysis

Official Document Source: https://doi.org/10.1038/s41586-024-08157-7

Dataset Download: https://doi.org/10.35090/gatech/75759

Associated Paper: https://www.nature.com/articles/d41586-024-03546-4

Supplementary Information: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08157-7/MediaObjects/41586_2024_8157_MOESM1_ESM.pdf


## Dataset Structure
Number of subjects: 15 users <br>
<small>(The first 10 users include both unactuated data for all tasks and actuated data for those that could be mimicked with a heuristic controller. The following five users include actuated data for all tasks using a preliminary joint moment estimation model)</small>

| **Dataset Component** | **Participants** | **Sessions per Subject** | **Activities / Conditions** | **Notes** |
|------------------------|------------------|---------------------------|------------------------------|-----------|
| **Phase 1** | 10 | 1 | 28 tasks (66 total conditions) | exo off + heuristic actuated |
| **Phase 2** | 5 | 1 | 28 tasks (66 total conditions) | exo on, pilot model |
| **Phase 3** | 10 | 3 | 28 tasks | Exoskeleton on, human outcomes |
| **Total** | **22 unique subjects** | — | **66 conditions × 28 tasks** | **>22 million  samples** |

- 22 particpants
- 66 conditions from 28 task groups
- 1-3 sessions per participant
- \> 22 million labels of ground-truth moments across the left and right legs per lower-limb joint <br>

## Movement / Gestures
### Cyclic
- level ground work
- 25 lb loaded walk
- backwards walk
- toe and heel walk
- inclined walk
- declined walk
- stair ascent
- stair descent
- run
### Impedance-like
- standing poses
- lunge
- sit and stand
- tug of war
- medicine ball toss
- step up
- jump across
- jump in place
- lift and place weight
- squat
### Unstructured
- start and stop
- cut
- step over
- turn
- meander
- twister
- push and pull recovery
- calisthentics
- curb

## Technical Specifications
### Sampling Rate:
- Exoskeleton control loop: 55 Hz
- Butterworth low pass filter: 10 Hz
- Data upsampled to 200 Hz to match frequency previously used for TCN
- Motion capture system: 200 Hz
- Overground force plates and an instrumented treadmill: 1,000 Hz
- (two participants in phase 1 were collected at 120 Hz and upsampled to 200)
### Sensor types:
- 6 IMU's **(most important)**: thigh, shank, foot (bilateral)
- joint encoders on hips & knees
- wireless force-sensitive insoles
- force-plates/instrumented treadmill
- exoskeleton controller sensors

### Data format

TCN Implementation: https://codeocean.com/capsule/5421243/tree/v2

folder contains 2 .csv's:
- **Exo.csv**: contains data from joint encoders moments
- **Joint_Moments_Filt.csv**: contains data from IMU's

^^ that's pretty much all they train their NN on, the rest of the data is used more for analysis

## Data Distribution Analysis
### Class balance
28 movement classes divided into: cyclic, impedance-like, unstructured

### Sample lengths
data is at 200 hz after up-sampling.
### Missing data
first two participants were recorded at 120 Hz and upsampled to 200 Hz.
## Known Issues & Limitations
- all participants were able-bodied adults (data may not generalize to mobility impairments)
- data collected in lab environment with controlled instrumentation
- although dataset is pretty extensive, the paper mentions using novel tasks for generalization (basketball layup, burpee), which the moderl underperformed in. (performance deterioriates at tasks at the extreme boundaries)
> "when pushed to extremely dynamic behaviours outside of the training set, our approach provided directionally correct assistance, but the magnitude and shape lost accuracy"
- sample lengths vary by task, could bias modelling
- the upsampling of two participants could introduce interpolation artifacts
- modelling architecture was relatively simple and participants limited ([see TCN architecture](https://www.nature.com/articles/s41586-024-08157-7/tables/1)), the paper mentions a performance ceiling
