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

### Phase 1 Subjects (BT01–BT12) — Unpowered Baseline

These 12 subjects provide:
- Pure human biomechanics (no exo influence)
- Complete OpenSim joint angles, velocities, powers
- Biological joint moments (after subtracting interaction torques)
- Exoskeleton sensor data (hip/knee encoders, IMUs, insoles)
- Exoskeleton unpowered for **all** tasks
- Some tasks repeated with **heuristic control** enabled

**Purpose:** Establish clean ground-truth behavior suitable for training biological joint moment estimators. This is what should matter for us.

### Phase 2 Subjects (BT13–BT17) — Preliminary Model-In-The-Loop

These 5 subjects walked with:
- A **preliminary learned torque-estimation model** running in real time
- Exoskeleton **powered for all tasks**
- Interaction torques, desired torque, and measured torque available at 200 Hz

Files include:
- `torque_estimated`
    Output of the real-time neural network estimator used during Phases 2 and 3.  
    Represents the model’s prediction of the user’s biological joint moment.  
    Time-aligned to correct for estimator delay. Not ground truth.
- `torque_measured`
    Motor driver–derived torque representing the actual torque produced by the actuator.  
    Includes hardware dynamics (friction, inertia, gearing).

- `interaction_torque`
    Clean estimate of human–exo interaction torque:  
    `interaction_torque = torque_measured − motor_dynamics_compensation`.  
    This is the torque physically transmitted to the human and is used in computing biological joint moments.

**Purpose:** Capture human–exo coupling and provide powered training data.

### Phase 3 Subjects (BT01, BT02, BT13, BT18–BT24) — Validation Set

This final validation set contains:
- **10 total subjects**
- 3 returning subjects (BT01, BT02, BT13)
- 7 entirely new subjects (BT18–BT24)

All trials in Phase 3:
- Exoskeleton **powered**
- Include the full task set (28 tasks × 66 conditions)
- Serve as the **held-out test set** used in the publication

**Purpose:** Evaluate generalization to unseen users and unseen sessions.


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


#### **1. Exo.csv — Exoskeleton Sensor & Controller Data (Model Inputs)**  
This file contains all real-time measurements recorded directly from the exoskeleton hardware at **200 Hz**.  
It is the most feature-dense file and includes:

- **Joint kinematics**  
  - `hip_angle_{l,r}`, `hip_angle_{l,r}_velocity`, `hip_angle_{l,r}_velocity_filt`  
  - `knee_angle_{l,r}`, `knee_angle_{l,r}_velocity`, `knee_angle_{l,r}_velocity_filt`  

- **Actuator torque signals**  
  - `*_torque_estimated` — real-time neural-network estimator output  
  - `*_torque_desired` — controller-commanded torque  
  - `*_torque_measured` — actuator torque after motor sensing  
  - `*_torque_interaction` — human–exo interaction torque (measured minus modeled actuator dynamics)

- **IMU measurements (thigh, shank, foot; bilateral)**  
  - Accelerations: `{bodypart}_imu_{l,r}_accel_{x,y,z}`  
  - Angular rates: `{bodypart}_imu_{l,r}_gyro_{x,y,z}`  
  These IMUs are the primary sensing modality used for moment estimation.

- **Insole force & center-of-pressure**  
  - `insole_{l,r}_force_y`  
  - `insole_{l,r}_cop_x`, `insole_{l,r}_cop_z`

**Purpose:**  
Represents the full observable state of the exoskeleton. These signals serve as **model inputs** for joint-moment prediction and movement classification.

#### **2. Joint_Moments_Filt.csv — Ground-Truth Biological Joint Moments (Labels)**  
This file contains **filtered inverse-dynamics estimates of biological joint moments**, derived from motion capture + force plate data and aligned to the exoskeleton’s 200 Hz timeline.

Columns include:

- `hip_flexion_{l,r}_moment`  
- `hip_adduction_{l,r}_moment`  
- `hip_rotation_{l,r}_moment`  
- `knee_angle_{l,r}_moment`  
- `ankle_angle_{l,r}_moment`  
- `subtalar_angle_{l,r}_moment`

**Purpose:**  
Provides the **supervised learning targets** used for training the Temporal Convolutional Network described in the paper.


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
