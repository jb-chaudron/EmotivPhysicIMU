# emotivphysicimu

Small API for pairing Emotiv IMU movement features with EEG targets, fitting light artefact models, and plotting predictions.

```python

from emotivphysicimu.features import extract_features
from emotivphysicimu.constants import IMU_CHANNELS_UNIQUE, EMOTIV_CHANNELS
import mne
path = ""
raw = mne.io.read_raw(path)
raw.load_data()
raw = raw.filter(1, 40)
fs = extract_features(raw, imu_channels=IMU_CHANNELS_UNIQUE)
print("X:", fs.X.shape, "y:", fs.y.shape, "sfreq:", fs.sfreq)

from emotivphysicimu.model import IMURegressor
model = IMURegressor(sfreq=fs.sfreq, eeg_channels=EMOTIV_CHANNELS, channel_handling="all_channels",conformal=True)


y_pred = model.predict(fs.X)


from emotivphysicimu import IMUReport

IMUReport(model, X_train=fs.X, y_train=fs.y).generate("report.html")
```

The feature API expects two `mne.io.Raw` objects: one EEG stream and one IMU stream. EEG targets are averaged over `target_ratio` raw samples, so by default four EEG samples are paired with each IMU sample.

## Physics-inspired features

This package assumes two main physical drivers of artefacts:

1. Head movements
2. Ambient magnetic field

From these two drivers, it builds features that try to explain how motion and magnetic perturbations affect EEG recordings.

---

## 1. Magnetic field-related features

### 1.1 Pseudo-Lorentzian features

From the Lorentz force, a moving charge in a magnetic field experiences:

$$
\vec{F} = q\,(\vec{v} \times \vec{B})
$$

where $q$ is the charge, $\vec{v}$ the velocity, and $\vec{B}$ the magnetic field.

The exact microscopic charges and currents in EEG electrodes and cables are unknown, so the package does not attempt to model the true Lorentz force. Instead, it builds **proxy features** of the form $\vec{v} \times \vec{B}$, then lets a regression model learn how much these terms explain EEG artefacts.

In practice, the magnetic features are built in three steps:

1. Estimate the IMU translational velocity from linear acceleration.
2. Estimate rotational electrode velocity from head angular velocity and electrode position.
3. Compute cross-product terms with the measured magnetic field.

This leads to two main families of features:

- A translational term based on $\vec{v}_{\text{IMU}} \times \vec{B}$
- A rotational term based on $(\vec{\omega} \times \vec{r}_e) \times \vec{B}$

Each term can then be projected onto the electrode normal to obtain a scalar per-electrode feature.

### 1.2 Limitations

In typical recording environments, the magnetic field remains weak to moderate. The Earth's magnetic field is usually around 25 to 65 µT, and values around 100 µT still remain many orders of magnitude below MRI-scale fields.

For that reason, these pseudo-Lorentzian features should be treated as empirical nuisance regressors rather than exact physical forces. In practice, they may explain a small fraction of variance, but they are unlikely to be the dominant source of artefacts outside extreme magnetic environments.

---

## 2. Head movement features

The package models the head as a rigid body and electrodes as points placed on a sphere, using 10-20 or Emotiv-compatible coordinates. This is an approximation, but it is useful for building interpretable motion features tied to electrode geometry.

### 2.1 Kinematic model

Let:

- $\vec{r}_e$: position of electrode $e$ in the head frame
- $\vec{n}_e$: outward unit normal of electrode $e$, defined as $\vec{n}_e = \vec{r}_e / \|\vec{r}_e\|$
- $\vec{a}_{\text{trans}}$: translational acceleration of the head
- $\vec{\omega}$: angular velocity of the head
- $\vec{\alpha} = \dot{\vec{\omega}}$: angular acceleration of the head

The total acceleration at electrode $e$ is approximated by:

$$
\vec{a}_e = \vec{a}_{\text{trans}} + \vec{\alpha} \times \vec{r}_e + \vec{\omega} \times (\vec{\omega} \times \vec{r}_e)
$$

This decomposes electrode motion into three effects:

- $\vec{a}_{\text{trans}}$: global head translation
- $\vec{\alpha} \times \vec{r}_e$: tangential acceleration due to changes in rotation
- $\vec{\omega} \times (\vec{\omega} \times \vec{r}_e)$: centripetal acceleration due to sustained rotation

### 2.2 Pressure features

Pressure-like features are defined as the component of motion aligned with the electrode normal.

For any electrode acceleration $\vec{a}_e$, the scalar normal component is:

$$
p_e = \vec{a}_e \cdot \vec{n}_e
$$

This scalar is used as a proxy for pressure variation along the electrode axis. Because $\vec{a}_e$ is a sum of three terms, three separate pressure features can be built:

- $p_{\text{trans},e} = \vec{a}_{\text{trans}} \cdot \vec{n}_e$
- $p_{\alpha,e} = (\vec{\alpha} \times \vec{r}_e) \cdot \vec{n}_e$
- $p_{\omega^2,e} = [\vec{\omega} \times (\vec{\omega} \times \vec{r}_e)] \cdot \vec{n}_e$

These features are interpretable and can be given separately to the model so that it learns which physical contribution matters most for each channel.

A total pressure feature can also be defined as:

$$
p_{\text{total},e} = p_{\text{trans},e} + p_{\alpha,e} + p_{\omega^2,e}
$$

### 2.3 Shear features

Shear-like features describe motion in the plane orthogonal to the electrode normal, that is, lateral motion relative to the skin.

For any vector $\vec{u}_e$, its normal projection is:

$$
\vec{u}_{\text{normal},e} = (\vec{u}_e \cdot \vec{n}_e)\,\vec{n}_e
$$

Its orthogonal planar component is then:

$$
\vec{u}_{\text{shear},e} = \vec{u}_e - \vec{u}_{\text{normal},e}
$$

and the corresponding scalar shear feature is:

$$
s_e = \|\vec{u}_{\text{shear},e}\|
$$

This can be applied independently to each term of the motion model:

- $s_{\text{trans},e}$: shear due to head translation
- $s_{\alpha,e}$: shear due to changes in rotation
- $s_{\omega^2,e}$: shear due to sustained rotation

A total shear feature can also be obtained from the full acceleration vector.

### 2.4 Electrode speed

Using angular velocity and electrode position, the linear velocity of electrode $e$ is:

$$
\vec{v}_e = \vec{\omega} \times \vec{r}_e
$$

The scalar speed feature is then:

$$
v_e = \|\vec{v}_e\|
$$

This feature acts as a proxy for overall mechanical intensity. Faster electrode motion can be associated with stronger traction, larger inertial effects, and larger movement artefacts.

### 2.5 Jerk

Jerk is the time derivative of acceleration, or more generally the time derivative of a motion feature:

$$
\vec{j}(t) = \frac{d\vec{a}(t)}{dt}
$$

In practice, jerk is often computed numerically as a gradient of a scalar motion feature such as electrode speed or pressure. High jerk values indicate abrupt motion changes such as kicks, shocks, or fast head movement transitions, which are often more artefact-inducing than smooth motion.

## 3. Raw features

We also include the raw IMU data as features. This is useful to have a full picture of the movement and to be able to use the IMU data in a more flexible way. One can also stack several windows of features to increase the temporal resolution.

---

## Practical interpretation

The package is not meant to recover exact contact forces or exact electromagnetic forces. Instead, it produces physically informed regressors that are useful for explaining variance in EEG recordings through:

- pressure-like effects along the electrode axis,
- shear-like effects along the skin surface,
- overall motion intensity,
- abrupt mechanical transitions,
- and magnetic coupling proxies.

These features can be fed into lightweight linear or regularized models to estimate the movement-related contribution to EEG activity, either channel-wise or jointly across channels.

---

# Models 

We've included three models in the package:

- A linear regression model
- A Tweedie regression model (a generalization of the Poisson distribution to include overdispersion)
- A CatBoost Regressor model that can handle categorical features and use channel name as confounder.

On top of that, we allow for a residual fitting of the models with a first model taking into account the head movements, the second the magnetic field effect, the last a being fitted on the residual of the previous two models with the raw features.

Finally a conformal head can be added to mitigate the effect of the noise removal depending on the spread of the conformal interval.

# Quality metrics

We use the following quality metrics to evaluate the models:

**Prediction related**

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score

**Correlation between IMU and EEG**

- Coherence
- Mutual Information

**EEG related**

- Spectral slope $f_\beta$
- Ratio of high frequency power to total power $HFP/TP$
- Distribution Kurtosis ~ 3 as the EEG are assumed to be as Gaussian distributed

**Visualization**

- Scatter plot of true vs predicted
- Time series of true vs predicted
- Residual plot
- Correlation matrix between IMU and EEG
- Correlation matrix between IMU and EEG
- ICA components before and after noise removal