# Modeling Analog Error in a CVAE for Scene-Aware Multiple Trajectory Prediction

## Project Overview

This project implements a conditional variational autoencoder (CVAE) for multimodal trajectory prediction on KITTI-style vehicle trajectory datasets with scene context. The primary objective is to model the conditional distribution of future trajectories given past observations and rasterized scene representations, while also investigating the impact of analog in-memory computing on prediction accuracy.

The project enables direct comparison between digital PyTorch inference and analog device-level simulation using CrossSim, providing insights into how hardware non-idealities affect trajectory prediction performance.

---

## 1. Problem Formulation

### 1.1 Trajectory Prediction Task

The goal is to model the conditional distribution of future trajectories given the past trajectory and a rasterized scene:

$$\hat{p}_\theta(y|c)$$

where:
- $y$ is the future trajectory (a sequence of 2D points over 4 seconds)
- $c$ is the conditioning variable, consisting of:
  - the past trajectory (2 seconds of 2D points)
  - a top-down scene representation (one-hot raster of lanes, roads, etc.)

The model is trained as a generative model: it defines an explicit latent variable model $\hat{p}_\theta(x,y|c)$ and learns $\theta$ by maximizing a variational lower bound on the conditional log-likelihood $\log \hat{p}_\theta(y|c)$.

---

## 2. Generative Model Architecture

### 2.1 Latent Variable Model

We introduce a latent variable $x$ that captures high-level intentions/modes of future motion (e.g., going straight, turning, changing lanes). For each observed pair $(c,y)$:

1. Sample latent state from a conditional prior: $\hat{p}_\theta(x|c)$
2. Given $(x,c)$, sample the future trajectory: $\hat{p}_\theta(y|x,c)$

The joint model is:

$$\hat{p}_\theta(x,y|c) = \hat{p}_\theta(x|c) \cdot \hat{p}_\theta(y|x,c)$$

### 2.2 Parameterization

We use Gaussian distributions with neural-network parameters:

**Conditional prior:**

$$\hat{p}_\theta(x|c) = \mathcal{N}(x; \mu_p(c;\theta), \Sigma_p(c;\theta))$$

where $\mu_p$ and $\Sigma_p$ are outputs of a neural network that takes the encoded past trajectory and encoded scene as inputs.

**Emission model:**

$$\hat{p}_\theta(y|x,c) = \mathcal{N}(y; \mu_y(x,c;\theta), \sigma^2 I)$$

In the implementation, $\mu_y(x,c;\theta)$ is produced by an autoregressive GRU decoder that rolls out a sequence of future displacements, with a fixed, isotropic covariance $\sigma^2 I$. Using mean-squared error (MSE) as the reconstruction term corresponds exactly to this Gaussian assumption.

---

## 3. Recognition Model (Approximate Posterior)

Exact expectations under the true posterior are intractable:

$$\hat{p}_\theta(x|y,c) \propto \hat{p}_\theta(x|c) \cdot \hat{p}_\theta(y|x,c)$$

We introduce a recognition distribution (encoder):

$$\hat{r}_\phi(x|y,c) = \mathcal{N}(x; \mu_r(y,c;\phi), \Sigma_r(y,c;\phi))$$

where $\mu_r$ and $\Sigma_r$ are functions of:
- the encoded past trajectory
- the encoded future trajectory
- the encoded scene

All three encoders are GRU-based (plus some 1D/2D convolutions), so $\mu_r$ and $\Sigma_r$ are effectively neural networks over the whole sequence and scene.

---

## 4. Conditional ELBO and Training Objective

For a given context $c$ and future $y$, the conditional log-likelihood is:

$$\log \hat{p}_\theta(y|c) = \log \int \hat{p}_\theta(x|c) \cdot \hat{p}_\theta(y|x,c) \, dx$$

We introduce the recognition distribution and use Jensen's inequality to obtain the conditional evidence lower bound (ELBO):

$$\log \hat{p}_\theta(y|c) \geq \mathbb{E}_{x \sim \hat{r}_\phi(x|y,c)}[\log \hat{p}_\theta(y|x,c)] - \text{KL}(\hat{r}_\phi(x|y,c) \| \hat{p}_\theta(x|c))$$

The loss function is the negative ELBO:

$$\mathcal{L}(\theta,\phi) = -\mathbb{E}_{x \sim \hat{r}_\phi(x|y,c)}[\log \hat{p}_\theta(y|x,c)] + \text{KL}(\hat{r}_\phi(x|y,c) \| \hat{p}_\theta(x|c))$$

We use a **β-weighted version**:

$$\mathcal{L}_\beta(\theta,\phi) = -\mathbb{E}_{x \sim \hat{r}_\phi(x|y,c)}[\log \hat{p}_\theta(y|x,c)] + \beta \cdot \text{KL}(\hat{r}_\phi(x|y,c) \| \hat{p}_\theta(x|c))$$

with $\beta$ annealed from a small value to 1 over the first several epochs. This encourages the model to first learn good reconstruction before enforcing strong prior-posterior match.

### 4.1 Connection to Code

Using the Gaussian emission with fixed covariance:

$$-\log \hat{p}_\theta(y|x,c) = \frac{1}{2\sigma^2} \|y - \mu_y(x,c;\theta)\|^2 + \text{const}$$

In the code:
```python
recon_loss = F.mse_loss(future_pred, future, reduction="mean")
```

The KL term is computed in closed form between two diagonal Gaussians:
```python
kl_loss = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
```

Final loss:
```python
loss = recon_loss + beta * kl_loss
```

---

## 5. Network Architecture

### 5.1 Context Encoding

**Past trajectory** $c_{\text{past}} \in \mathbb{R}^{T_{\text{past}} \times 2}$:
- Encoded by 1D convolution + GRU into $h_{\text{past}}(c) \in \mathbb{R}^D$

**Scene** $c_{\text{scene}}$:
- Raster of size $180 \times 180 \times 4$, passed through 2D convolutions and a GRU
- Produces $h_{\text{scene}}(c) \in \mathbb{R}^D$

Context embedding: $h_c = [h_{\text{past}}(c), h_{\text{scene}}(c)]$

### 5.2 Generative Model

**Conditional prior:**

$$\hat{p}_\theta(x|c) = \mathcal{N}(x; \mu_p(h_c;\theta), \Sigma_p(h_c;\theta))$$

where `fc_prior` computes $[\mu_p, \log \Sigma_p]$.

**Decoder:**

Given sampled $x$ and context $c$, form initial hidden state:

$$h_{\text{dec},0} = f_\theta(h_{\text{past}}(c), h_{\text{scene}}(c), x)$$

Use this as the initial hidden state of a GRU that autoregressively generates future displacements, producing $\mu_y(x,c;\theta)$.

### 5.3 Recognition Model

**Encoder/recognition distribution:**

$$\hat{r}_\phi(x|y,c) = \mathcal{N}(x; \mu_r(h_c, h_{\text{fut}};\phi), \Sigma_r(h_c, h_{\text{fut}};\phi))$$

where $h_{\text{fut}}$ is obtained from the future trajectory via conv+GRU, and `fc_q` produces $[\mu_r, \log \Sigma_r]$.

Sampling $x$ from $\hat{r}_\phi(x|y,c)$ uses the reparameterization trick.

---

## 6. Training and Evaluation

### 6.1 Training

**Objective:** Minimize the β-weighted negative conditional ELBO $\mathcal{L}_\beta(\theta,\phi)$ over the training set.

**β-annealing:** $\beta$ starts near 0 and linearly increases to 1 over a fixed number of warmup epochs. This allows the model to first focus on accurate reconstruction before enforcing alignment between the posterior and prior distributions.

The training script:
- Loads `CVAE_Predictor`
- Iterates over mini-batches $(c,y)$
- Computes `loss, recon, kl = model.compute_loss(...)`
- Updates $\theta, \phi$ via Adam optimizer
- Periodically evaluates on test set using ADE/FDE metrics (in meters at 1s, 2s, 3s, 4s horizons)

### 6.2 Sampling/Generation

At test time, to generate multimodal trajectory predictions for context $c = (c_{\text{past}}, c_{\text{scene}})$:

1. Encode the context to get $h_{\text{past}}(c)$ and $h_{\text{scene}}(c)$
2. Compute the conditional prior $\hat{p}_\theta(x|c)$
3. Draw samples $x^{(1)}, \ldots, x^{(K)} \sim \hat{p}_\theta(x|c)$ using the reparameterization trick
4. For each $x^{(k)}$, run the decoder GRU to obtain trajectory mean $\mu_y(x^{(k)}, c;\theta)$

In code:
```python
model.sample(past, scene_one_hot, num_samples=K)
```
returns a tensor of shape `[B, K, T_future, 2]` with $K$ trajectories per agent.

---

## 7. Analog In-Memory Computing with CrossSim

### 7.1 SONOS Device Technology

SONOS (Silicon-Oxide-Nitride-Oxide-Silicon) is a charge-trapping memory technology used in analog in-memory computing. Unlike traditional floating-gate memory, SONOS devices store charge in a nitride layer, offering several advantages:

- **Non-volatility:** Retains information without power
- **Multi-level storage:** Can represent analog weights through varying charge levels
- **Scalability:** Compatible with standard CMOS processes
- **Lower programming voltages:** Compared to floating-gate devices

In the context of analog computing, SONOS devices are arranged in crossbar arrays where:
- Conductance values represent neural network weights
- Matrix-vector multiplication is performed through Ohm's law and Kirchhoff's current law
- Analog voltage inputs produce current outputs proportional to stored weights

### 7.2 CrossSim Analog Modeling

**CrossSim** is a device-level simulator for analog in-memory computing that models the non-ideal behavior of crossbar arrays. For this project, we use CrossSim to simulate SONOS-based RRAM (Resistive Random-Access Memory) arrays.

**Key device parameters:**
- **Resistance range:** $R_{\text{min}} = 50 \text{ k}\Omega$ to $R_{\text{max}} = 50 \text{ M}\Omega$
- **Read voltage:** $V_{\text{read}} = R_{\text{min}} \times 1 \mu\text{A}$
- **Programming error:** Stochastic variation in weight programming
- **Device model:** SONOS-specific error characteristics

**Simulation configuration:**
```python
params = CrossSimParameters()
params.xbar.device.Rmin = 50e3
params.xbar.device.Rmax = 50e6
params.xbar.device.Vread = params.xbar.device.Rmin * 1e-6
params.xbar.device.programming_error.enable = True
params.xbar.device.programming_error.model = "SONOS"
```

### 7.3 Sources of Digital-Analog Differences

When converting from digital PyTorch models to analog hardware simulation, several non-idealities introduce differences:

1. **Weight quantization:** Continuous weights are mapped to discrete conductance levels within $[R_{\text{min}}, R_{\text{max}}]$
2. **Programming error:** Stochastic variation when writing weights to analog devices
3. **Asymmetric weight encoding:** Positive and negative weights require different crossbar configurations
4. **Crossbar range clipping:** Values outside the resistance range are clipped
5. **Limited precision:** Analog devices have inherent precision limits unlike floating-point arithmetic

### 7.4 Handling Stochasticity

Analog device simulation introduces stochastic behavior due to programming errors and device variability. To obtain statistically robust results:

- **Multiple runs:** Each analog configuration is evaluated 3 times (configurable via `--analog_runs`)
- **Statistical aggregation:** Mean and standard deviation are computed across runs
- **Comparison methodology:** Digital results (single deterministic run) are compared against analog results (averaged over multiple stochastic runs)

The evaluation script `eval_multiple_cvae.py` implements this methodology:

```python
for run_idx in range(args.analog_runs):
    metrics = run_single_evaluation(config, run_label)
    # Accumulate metrics across runs
    for k, v in metrics.items():
        analog_results[k].append(v)

# Compute statistics
mean_val = np.mean(values)
std_val = np.std(values, ddof=1)
```

### 7.5 CrossSim Limitations

**Important Note:** The analog memory simulation uses CrossSim, which does not natively support simulating GRU layers in the stable release. The project was originally completed using the `cross_sim_development` repository (a non-public branch) which does support AnalogGRU.

For the results presented here using the public CrossSim release:
- Conv2D and Linear layers are fully simulated with analog non-idealities
- GRU layers remain in digital mode
- Results are largely similar to the full analog simulation
- This provides insights into how convolutional and linear operations degrade under analog constraints

---

## 8. Relation to "Generative vs. Discriminative"

Even though we train on paired data $(c,y)$, this is not a purely discriminative model. Instead, we have:

- An explicit latent variable $x$ representing motion modes
- A joint conditional model $\hat{p}_\theta(x,y|c)$
- A conditional prior $\hat{p}_\theta(x|c)$ that captures diverse future possibilities
- A likelihood $\hat{p}_\theta(y|x,c)$ with probabilistic interpretation

We fit $\theta$ and $\phi$ by maximizing a variational lower bound on $\log \hat{p}_\theta(y|c)$, exactly in the spirit of VAE/variational inference, extended to the conditional setting. The model can generate samples of $y$ for a given context $c$ by ancestral sampling through the latent variable, which is the defining property of a generative model.

---

## 9. Experimental Results

### 9.1 Model Configuration

```yaml
dataset_file: kitti_dataset.json
past_len: 20
future_len: 40
dim_embedding_key: 48
latent_dim: 8
batch_size: 32
learning_rate: 0.0001
max_epochs: 200
beta_max: 0.8
beta_warmup_epochs: 80
num_samples_eval: 5
```

### 9.2 Example Predictions

The model generates multiple diverse trajectory predictions for each scenario, capturing the multimodal nature of future motion:

![CVAE Trajectory Test Predictions Example 1](https://github.com/MadSons/ECE60131-Project/blob/main/test/2025-12-10_21-57-05_cvae_eval_14-14-19/example_002.png)
![CVAE Trajectory Test Predictions Example 2](https://github.com/MadSons/ECE60131-Project/blob/main/test/2025-12-10_21-57-05_cvae_eval_14-14-19/example_003.png)


*Figure: Multiple trajectory predictions generated by the CVAE model. Blue indicates past trajectory, green shows ground truth future, and red trajectories represent K=5 predictions ranked by likelihood (darker = higher confidence).*

### 9.3 Quantitative Results

**Best of 5 Samples (Digital):**

| Metric | Value (m) |
|--------|-----------|
| eucl_mean | 1.1471 |
| ADE_1s | 0.2323 |
| ADE_2s | 0.4373 |
| ADE_3s | 0.7315 |
| FDE_1s (horizon10s) | 0.3809 |
| FDE_2s (horizon20s) | 0.8937 |
| FDE_3s (horizon30s) | 1.7255 |
| FDE_4s (horizon40s) | 3.0206 |

### 9.3 Comparison with MANTRA Baseline

Performance comparison with the MANTRA paper and baseline methods:

**Average Displacement Error (ADE) in meters:**

| Method | 1s | 2s | 3s | 4s |
|--------|-----|-----|-----|-----|
| Kalman | 0.51 | 1.14 | 1.99 | 3.03 |
| Linear | 0.20 | 0.49 | 0.96 | 1.64 |
| MLP | 0.20 | 0.49 | 0.93 | 1.53 |
| MANTRA (top 1) | 0.24 | 0.57 | 1.08 | 1.78 |
| MANTRA (top 5) | 0.17 | 0.36 | 0.61 | 0.94 |
| MANTRA (top 10) | 0.16 | 0.30 | 0.48 | 0.73 |
| MANTRA (top 20) | 0.16 | 0.27 | 0.40 | 0.59 |
| **CVAE (top 5)** | **0.23** | **0.44** | **0.74** | **1.17** |
| **CVAE (top 10)** | **0.23** | **0.41** | **0.66** | **1.02** |

**Final Displacement Error (FDE) in meters:**

| Method | 1s | 2s | 3s | 4s |
|--------|-----|-----|-----|-----|
| Kalman | 0.97 | 2.54 | 4.71 | 7.41 |
| Linear | 0.40 | 1.18 | 2.56 | 4.73 |
| MLP | 0.40 | 1.17 | 2.39 | 4.12 |
| MANTRA (top 1) | 0.44 | 1.34 | 2.79 | 4.83 |
| MANTRA (top 5) | 0.30 | 0.75 | 1.43 | 2.48 |
| MANTRA (top 10) | 0.26 | 0.59 | 1.07 | 1.88 |
| MANTRA (top 20) | 0.25 | 0.49 | 0.83 | 1.49 |
| **CVAE (top 5)** | **0.38** | **0.91** | **1.76** | **3.09** |
| **CVAE (top 10)** | **0.37** | **0.80** | **1.49** | **2.67** |

**Key Observations:**
- The CVAE model demonstrates competitive performance with MANTRA, particularly at shorter horizons (1-2 seconds)
- Performance improves with more samples (best-of-K), showcasing the model's ability to capture multimodal trajectory distributions
- The model effectively leverages scene context through the rasterized top-down representation
- While MANTRA achieves better performance at longer horizons, the CVAE provides a simpler, more interpretable architecture
- The generative nature of the CVAE allows for explicit uncertainty quantification through the latent variable

### 9.4 Digital vs. Analog Performance

The analog simulation reveals significant impacts of hardware non-idealities on prediction accuracy. The following figure compares performance across all metrics:

![Digital vs Analog Performance Comparison](https://github.com/MadSons/ECE60131-Project/blob/main/comparison_plot.png)

*Figure: Performance comparison between digital and analog implementations for both CVAE and MANTRA models. Error bars represent standard deviation across 10 analog runs.*

#### CVAE Performance Analysis

**Digital Performance (Single Run):**

| Metric | Value (m) |
|--------|-----------|
| eucl_mean | 1.1471 |
| ADE_1s | 0.2323 |
| ADE_2s | 0.4373 |
| ADE_3s | 0.7315 |
| FDE_1s (horizon10s) | 0.3809 |
| FDE_2s (horizon20s) | 0.8937 |
| FDE_3s (horizon30s) | 1.7255 |
| FDE_4s (horizon40s) | 3.0206 |

**Analog Performance (Mean ± Std over 10 runs):**

| Metric | Value (m) | Relative Increase |
|--------|-----------|-------------------|
| eucl_mean | 2.3116 ± 0.5797 | **+101.5%** |
| ADE_1s | 0.5867 ± 0.2115 | **+152.6%** |
| ADE_2s | 1.0480 ± 0.3031 | **+139.7%** |
| ADE_3s | 1.6164 ± 0.4180 | **+120.9%** |
| FDE_1s (horizon10s) | 0.9671 ± 0.3072 | **+153.9%** |
| FDE_2s (horizon20s) | 2.0083 ± 0.5027 | **+124.7%** |
| FDE_3s (horizon30s) | 3.4185 ± 0.8526 | **+98.1%** |
| FDE_4s (horizon40s) | 5.2738 ± 1.3442 | **+74.6%** |

#### MANTRA Performance Analysis

**Digital Performance (Single Run):**

| Metric | Value (m) |
|--------|-----------|
| eucl_mean | 1.1610 |
| ADE_1s | 0.2110 |
| ADE_2s | 0.4380 |
| ADE_3s | 0.7380 |
| FDE_1s (horizon10s) | 0.3820 |
| FDE_2s (horizon20s) | 0.9190 |
| FDE_3s (horizon30s) | 1.7360 |
| FDE_4s (horizon40s) | 3.0970 |

**Analog Performance (Mean ± Std over 10 runs):**

| Metric | Value (m) | Relative Increase |
|--------|-----------|-------------------|
| eucl_mean | 1.7954 ± 0.5465 | **+54.7%** |
| ADE_1s | 0.4400 ± 0.1427 | **+108.5%** |
| ADE_2s | 0.8067 ± 0.2558 | **+84.2%** |
| ADE_3s | 1.2387 ± 0.3935 | **+67.8%** |
| FDE_1s (horizon10s) | 0.7519 ± 0.2531 | **+96.8%** |
| FDE_2s (horizon20s) | 1.5464 ± 0.5107 | **+68.3%** |
| FDE_3s (horizon30s) | 2.6076 ± 0.8493 | **+50.2%** |
| FDE_4s (horizon40s) | 4.3017 ± 1.1631 | **+38.9%** |

#### Key Findings

**Error Propagation Patterns:**
- **Short-term predictions (1s):** Both models show 100-150% error increases in analog mode
- **Long-term predictions (4s):** Error increases moderate to 39-75%, suggesting error saturation
- **Variance analysis:** High standard deviations (±20-30% of mean) indicate significant device stochasticity

**Model Comparison:**
- **MANTRA robustness:** MANTRA shows better analog resilience with 39-109% degradation vs CVAE's 75-154%
- **Memory architecture advantage:** MANTRA's memory-augmented design may be more robust to analog noise
- **CVAE sensitivity:** The fully generative CVAE architecture appears more sensitive to weight quantization errors

**Critical Observations:**
1. **Analog degradation:** Both models experience substantial accuracy loss due to SONOS device non-idealities
2. **Compounding effects:** Errors accumulate through recurrent decoder layers, severely impacting long-horizon predictions
3. **Stochastic variability:** Large standard deviations indicate unreliable predictions across different device realizations
4. **Horizon dependency:** Relative error increase is most severe at short horizons where digital models are most accurate

---

## 10. Conclusions

This project demonstrates:

1. **Effective CVAE architecture:** Successfully models multimodal trajectory prediction with competitive performance
2. **Analog computing feasibility:** Shows that trajectory prediction can be deployed on analog hardware with quantifiable accuracy trade-offs
3. **Comprehensive evaluation framework:** Provides tools for comparing digital and analog implementations with proper statistical analysis
4. **Hardware-aware design:** Highlights the importance of considering device non-idealities in neural network deployment

Future work could explore:
- Analog-aware training techniques to improve robustness
- Energy efficiency measurements comparing digital vs. analog inference
- Architectural modifications to minimize analog error propagation

---

## References

**MANTRA paper:**
- Marchetti, F., Becattini, F., Seidenari, L., & Del Bimbo, A. (2021). MANTRA: Memory augmented networks for multiple trajectory prediction. arXiv. https://arxiv.org/abs/2006.03340

**CrossSim:**
- Analog in-memory computing simulation framework for evaluating neural networks on resistive crossbar arrays
- https://cross-sim.sandia.gov/

**Dataset:**
- KITTI Vision Benchmark Suite for autonomous driving scenarios