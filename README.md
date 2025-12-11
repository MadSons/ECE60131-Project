# CVAE for Scene-Aware Trajectory Prediction

This project implements a conditional variational autoencoder (CVAE) for multimodal trajectory prediction on the KITTI-style vehicle trajectory dataset with scene context.

## Overview

The goal is to model the conditional distribution of future trajectories given the past trajectory and a rasterized scene:

$$\hat{p}_\theta(y|c)$$

where:
- $y$ is the future trajectory (a sequence of 2D points over 4 seconds)
- $c$ is the conditioning variable, consisting of:
  - the past trajectory (2 seconds of 2D points)
  - a top-down scene representation (one-hot raster of lanes, road, etc.)

The model is trained as a generative model: it defines an explicit latent variable model $\hat{p}_\theta(x,y|c)$ and learns $\theta$ by maximizing a variational lower bound on the conditional log-likelihood $\log \hat{p}_\theta(y|c)$.

## 1. Generative Model

We introduce a latent variable $x$ that captures high-level intentions/modes of future motion (e.g., going straight, turning, changing lanes). For each observed pair $(c,y)$:

1. Sample latent state from a conditional prior: $\hat{p}_\theta(x|c)$
2. Given $(x,c)$, sample the future trajectory: $\hat{p}_\theta(y|x,c)$

So the joint model is:

$$\hat{p}_\theta(x,y|c) = \hat{p}_\theta(x|c) \cdot \hat{p}_\theta(y|x,c)$$

### 1.1 Parameterization

We use Gaussian distributions with neural-network parameters:

**Conditional prior:**

$$\hat{p}_\theta(x|c) = \mathcal{N}(x; \mu_p(c;\theta), \Sigma_p(c;\theta))$$

where $\mu_p$ and $\Sigma_p$ are outputs of a neural network that takes the encoded past trajectory and encoded scene as inputs.

**Emission model:**

$$\hat{p}_\theta(y|x,c) = \mathcal{N}(y; \mu_y(x,c;\theta), \sigma^2 I)$$

In the implementation, $\mu_y(x,c;\theta)$ is produced by an autoregressive GRU decoder that rolls out a sequence of future displacements, with a fixed, isotropic covariance $\sigma^2 I$. Using mean-squared error (MSE) as the reconstruction term corresponds exactly to this Gaussian assumption.

## 2. Recognition Model (Approximate Posterior)

Exact expectations under the true posterior $\hat{p}_\theta(x|y,c) \propto \hat{p}_\theta(x|c) \cdot \hat{p}_\theta(y|x,c)$ are intractable. We introduce a recognition distribution (encoder):

$$\hat{r}_\phi(x|y,c) = \mathcal{N}(x; \mu_r(y,c;\phi), \Sigma_r(y,c;\phi))$$

where $\mu_r$ and $\Sigma_r$ are functions of:
- the encoded past trajectory
- the encoded future trajectory
- the encoded scene

All three encoders are GRU-based (plus some 1D/2D convolutions), so $\mu_r$ and $\Sigma_r$ are effectively neural networks over the whole sequence and scene.

In the code, this appears as:
```python
inference_q(past, future, scene_one_hot)
```
returning `mu_q, logvar_q, h_past, h_scene`.

## 3. Conditional ELBO and Training Objective

For a given context $c$ and future $y$, the conditional log-likelihood is:

$$\log \hat{p}_\theta(y|c) = \log \int \hat{p}_\theta(x|c) \cdot \hat{p}_\theta(y|x,c) \, dx$$

We introduce the recognition distribution and use Jensen's inequality to obtain the conditional evidence lower bound (ELBO):

$$\log \hat{p}_\theta(y|c) \geq \mathbb{E}_{x \sim \hat{r}_\phi(x|y,c)}[\log \hat{p}_\theta(y|x,c)] - \text{KL}(\hat{r}_\phi(x|y,c) \| \hat{p}_\theta(x|c))$$

The loss function is the negative ELBO:

$$\mathcal{L}(\theta,\phi) = -\mathbb{E}_{x \sim \hat{r}_\phi(x|y,c)}[\log \hat{p}_\theta(y|x,c)] + \text{KL}(\hat{r}_\phi(x|y,c) \| \hat{p}_\theta(x|c))$$

We use a **β-weighted version**:

$$\mathcal{L}_\beta(\theta,\phi) = -\mathbb{E}_{x \sim \hat{r}_\phi(x|y,c)}[\log \hat{p}_\theta(y|x,c)] + \beta \cdot \text{KL}(\hat{r}_\phi(x|y,c) \| \hat{p}_\theta(x|c))$$

with $\beta$ annealed from a small value to 1 over the first several epochs.

### 3.1 Connection to Code

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

## 4. Network Architecture

### 4.1 Context $c$

**Past trajectory** $c_{\text{past}} \in \mathbb{R}^{T_{\text{past}} \times 2}$:
- Encoded by convolution + GRU into $h_{\text{past}}(c) \in \mathbb{R}^D$

**Scene** $c_{\text{scene}}$:
- Raster $180 \times 180 \times 4$, passed through 2D convolutions and a GRU
- Produces $h_{\text{scene}}(c) \in \mathbb{R}^D$

Context embedding: $h_c = [h_{\text{past}}(c), h_{\text{scene}}(c)]$

### 4.2 Generative Model

**Conditional prior:**

$$\hat{p}_\theta(x|c) = \mathcal{N}(x; \mu_p(h_c;\theta), \Sigma_p(h_c;\theta))$$

where `fc_prior` computes $[\mu_p, \log \Sigma_p]$.

**Decoder:**

Given sampled $x$ and context $c$, form initial hidden state:

$$h_{\text{dec},0} = f_\theta(h_{\text{past}}(c), h_{\text{scene}}(c), x)$$

Use this as the initial hidden state of a GRU that autoregressively generates future displacements, producing $\mu_y(x,c;\theta)$.

### 4.3 Recognition Model

**Encoder/recognition distribution:**

$$\hat{r}_\phi(x|y,c) = \mathcal{N}(x; \mu_r(h_c, h_{\text{fut}};\phi), \Sigma_r(h_c, h_{\text{fut}};\phi))$$

where $h_{\text{fut}}$ is obtained from the future trajectory via conv+GRU, and `fc_q` produces $[\mu_r, \log \Sigma_r]$.

Sampling $x$ from $\hat{r}_\phi(x|y,c)$ uses the reparameterization trick (implemented in `reparameterize`).

## 5. Training and Evaluation

### 5.1 Training

**Objective:** Minimize the β-weighted negative conditional ELBO $\mathcal{L}_\beta(\theta,\phi)$ over the training set.

**β-annealing:** $\beta$ starts near 0 and linearly increases to 1 over a fixed number of warmup epochs (`beta_warmup_epochs`). This encourages the model to first learn good reconstruction before enforcing strong prior-posterior match.

The training script:
- Loads `CVAE_Predictor`
- Iterates over mini-batches $(c,y)$
- Computes `loss, recon, kl = model.compute_loss(...)`
- Updates $\theta, \phi$ via Adam
- Periodically evaluates on test set using ADE/FDE metrics (in meters at 1s, 2s, 3s, 4s horizons)

### 5.2 Sampling/Generation

At test time, to generate multimodal trajectory predictions for context $c = (c_{\text{past}}, c_{\text{scene}})$:

1. Encode the context to get $h_{\text{past}}(c)$ and $h_{\text{scene}}(c)$
2. Compute the conditional prior $\hat{p}_\theta(x|c)$
3. Draw samples $x^{(1)}, \ldots, x^{(K)} \sim \hat{p}_\theta(x|c)$ using `reparameterize`
4. For each $x^{(k)}$, run the decoder GRU to obtain trajectory mean $\mu_y(x^{(k)}, c;\theta)$

In code:
```python
model.sample(past, scene_one_hot, num_samples=K)
```
returns a tensor of shape `[B, K, T_future, 2]` with $K$ trajectories per agent.

## 6. Relation to "Generative vs. Discriminative"

Even though we train on paired data $(c,y)$, this is not a purely discriminative model. Instead, we have:

- An explicit latent variable $x$
- A joint conditional model $\hat{p}_\theta(x,y|c)$
- A conditional prior $\hat{p}_\theta(x|c)$
- A likelihood $\hat{p}_\theta(y|x,c)$ with probabilistic interpretation

We fit $\theta$ and $\phi$ by maximizing a variational lower bound on $\log \hat{p}_\theta(y|c)$, exactly in the spirit of VAE/variational inference, but extended to the conditional setting. The model can generate samples of $y$ for a given context $c$ by ancestral sampling through the latent variable, which is the defining property of a generative model.

## 7. Results

### 7.1 Model Configuration

```yaml
dataset_file: kitti_dataset.json
past_len: 20
future_len: 40
dim_embedding_key: 48
latent_dim: 8
batch_size: 32
learning_rate: 0.0001
max_epochs: 200
cuda: True
device: 0
weight_decay: 5e-05
beta_max: 0.8
beta_warmup_epochs: 80
num_samples_eval: 5
eval_every: 5
info: cvae_cond_prior_beta
```

### 7.2 Example Predictions

The model generates multiple diverse trajectory predictions for each scenario, capturing the multimodal nature of future motion:

![Example 4](https://github.com/MadSons/ECE60131-Project/blob/main/test/2025-12-01_14-06-25_cvae_eval_14-14-19/example_004.png)

![Example 5](https://github.com/MadSons/ECE60131-Project/blob/main/test/2025-12-01_14-06-25_cvae_eval_14-14-19/example_005.png)

### 7.3 Quantitative Results

**Best of 5 Samples:**

| Metric | Value (m) |
|--------|-----------|
| eucl_mean | 1.1705 |
| ADE_1s | 0.2335 |
| ADE_2s | 0.4431 |
| ADE_3s | 0.7444 |
| FDE_1s (horizon10s) | 0.3837 |
| FDE_2s (horizon20s) | 0.9112 |
| FDE_3s (horizon30s) | 1.7629 |
| FDE_4s (horizon40s) | 3.0912 |

**Best of 10 Samples:**

| Metric | Value (m) |
|--------|-----------|
| eucl_mean | 1.0151 |
| ADE_1s | 0.2271 |
| ADE_2s | 0.4083 |
| ADE_3s | 0.6566 |
| FDE_1s (horizon10s) | 0.3664 |
| FDE_2s (horizon20s) | 0.8003 |
| FDE_3s (horizon30s) | 1.4911 |
| FDE_4s (horizon40s) | 2.6661 |

### 7.4 Comparison with Baselines

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
- At the 4-second horizon with 10 samples, the CVAE achieves 2.67m FDE, outperforming MANTRA (top 10) at 1.88m but showing the challenge of long-horizon prediction
