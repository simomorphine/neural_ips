# ComplexIPS: Complex-Valued Information Processing Systems in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

**ComplexIPS** is a PyTorch implementation of complex-valued neural networks grounded in the **Information Processing System (IPS)** framework. It formalizes neurons as quantum‑inspired systems that maintain a complex state, and learn to balance goal achievement against computational cost under an energy budget. The layer supports both standard autograd training and a **local, bio‑plausible learning rule** based on Wirtinger calculus, with an optional asymmetry factor that introduces an “arrow of time”.

This repository accompanies the theoretical work presented in:

- *Beyond Energy: A New Geometry for Information Processing Systems*  
- *Equilibrium in Asymmetric State Spaces*  
- *Humble Systems Theory: A Framework in Permanent Superposition Between Cosmic Joke and Universal Truth*

---

## Table of Contents
- [Overview](#overview)
- [Theory at a Glance](#theory-at-a-glance)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional neural networks operate on real numbers and rely on global backpropagation. **ComplexIPS** takes a different perspective:

- Each neuron (or layer) is an **Information Processing System (IPS)** with its own state \(|\psi\rangle \in \mathbb{C}^d\).
- The system has three intrinsic operators:
  - **Goal operator \(\hat{G}\)** – what it wants to achieve.
  - **Cost operator \(\hat{C}\)** – the computational cost of a state.
  - **Energy operator \(\hat{E}\)** – the energy consumption (here taken as the squared norm).
- Learning minimizes \(\langle\psi|\hat{C}|\psi\rangle\) while maximizing \(\langle\psi|\hat{G}|\psi\rangle\), subject to \(\langle\psi|\hat{E}|\psi\rangle \le E_{\text{budget}}\).
- Parameters are complex numbers (stored as separate real/imaginary parts), and activations preserve phase information.

The framework naturally gives rise to a **complex quasi‑metric** \(Q = d + i\cdot\text{debt}\) and a one‑parameter family of \(\gamma\)-distances, where \(\gamma\) controls the trade‑off between energy cost and informational debt. The code includes an optional **asymmetric update** that damps the imaginary component, modelling irreversibility and the arrow of time.

---

## Theory at a Glance

### The Complex Neuron
The neuron computes:
\[
|\psi\rangle = f\big( (W_r + i W_i) \mathbf{x} + (b_r + i b_i) \big)
\]
where \(f\) is a phase‑preserving activation (here \(\tanh\) on the magnitude). The output is a complex vector stacked as `(real, imag)`.

### IPS Operators
- **Goal operator \(\hat{G}\)**: a symmetric (Hermitian) matrix learned via a low‑rank parameterization \(G = L L^\top\). Its expectation \(\langle\psi|\hat{G}|\psi\rangle\) measures goal fulfilment.
- **Cost operator \(\hat{C}\)**: similarly parameterized, its expectation \(\langle\psi|\hat{C}|\psi\rangle\) is the primary cost to minimise.
- **Energy \(\langle\psi|\hat{E}|\psi\rangle\)**: taken as the squared norm \(\|\psi\|^2\).

### Loss Function (Autograd Mode)
\[
\mathcal{L} = \langle\hat{C}\rangle - \lambda\langle\hat{G}\rangle + \mu\max(0,\langle\hat{E}\rangle - E_{\text{budget}}) + \alpha\|\nabla L\|^2 + \gamma N_{\text{ops}}
\]
The last two terms are proxies for the geodesic penalty and operation count as described in the papers.

### Local Learning Rule (Wirtinger Update)
Without backprop, the weights can be updated locally using:
\[
\Delta W = \eta \, (\text{target} - \text{prediction}) \cdot \overline{\mathbf{x}}^{\top}
\]
with all quantities complex. This is a complex‑valued Hebbian‑like rule that uses only pre‑ and post‑synaptic information.

### Asymmetry and the Arrow of Time
By damping the imaginary part of the update (controlled by a parameter \(\gamma\)), the learning dynamics become irreversible – a mathematical embodiment of humility and the second law.

---

## Features

- ✅ **Complex‑valued linear layer** with separate real/imag parameters.
- ✅ **Phase‑preserving activation** (magnitude scaled by tanh).
- ✅ **IPS‑inspired loss** that balances goal, cost, energy, and operation count.
- ✅ **Hermitian operator parameterization** via \(L L^\top\).
- ✅ **Local Wirtinger update** for bio‑plausible, non‑backprop learning.
- ✅ **Adaptive learning rate** option for local updates.
- ✅ **Asymmetric update** (imaginary damping) to model irreversibility.
- ✅ **Quantum‑inspired diagnostics**: purity, phase coherence, entanglement proxy.
- ✅ Fully compatible with PyTorch’s autograd – can be used in standard pipelines or standalone.

