# Multi-Port Solid State Devices
This work was supported by:
How to cite: 


## Description (AI-generated translation of code into latex equations.)
# MPSSD Model Equations (LinDistFlow)

## Sets and Indices

| Symbol | Meaning | Code |
|---|---|---|
| $\mathcal{D}$ | MPSSD devices | `mpssd_device_set` |
| $\Phi_d \subseteq \{a,b,c\}$ | Phases of device $d$, with first phase $\phi_d^{1}$ | `mpssd_phases_by_device[d]` |
| $\mathcal{P} = \{(d,\phi) : d\in\mathcal{D},\ \phi\in\Phi_d\}$ | Device-phase ports | `mpssd_device_phase_set` |
| $\mathcal{K}$ | DC-bus labels | `mpssd_dc_bus_set` |
| $\mathcal{P}_k \subseteq \mathcal{P}$ | Ports attached to DC bus $k$ | `mpssd_ports_by_dc_bus[k]` |
| $\mathcal{T}$ | Time periods | `time_set` |
| $\beta(d)$ | AC bus of device $d$ | `mpssd_bus_by_device[d]` |
| $\mathcal{D}_{i,\phi} = \{d : \beta(d)=i,\ \phi\in\Phi_d\}$ | Devices on bus $i$, phase $\phi$ | `mpssd_devices_by_bus_phase` |

The control mode of each port is $c_{d,\phi} \in \{\text{NONE},\,\text{P},\,\text{Q},\,\text{PQ}\}$.

The phase-balance flag of each device is $\sigma_d\in\{0,1\}$.

## Parameters

The apparent-power rating is $\overline{S}_{d,\phi} > 0$.

The reactive-power limits are clamped to the rating:

$$
\underline{Q}_{d,\phi} = \max\!\left(-\overline{S}_{d,\phi},\ q^{\min,\text{input}}_{d,\phi}\right),
\qquad
\overline{Q}_{d,\phi} = \min\!\left(\overline{S}_{d,\phi},\ q^{\max,\text{input}}_{d,\phi}\right)
$$

If no input limit is given, the default is $q^{\min,\text{input}}_{d,\phi}=-\overline{S}_{d,\phi}$ and $q^{\max,\text{input}}_{d,\phi}=\overline{S}_{d,\phi}$.

The setpoints are $p^{\text{set}}_{d,\phi,t}$ and $q^{\text{set}}_{d,\phi,t}$. They are currently constant in $t$.

## Decision Variables

$$
p_{d,\phi,t},\; q_{d,\phi,t} \in \mathbb{R}, \qquad \forall (d,\phi)\in\mathcal{P},\ t\in\mathcal{T}
$$

Positive values denote injection into the AC bus.

## Nodal Power Injections

$$
\begin{aligned}
P^{\text{mpssd}}_{i,\phi,t} &= \sum_{d \in \mathcal{D}_{i,\phi}} p_{d,\phi,t} \\
Q^{\text{mpssd}}_{i,\phi,t} &= \sum_{d \in \mathcal{D}_{i,\phi}} q_{d,\phi,t}
\end{aligned}
$$

## Constraints

### 1. Fixed active power (modes NONE and Q)

$$
p_{d,\phi,t} = p^{\text{set}}_{d,\phi,t}
\qquad \forall (d,\phi)\in\mathcal{P}:\ c_{d,\phi}\in\{\text{NONE},\text{Q}\},\ \forall t\in\mathcal{T}
$$

### 2. Fixed reactive power (modes NONE and P)

$$
q_{d,\phi,t} = q^{\text{set}}_{d,\phi,t}
\qquad \forall (d,\phi)\in\mathcal{P}:\ c_{d,\phi}\in\{\text{NONE},\text{P}\},\ \forall t\in\mathcal{T}
$$

### 3. Rectangular operating limits (omitted if `equality_only`)

$$
\begin{aligned}
-\overline{S}_{d,\phi} \;\le\; &p_{d,\phi,t} \;\le\; \overline{S}_{d,\phi} \\
\underline{Q}_{d,\phi} \;\le\; &q_{d,\phi,t} \;\le\; \overline{Q}_{d,\phi}
\end{aligned}
\qquad \forall (d,\phi)\in\mathcal{P},\ \forall t\in\mathcal{T}
$$

### 4a. Circular apparent-power limit (if `circular_constraints`, the default)

$$
p_{d,\phi,t}^2 + q_{d,\phi,t}^2 \;\le\; \overline{S}_{d,\phi}^{\,2}
\qquad \forall (d,\phi)\in\mathcal{P}:\ c_{d,\phi}=\text{PQ},\ \forall t\in\mathcal{T}
$$

### 4b. Linear $n$-gon approximation (otherwise, $n=8$)

$$
a_j\, p_{d,\phi,t} + b_j\, q_{d,\phi,t} \;\le\; \overline{S}_{d,\phi}
\qquad j = 1,\dots,n,\quad \forall (d,\phi)\in\mathcal{P}:\ c_{d,\phi}=\text{PQ},\ \forall t\in\mathcal{T}
$$

The coefficient pairs $(a_j, b_j)$ are the edge coefficients returned by `ngon_line_equations(n)`. Each pair defines one edge of a regular $n$-gon that approximates the unit circle.

### 5. DC-bus active-power balance (lossless DC link)

$$
\sum_{(d,\phi)\in\mathcal{P}_k} p_{d,\phi,t} = 0
\qquad \forall k\in\mathcal{K},\ \forall t\in\mathcal{T}
$$

### 6. Phase balance (devices with $\sigma_d = 1$)

$$
\begin{aligned}
p_{d,\phi,t} &= p_{d,\phi_d^{1},t} \\
q_{d,\phi,t} &= q_{d,\phi_d^{1},t}
\end{aligned}
\qquad \forall d\in\mathcal{D}:\ \sigma_d=1,\ \forall \phi\in\Phi_d\setminus\{\phi_d^{1}\},\ \forall t\in\mathcal{T}
$$

---

## Pre-solve Feasibility Checks (from `validate_mpssd_data`)

In these checks, $\varepsilon$ is the tolerance `TOL`.

**Per-port setpoint checks**, applied when the corresponding quantity is fixed:

$$
|p^{\text{set}}_{d,\phi}| \le \overline{S}_{d,\phi}+\varepsilon,
\qquad
\underline{Q}_{d,\phi}-\varepsilon \le q^{\text{set}}_{d,\phi} \le \overline{Q}_{d,\phi}+\varepsilon,
\qquad
\underline{Q}_{d,\phi} \le \overline{Q}_{d,\phi}
$$

If both P and Q are fixed (mode NONE), the setpoints must also fit inside the rating circle:

$$
\left(p^{\text{set}}_{d,\phi}\right)^2 + \left(q^{\text{set}}_{d,\phi}\right)^2 \le \overline{S}_{d,\phi}^{\,2}+\varepsilon
$$

**Balanced devices with fixed setpoints:**

$$
\max_{\phi\in\Phi_d} p^{\text{set}}_{d,\phi} - \min_{\phi\in\Phi_d} p^{\text{set}}_{d,\phi} \le \varepsilon
\quad\text{(and likewise for } q^{\text{set}}\text{)}
$$

**Achievable active-power range of each device.** Let $n_d = |\Phi_d|$. Then

$$
[\underline{P}_d,\overline{P}_d] =
\begin{cases}
\left[\sum_{\phi\in\Phi_d} p^{\text{set}}_{d,\phi},\ \sum_{\phi\in\Phi_d} p^{\text{set}}_{d,\phi}\right] & c_d\in\{\text{NONE},\text{Q}\}\\[6pt]
\left[-n_d \min_{\phi\in\Phi_d}\overline{S}_{d,\phi},\ \ n_d \min_{\phi\in\Phi_d}\overline{S}_{d,\phi}\right] & \sigma_d=1\\[6pt]
\left[-\sum_{\phi\in\Phi_d}\overline{S}_{d,\phi},\ \ \sum_{\phi\in\Phi_d}\overline{S}_{d,\phi}\right] & \text{otherwise}
\end{cases}
$$

**DC-bus feasibility.** Let $\mathcal{D}_k$ be the set of devices whose ports connect to DC bus $k$. The case is declared infeasible unless

$$
\sum_{d\in\mathcal{D}_k}\underline{P}_d \le \varepsilon
\quad\text{and}\quad
\sum_{d\in\mathcal{D}_k}\overline{P}_d \ge -\varepsilon
\qquad \forall k\in\mathcal{K}
$$

This check uses only rows that passed all other validation.