# ADMM Mathematical Formulation Report

**Date:** 2026-08-21  
**Scope:** Mathematical translation of [`admm_agents.py`](src/distopf/distributed/spatial/admm_agents.py) and the directly related ADMM penalty functions.

## Summary

This report translates the pairwise, scaled-ADMM coordination implemented by [`solve_admm()`](src/distopf/distributed/spatial/admm_agents.py:520) into mathematical notation. Each decomposed area solves a local OPF with quadratic penalties, exchanges boundary voltage and complex-power values with neighboring areas, averages the two interface copies, updates a scaled dual variable, and writes the shifted consensus target into the next local solve.

The most important cross-checks are:

1. The consensus update averages voltage magnitudes and complex powers directly.
2. The local voltage penalty is applied to squared voltage magnitude, i.e. $(V^2-\widehat V^2)^2$.
3. The local complex-power penalty is Euclidean distance in $(P,Q)$.
4. The implementation uses a primal consensus residual only; it does not compute a standard dual residual.
5. Dual variables are stored locally and directionally for each interface.

## 1. Notation

Let the decomposed network consist of areas indexed by $i\in\mathcal A$. Let $\mathcal C_i$ be the set of child areas of area $i$, and let $p(i)$ denote the parent of area $i$ when it exists.

Let
$$
\alpha=(\phi,t),
\qquad \phi\in\{a,b,c\},
$$
where $\phi$ is a phase and $t$ is a time point. Unless explicitly stated otherwise, sums over $\alpha$ include all phases and time points.

The boundary variables described in [`BoundaryVars`](src/distopf/distributed/spatial/enapp_agents.py:88) are:

- $v_i^{\uparrow}$: voltage at area $i$'s upstream/swing boundary bus.
- $s_i^{\uparrow}$: complex power flowing from area $i$ toward its parent.
- $v_{ij}^{\downarrow}$: voltage at area $i$'s boundary bus associated with child $j$.
- $s_{ij}^{\downarrow}$: complex power at area $i$'s boundary bus associated with child $j$.

Complex power is represented phasewise as
$$
s=p+\mathrm{i}q,
$$
where $p$ is active power and $q$ is reactive power.

The boundary extraction functions are [`parse_s_up()`](src/distopf/distributed/spatial/enapp_agents.py:184), [`parse_v_up()`](src/distopf/distributed/spatial/enapp_agents.py:171), [`parse_s_dn()`](src/distopf/distributed/spatial/enapp_agents.py:229), and [`parse_v_dn()`](src/distopf/distributed/spatial/enapp_agents.py:213).

## 2. Interface consensus constraints

For a child area $i$ and its parent $p(i)$, the two copies of an upstream boundary variable are:

| Quantity | Child-side copy | Parent-side copy |
|---|---|---|
| Voltage | $v_i^{\uparrow}$ | $v_{p(i),i}^{\downarrow}$ |
| Complex power | $s_i^{\uparrow}$ | $s_{p(i),i}^{\downarrow}$ |

The desired consensus constraints are
$$
v_i^{\uparrow}=v_{p(i),i}^{\downarrow},
\qquad
s_i^{\uparrow}=s_{p(i),i}^{\downarrow}.
$$

Equivalently, for a parent area $i$ and child $j$,
$$
v_{ij}^{\downarrow}=v_j^{\uparrow},
\qquad
s_{ij}^{\downarrow}=s_j^{\uparrow}.
$$

The implementation constructs these local/remote pairs in [`_interface_pairs()`](src/distopf/distributed/spatial/admm_agents.py:152).

For a generic interface $e$, write the two current primal copies as
$$
x_{e,\mathrm{loc}}^{(k)},
\qquad
x_{e,\mathrm{rem}}^{(k)},
$$
where $x$ may be a voltage or a complex power component.

## 3. Local OPF subproblem

At iteration $k$, area $i$ solves a local OPF problem of the abstract form
$$
\begin{aligned}
\min_{x_i}\quad & f_i(x_i)+\Phi_i^{(k)}(x_i),\\
\text{s.t.}\quad & x_i\in\mathcal X_i,
\end{aligned}
$$
where:

- $x_i$ contains the local OPF variables;
- $f_i$ is the selected primary objective;
- $\mathcal X_i$ contains the local physical and operational constraints;
- $\Phi_i^{(k)}$ is the sum of the ADMM quadratic penalties.

The local solves are coordinated by [`_solve_iteration()`](src/distopf/distributed/spatial/enapp_agents.py:660) and called from [`solve_admm()`](src/distopf/distributed/spatial/admm_agents.py:574).

The ADMM driver forces
$$
\texttt{free\_swing\_voltage}=\texttt{True},
\qquad
\texttt{free\_boundary\_loads}=\texttt{True},
$$
through [`solve_admm()`](src/distopf/distributed/spatial/admm_agents.py:552), allowing boundary variables to move toward consensus during the local optimization.

The total local penalty is
$$
\Phi_i^{(k)}
=
\Phi_{i,v^{\uparrow}}^{(k)}
+
\Phi_{i,s^{\uparrow}}^{(k)}
+
\Phi_{i,v^{\downarrow}}^{(k)}
+
\Phi_{i,s^{\downarrow}}^{(k)}.
$$

The corresponding penalty parameters are
$$
\rho_{v^{\uparrow}},
\quad
\rho_{s^{\uparrow}},
\quad
\rho_{v^{\downarrow}},
\quad
\rho_{s^{\downarrow}}.
$$

They are passed by [`solve_admm()`](src/distopf/distributed/spatial/admm_agents.py:524) and added to the local objective by [`pyomo_wrapper.py`](src/distopf/wrappers/pyomo_wrapper.py:168).

## 4. ADMM penalty equations

### 4.1 Upstream-voltage penalty

The function [`admm_v_up_penalty()`](src/distopf/pyomo_models/objectives.py:540) compares the local squared voltage variable $V_{i,\alpha}^2$ with the square of the schedule target $\widehat V_{i,\alpha}^{\uparrow,(k)}$:

$$
\Phi_{i,v^{\uparrow}}^{(k)}
=
\frac{\rho_{v^{\uparrow}}}{2}
\sum_{\alpha}
\left(
V_{i,\alpha}^{2}
-
\left(\widehat V_{i,\alpha}^{\uparrow,(k)}\right)^2
\right)^2.
$$

This is not the same as a penalty on voltage magnitude itself,

$$
\frac{\rho_{v^{\uparrow}}}{2}
\sum_{\alpha}
\left(V_{i,\alpha}-\widehat V_{i,\alpha}^{\uparrow,(k)}\right)^2.
$$

The distinction matters when comparing the code with a reference formulation.

### 4.2 Upstream-power penalty

For an upstream boundary, let $\mathcal L_i^{\uparrow}$ be the set of local branches leaving the upstream/swing bus. The total local upstream active and reactive powers are
$$
P_{i,\alpha}^{\uparrow}
=
\sum_{\ell\in\mathcal L_i^{\uparrow}}P_{\ell,\alpha},
\qquad
Q_{i,\alpha}^{\uparrow}
=
\sum_{\ell\in\mathcal L_i^{\uparrow}}Q_{\ell,\alpha}.
$$

The function [`admm_s_up_penalty()`](src/distopf/pyomo_models/objectives.py:558) applies

$$
\Phi_{i,s^{\uparrow}}^{(k)}
=
\frac{\rho_{s^{\uparrow}}}{2}
\sum_{\alpha}
\left[
\left(P_{i,\alpha}^{\uparrow}-\widehat P_{i,\alpha}^{\uparrow,(k)}\right)^2
+
\left(Q_{i,\alpha}^{\uparrow}-\widehat Q_{i,\alpha}^{\uparrow,(k)}\right)^2
\right].
$$

With
$$
S_{i,\alpha}^{\uparrow}
=P_{i,\alpha}^{\uparrow}+\mathrm{i}Q_{i,\alpha}^{\uparrow},
\qquad
\widehat S_{i,\alpha}^{\uparrow,(k)}
=\widehat P_{i,\alpha}^{\uparrow,(k)}
+\mathrm{i}\widehat Q_{i,\alpha}^{\uparrow,(k)},
$$
this is equivalently

$$
\Phi_{i,s^{\uparrow}}^{(k)}
=
\frac{\rho_{s^{\uparrow}}}{2}
\sum_{\alpha}
\left|
S_{i,\alpha}^{\uparrow}
-
\widehat S_{i,\alpha}^{\uparrow,(k)}
\right|^2.
$$

### 4.3 Downstream-voltage penalty

For each child $j\in\mathcal C_i$, the function [`admm_v_down_penalty()`](src/distopf/pyomo_models/objectives.py:589) applies

$$
\Phi_{i,v^{\downarrow}}^{(k)}
=
\frac{\rho_{v^{\downarrow}}}{2}
\sum_{j\in\mathcal C_i}
\sum_{\alpha}
\left(
V_{ij,\alpha}^{2}
-
\left(\widehat V_{ij,\alpha}^{\downarrow,(k)}\right)^2
\right)^2.
$$

Here $V_{ij,\alpha}^2$ is the local squared-voltage variable at the child-boundary bus, and $\widehat V_{ij,\alpha}^{\downarrow,(k)}$ is the voltage target supplied through the schedule.

### 4.4 Downstream-power penalty

For each child boundary, the function [`admm_s_down_penalty()`](src/distopf/pyomo_models/objectives.py:610) applies

$$
\Phi_{i,s^{\downarrow}}^{(k)}
=
\frac{\rho_{s^{\downarrow}}}{2}
\sum_{j\in\mathcal C_i}
\sum_{\alpha}
\left[
\left(P_{ij,\alpha}^{\downarrow}
-
\widehat P_{ij,\alpha}^{\downarrow,(k)}\right)^2
+
\left(Q_{ij,\alpha}^{\downarrow}
-
\widehat Q_{ij,\alpha}^{\downarrow,(k)}\right)^2
\right].
$$

Equivalently, using complex power,
$$
S_{ij,\alpha}^{\downarrow}
=P_{ij,\alpha}^{\downarrow}+\mathrm{i}Q_{ij,\alpha}^{\downarrow},
$$

$$
\Phi_{i,s^{\downarrow}}^{(k)}
=
\frac{\rho_{s^{\downarrow}}}{2}
\sum_{j\in\mathcal C_i}
\sum_{\alpha}
\left|
S_{ij,\alpha}^{\downarrow}
-
\widehat S_{ij,\alpha}^{\downarrow,(k)}
\right|^2.
$$

## 5. Boundary message exchange

After a local solve, [`set_result()`](src/distopf/distributed/spatial/enapp_agents.py:403) extracts the boundary values. The resulting messages are sent by [`send_all_agent_messages()`](src/distopf/distributed/spatial/enapp_agents.py:579).

For an interface $e$, denote the two received values at iteration $k$ by
$$
x_{e,\mathrm{loc}}^{(k)},
\qquad
x_{e,\mathrm{rem}}^{(k)}.
$$

The local/remote assignment is directional:

- For a child area, the local value is $v^{\uparrow}$ or $s^{\uparrow}$ and the remote value is the parent's $v^{\downarrow}$ or $s^{\downarrow}$.
- For a parent area, the local value is $v^{\downarrow}$ or $s^{\downarrow}$ and the remote value is the child's $v^{\uparrow}$ or $s^{\uparrow}$.

This pairing is implemented by [`_interface_pairs()`](src/distopf/distributed/spatial/admm_agents.py:152).

## 6. Primal consensus residual

The function [`_frame_residual()`](src/distopf/distributed/spatial/admm_agents.py:297) computes the maximum absolute mismatch over all phases and time points:

$$
r_e^{(k)}
=
\left\|
 x_{e,\mathrm{loc}}^{(k)}
-
 x_{e,\mathrm{rem}}^{(k)}
\right\|_{\infty}.
$$

For a voltage interface this means

$$
r_{e,v}^{(k)}
=
\max_{\phi,t}
\left|
V_{e,\mathrm{loc},\phi,t}^{(k)}
-
V_{e,\mathrm{rem},\phi,t}^{(k)}
\right|.
$$

For a complex-power interface it means

$$
r_{e,s}^{(k)}
=
\max_{\phi,t}
\left|
S_{e,\mathrm{loc},\phi,t}^{(k)}
-
S_{e,\mathrm{rem},\phi,t}^{(k)}
\right|.
$$

The absolute value for complex power is the complex modulus. Residual histories are recorded by [`_record_residual()`](src/distopf/distributed/spatial/admm_agents.py:326).

The global convergence metric is

$$
r^{(k)}
=
\max_{e}r_e^{(k)},
$$
which is returned by [`_global_primal_residual()`](src/distopf/distributed/spatial/admm_agents.py:505).

## 7. Consensus-variable update

For each interface pair, [`_average_boundary_frames()`](src/distopf/distributed/spatial/admm_agents.py:56) computes the arithmetic average

$$
z_e^{(k)}
=
\frac{1}{2}
\left(
 x_{e,\mathrm{loc}}^{(k)}
+
 x_{e,\mathrm{rem}}^{(k)}
\right).
$$

Componentwise,

$$
z_{e,\phi,t}^{(k)}
=
\frac{
 x_{e,\mathrm{loc},\phi,t}^{(k)}
+
 x_{e,\mathrm{rem},\phi,t}^{(k)}
}{2}.
$$

For two scalar copies, this is the solution of

$$
\min_z
\frac{1}{2}\left\|x_{\mathrm{loc}}-z\right\|_2^2
+
\frac{1}{2}\left\|x_{\mathrm{rem}}-z\right\|_2^2.
$$

The averaging is performed on voltage values and complex-power values directly. In particular, voltage consensus is not computed by averaging squared voltages.

## 8. Scaled-dual update

Each interface has a stored scaled dual variable $u_e^{(k)}$. The update in [`_process_interface_pair()`](src/distopf/distributed/spatial/admm_agents.py:338) is

$$
u_e^{(k+1)}
=
u_e^{(k)}
+x_{e,\mathrm{loc}}^{(k)}
-z_e^{(k)}.
$$

Using the average definition,

$$
u_e^{(k+1)}
=
u_e^{(k)}
+
\frac{1}{2}
\left(
 x_{e,\mathrm{loc}}^{(k)}
-
 x_{e,\mathrm{rem}}^{(k)}
\right).
$$

The initial duals are zero. This is established by [`_initialize_duals_if_needed()`](src/distopf/distributed/spatial/admm_agents.py:132):

$$
u_e^{(0)}=0.
$$

The implementation stores separate directional dual frames for
$$
u_{v^{\uparrow}},
\quad
u_{s^{\uparrow}},
\quad
u_{v^{\downarrow}},
\quad
u_{s^{\downarrow}}.
$$

Thus, the code does not store one globally shared dual object per physical interface. It stores each area's local dual representation, including child-specific slices maintained by [`_update_dual_slice()`](src/distopf/distributed/spatial/admm_agents.py:411).

## 9. Next local target

After updating the dual, the target written into the next local schedule is computed by [`_minus_frame()`](src/distopf/distributed/spatial/admm_agents.py:75):

$$
\widehat x_{e,\mathrm{loc}}^{(k+1)}
=
z_e^{(k)}-u_e^{(k+1)}.
$$

Substituting the dual update gives

$$
\widehat x_{e,\mathrm{loc}}^{(k+1)}
=
z_e^{(k)}
-u_e^{(k)}
-x_{e,\mathrm{loc}}^{(k)}
+z_e^{(k)}.
$$

Therefore,

$$
\widehat x_{e,\mathrm{loc}}^{(k+1)}
=
2z_e^{(k)}
-x_{e,\mathrm{loc}}^{(k)}
-u_e^{(k)}.
$$

Since $2z_e^{(k)}=x_{e,\mathrm{loc}}^{(k)}+x_{e,\mathrm{rem}}^{(k)}$, an especially useful identity is

$$
\boxed{
\widehat x_{e,\mathrm{loc}}^{(k+1)}
=
x_{e,\mathrm{rem}}^{(k)}-u_e^{(k)}
}.
$$

The complete pair-processing map is therefore

$$
\boxed{
\begin{aligned}
r_e^{(k)}
&=
\left\|
 x_{e,\mathrm{loc}}^{(k)}
-
 x_{e,\mathrm{rem}}^{(k)}
\right\|_{\infty},\\[1mm]
z_e^{(k)}
&=
\frac{1}{2}
\left(
 x_{e,\mathrm{loc}}^{(k)}
+
 x_{e,\mathrm{rem}}^{(k)}
\right),\\[1mm]
u_e^{(k+1)}
&=
 u_e^{(k)}
+x_{e,\mathrm{loc}}^{(k)}
-z_e^{(k)},\\[1mm]
\widehat x_{e,\mathrm{loc}}^{(k+1)}
&=
 z_e^{(k)}-u_e^{(k+1)}.
\end{aligned}
}
$$

These operations are applied by [`ADMMAgent.apply_messages()`](src/distopf/distributed/spatial/admm_agents.py:384).

## 10. Schedule writes

The shifted target is written according to the interface direction:

- Upstream voltage: [`_write_v_up_target()`](src/distopf/distributed/spatial/admm_agents.py:435).
- Upstream power: [`_write_s_up_target()`](src/distopf/distributed/spatial/admm_agents.py:448).
- Downstream voltage: [`_write_v_down_target()`](src/distopf/distributed/spatial/admm_agents.py:460).
- Downstream power: [`_write_s_down_target()`](src/distopf/distributed/spatial/admm_agents.py:471).

For complex power, [`add_s_to_schedules()`](src/distopf/distributed/spatial/enapp_agents.py:336) separates the target into active and reactive parts:

$$
\widehat P_{e,\alpha}^{(k+1)}
=\operatorname{Re}\left(\widehat S_{e,\alpha}^{(k+1)}\right),
\qquad
\widehat Q_{e,\alpha}^{(k+1)}
=\operatorname{Im}\left(\widehat S_{e,\alpha}^{(k+1)}\right).
$$

For voltage, the target is stored as a voltage magnitude and squared only when the local penalty is formed:

$$
\widehat V_{e,\alpha}^{(k+1)}
\longmapsto
\left(\widehat V_{e,\alpha}^{(k+1)}\right)^2.
$$

The relevant schedule helpers are [`add_v_swing_to_schedules()`](src/distopf/distributed/spatial/enapp_agents.py:267) and [`add_v_down_to_schedules()`](src/distopf/distributed/spatial/enapp_agents.py:305).

## 11. Full iteration map

Using a conventional indexing in which the local solve produces the $(k+1)$-st primal values, one ADMM iteration is:

### Step 1: Local area solves

$$
x_i^{(k+1)}
\in
\arg\min_{x_i\in\mathcal X_i}
\left[
 f_i(x_i)
+
\Phi_i\left(x_i;z^{(k)},u_i^{(k)}\right)
\right].
$$

The ideal scaled-ADMM form of the local quadratic term is

$$
\Phi_i\left(x_i;z^{(k)},u_i^{(k)}\right)
=
\sum_{e\sim i}
\frac{\rho_e}{2}
\left\|
\mathcal T_e(x_i)-z_e^{(k)}+u_{i,e}^{(k)}
\right\|_2^2,
$$
with the qualification that the implemented voltage terms use squared-voltage coordinates.

### Step 2: Boundary extraction

$$
\left(x_{i,e}^{(k+1)}\right)_{e\sim i}
\leftarrow
\text{boundary quantities extracted from }x_i^{(k+1)}.
$$

### Step 3: Message exchange

Each area sends its upstream and downstream boundary frames to neighboring areas through [`send_all_agent_messages()`](src/distopf/distributed/spatial/enapp_agents.py:579).

### Step 4: Residual calculation

$$
r^{(k+1)}
=
\max_e
\left\|
 x_{e,\mathrm{loc}}^{(k+1)}
-
 x_{e,\mathrm{rem}}^{(k+1)}
\right\|_{\infty}.
$$

### Step 5: Consensus and dual updates

$$
z_e^{(k+1)}
=
\frac{1}{2}
\left(
 x_{e,\mathrm{loc}}^{(k+1)}
+
 x_{e,\mathrm{rem}}^{(k+1)}
\right),
$$

$$
u_{i,e}^{(k+1)}
=
u_{i,e}^{(k)}
+x_{i,e}^{(k+1)}
-z_e^{(k+1)}.
$$

### Step 6: Next schedule target

$$
\widehat x_{i,e}^{(k+1)}
=
z_e^{(k+1)}-u_{i,e}^{(k+1)}.
$$

The exact Python execution order is local solve, boundary extraction, message routing, residual evaluation, and then target update, as shown in [`solve_admm()`](src/distopf/distributed/spatial/admm_agents.py:574).

## 12. Convergence test

The solver declares convergence when the global primal residual is below `tol` and no area solve failed during that iteration. Mathematically,

$$
 r^{(k)}<\varepsilon
 \quad\text{and}\quad
 \text{no local solve failed},
$$

where
$$
\varepsilon=\texttt{tol}.
$$

This condition is implemented in [`solve_admm()`](src/distopf/distributed/spatial/admm_agents.py:614).

The recorded residual history is

$$
\left\{
 r^{(1)},r^{(2)},\ldots,r^{(K)}
\right\}.
$$

No standard dual residual of the form

$$
 r_{\mathrm{dual}}^{(k)}
=\rho\left\|z^{(k)}-z^{(k-1)}\right\|
$$

is computed for stopping.

## 13. Cross-check findings

### 13.1 Scaled-dual convention

The update pattern is

$$
 z=\frac{x_i+x_j}{2},
\qquad
 u_i^+=u_i+x_i-z,
\qquad
 \widehat x_i^+=z-u_i^+.
$$

There is no explicit $\rho$ in the dual update. The penalty coefficients appear in the local objective as $\rho/2$ times a squared mismatch. This is consistent with a scaled-dual formulation if the stored $u$ variables are scaled duals.

### 13.2 Directional dual storage

The consensus average is symmetric,

$$
 z_{ij}=\frac{x_i+x_j}{2},
$$

but the dual storage is directional. Each area independently stores and updates its local dual frame. The implementation does not explicitly exchange dual variables.

If the reference formulation uses one unscaled dual per physical interface with an antisymmetry condition such as

$$
\lambda_{ij}=-\lambda_{ji},
$$

that relationship should be checked separately against the two local directional dual copies used by the code.

### 13.3 Voltage coordinate mismatch to verify

The consensus step uses

$$
 z_V=\frac{V_i+V_j}{2},
$$

while the local voltage penalty uses

$$
\frac{\rho_V}{2}
\left(V_i^2-z_V^2\right)^2.
$$

Therefore, the code does not implement either of the following alternatives:

$$
\frac{\rho_V}{2}(V_i-z_V)^2,
$$

or

$$
\frac{\rho_V}{2}
\left(
V_i^2-\frac{V_i^2+V_j^2}{2}
\right)^2.
$$

This is a key point for comparison with the reference mathematics.

### 13.4 Complex-power metric

The power penalty is

$$
|S-\widehat S|^2
=
(P-\widehat P)^2+(Q-\widehat Q)^2.
$$

Thus the code uses the Euclidean norm in the real-imaginary power plane, not a complex algebraic square $(S-\widehat S)^2$ and not an $\ell_\infty$ penalty.

### 13.5 Residual metric

The convergence test uses only the maximum primal interface mismatch:

$$
 r_{\mathrm{prim}}^{(k)}
=
\max_e
\left\|
 x_{e,\mathrm{loc}}^{(k)}
-
 x_{e,\mathrm{rem}}^{(k)}
\right\|_{\infty}.
$$

It does not use a combined primal-and-dual stopping test.

### 13.6 Root-area treatment

The parallel helper [`_solve_all_parallel()`](src/distopf/distributed/spatial/enapp_agents.py:631) removes `free_swing_voltage` for an area with no upstream recipients. Consequently, the root area can retain special physical swing-voltage treatment while non-root areas use a free boundary voltage.

This should be included when comparing the implementation with a mathematical model that assumes identical local subproblems for every area.

### 13.7 Failed local solves

If a local solve fails, [`_record_iteration_results()`](src/distopf/distributed/spatial/enapp_agents.py:677) marks the area as failed and retains its previous successful result when one exists. Convergence is not declared in that iteration. This is an implementation safeguard rather than part of the ideal ADMM recurrence.

## 14. Compact reference algorithm

The core pairwise scaled-ADMM algorithm can be summarized as

$$
\boxed{
\begin{aligned}
x_i^{k+1}
&\in
\arg\min_{x_i\in\mathcal X_i}
\left[
 f_i(x_i)
+
\sum_{e\sim i}
\frac{\rho_e}{2}
\left\|
\mathcal T_e(x_i)-z_e^k+u_{i,e}^k
\right\|_2^2
\right],\\[1mm]
z_e^{k+1}
&=
\frac{x_{i,e}^{k+1}+x_{j,e}^{k+1}}{2},\\[1mm]
u_{i,e}^{k+1}
&=
u_{i,e}^{k}
+x_{i,e}^{k+1}-z_e^{k+1},\\[1mm]
r^{k+1}
&=
\max_e
\left\|
 x_{i,e}^{k+1}-x_{j,e}^{k+1}
\right\|_{\infty}.
\end{aligned}
}
$$

For voltage interfaces, the abstract local term must be interpreted using the implementation's squared-voltage mismatch:

$$
\left(V_{i,e}^2-z_{V,e}^2\right)^2.
$$

For complex-power interfaces, it is interpreted as

$$
\left|S_{i,e}-z_{S,e}\right|^2
=
\left(P_{i,e}-z_{P,e}\right)^2
+
\left(Q_{i,e}-z_{Q,e}\right)^2.
$$

These equations provide the principal reference form for checking [`admm_agents.py`](src/distopf/distributed/spatial/admm_agents.py) against the intended ADMM derivation.
