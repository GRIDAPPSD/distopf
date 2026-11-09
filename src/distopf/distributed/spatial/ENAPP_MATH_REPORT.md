# Mathematical Formulation of the ENAPP Spatial Agent Implementation

**Date:** 2026-08-21  
**Source analyzed:** [`enapp_agents.py`](src/distopf/distributed/spatial/enapp_agents.py)  
**Purpose:** Translate the implemented ENAPP boundary exchange, local-solve, damping, and convergence logic into mathematical notation for comparison with a reference formulation.

## Summary

The implementation decomposes a distribution-system OPF/PF problem into area-local problems. Each area solves its local model, extracts boundary voltages and complex powers, exchanges selected boundary quantities with neighboring areas, updates its schedules, and repeats until the measured boundary change is below tolerance.

The implemented iteration is best represented as a synchronous fixed-point iteration. However, the code does not currently apply every exchanged message type, and its convergence metric only includes the upstream complex-power boundary and downstream voltage boundary. These implementation details are important when comparing the code against a complete ENAPP reference algorithm.

## 1. Notation and Sets

Let:

- $\mathcal{A}$ be the set of decomposed areas.
- $a \in \mathcal{A}$ denote an area.
- $\mathcal{T}$ be the set of time periods.
- $\Phi = \{a,b,c\}$ be the set of phases. To avoid confusion with area indices, phase indices below are written as $\phi \in \Phi$.
- $\mathcal{C}(a)$ be the set of downstream child areas of area $a$.
- $\mathcal{P}(a)$ be the unique parent area of $a$, when one exists.
- $k$ denote the ENAPP iteration number.
- $t \in \mathcal{T}$ denote a time period.

For each phase and time, the complex power is

$$
S_{\phi,t} = P_{\phi,t} + \mathrm{i}Q_{\phi,t},
$$

where $P_{\phi,t}$ is active power, $Q_{\phi,t}$ is reactive power, and $\mathrm{i}=\sqrt{-1}$.

The local area state at iteration $k$ can be represented abstractly as

$$
\mathbf{x}_a^{(k)} = \left(\mathbf{v}_a^{(k)},\mathbf{S}_a^{(k)},\text{local OPF variables}\right).
$$

The schedules supplied to the local solver contain time-dependent voltage and power boundary values.

## 2. Boundary Variables

The [`BoundaryVars`](src/distopf/distributed/spatial/enapp_agents.py:89) data structure stores four classes of boundary quantities for each area.

### 2.1 Upstream power boundary

The upstream power boundary $\mathbf{S}_{a,\mathrm{up}}$ is extracted from flows whose `from_name` is the swing bus. For each time $t$ and phase $\phi$,

$$
S_{a,\mathrm{up},\phi,t}
= P_{a,\mathrm{up},\phi,t}
+ \mathrm{i}Q_{a,\mathrm{up},\phi,t}.
$$

If multiple outgoing branches from the swing bus exist at the same time, the implementation sums them:

$$
S_{a,\mathrm{up},\phi,t}
= \sum_{\ell:\,\mathrm{from}(\ell)=\mathrm{swing}}
\left(P_{\ell,\phi,t}+\mathrm{i}Q_{\ell,\phi,t}\right).
$$

This is implemented by [`parse_s_up()`](src/distopf/distributed/spatial/enapp_agents.py:184), which first constructs $P+\mathrm{i}Q$ and then groups by the swing-bus name and time.

### 2.2 Upstream voltage boundary

The upstream voltage boundary $\mathbf{V}_{a,\mathrm{up}}$ is the phase voltage at the area swing bus:

$$
V_{a,\mathrm{up},\phi,t}
= V_{\mathrm{swing}(a),\phi,t}.
$$

The code treats this as a three-component voltage quantity

$$
\mathbf{V}_{a,\mathrm{up},t}
= \begin{bmatrix}
V_{a,\mathrm{up},a,t} \\
V_{a,\mathrm{up},b,t} \\
V_{a,\mathrm{up},c,t}
\end{bmatrix}.
$$

It is extracted by [`parse_v_up()`](src/distopf/distributed/spatial/enapp_agents.py:171).

### 2.3 Downstream voltage boundary

For each child boundary bus $b \in \mathcal{C}(a)$, the downstream voltage boundary is

$$
V_{a\rightarrow b,\phi,t}
= V_{b,\phi,t}.
$$

Equivalently, the downstream voltage vector associated with child $b$ is

$$
\mathbf{V}_{a\rightarrow b,t}
= \begin{bmatrix}
V_{b,a,t} \\
V_{b,b,t} \\
V_{b,c,t}
\end{bmatrix}.
$$

The implementation obtains these values by selecting voltage rows whose bus names belong to `down_buses`; see [`parse_v_dn()`](src/distopf/distributed/spatial/enapp_agents.py:213).

### 2.4 Downstream power boundary

For each downstream boundary bus $b$, the code sums all branch powers whose `to_name` equals $b$:

$$
S_{a\rightarrow b,\phi,t}
= \sum_{\ell:\,\mathrm{to}(\ell)=b}
\left(P_{\ell,\phi,t}+\mathrm{i}Q_{\ell,\phi,t}\right).
$$

Thus,

$$
S_{a\rightarrow b,\phi,t}
= P_{a\rightarrow b,\phi,t}
+ \mathrm{i}Q_{a\rightarrow b,\phi,t}.
$$

This is the operation performed by [`parse_s_dn()`](src/distopf/distributed/spatial/enapp_agents.py:229). The sign convention is inherited directly from the local power-flow result: the code does not negate the selected branch flow.

## 3. Boundary Differences and Absolute Values

For two boundary states $\mathbf{B}^{(1)}$ and $\mathbf{B}^{(2)}$, the implementation computes a row-aligned difference for each boundary field. For a scalar boundary component $x_{\phi,t}$,

$$
\Delta x_{\phi,t}
= x_{\phi,t}^{(1)}-x_{\phi,t}^{(2)}.
$$

The row keys are the pair $(\text{name},t)$. The implementation requires those keys to be unique and identical between the two frames; see [`_validate_boundary_frame_pair()`](src/distopf/distributed/spatial/enapp_agents.py:48).

The intended absolute difference is mathematically

$$
|\Delta x_{\phi,t}|.
$$

This is implemented by [`BoundaryVars.__abs__()`](src/distopf/distributed/spatial/enapp_agents.py:142) through [`_abs_frame()`](src/distopf/distributed/spatial/enapp_agents.py:126).

### Important complex-power caveat

For complex entries, applying Python/NumPy absolute value gives the complex magnitude:

$$
|\Delta S_{\phi,t}|
= \sqrt{\left(\operatorname{Re}\Delta S_{\phi,t}\right)^2
+\left(\operatorname{Im}\Delta S_{\phi,t}\right)^2}.
$$

Consequently, after `__abs__`, the stored complex-power entry is generally real and nonnegative. The later decomposition into real and imaginary parts therefore produces

$$
\operatorname{Re}\left(|\Delta S_{\phi,t}|\right)=|\Delta S_{\phi,t}|,
\qquad
\operatorname{Im}\left(|\Delta S_{\phi,t}|\right)=0.
$$

Therefore, the convergence calculation does **not** separately measure $|\Delta P|$ and $|\Delta Q|$ as its code structure may suggest. It measures the magnitude of the complex-power difference through the real-part channel, while the reactive channel is zero after the absolute-value operation.

If the reference method requires separate componentwise residuals, the mathematically direct alternative would be

$$
\epsilon_P = \max_{\phi,t}|P_{\phi,t}^{(k)}-P_{\phi,t}^{(k-1)}|,
$$

$$
\epsilon_Q = \max_{\phi,t}|Q_{\phi,t}^{(k)}-Q_{\phi,t}^{(k-1)}|,
$$

rather than applying $|\cdot|$ to $P+\mathrm{i}Q$ before splitting real and imaginary parts.

## 4. Boundary Message Equations

The four message kinds are declared in [`enapp_agents.py`](src/distopf/distributed/spatial/enapp_agents.py:33):

- $S_{\mathrm{up}}$: upstream power message.
- $V_{\mathrm{up}}$: upstream voltage message.
- $S_{\mathrm{down}}$: downstream power message.
- $V_{\mathrm{down}}$: downstream voltage message.

### 4.1 Upstream power message

An area sends its upstream power boundary to each configured upstream recipient:

$$
\mathcal{M}_{a\rightarrow r}^{S_{\mathrm{up}},(k)}
= \left\{S_{a,\mathrm{up},\phi,t}^{(k)}\right\}_{\phi\in\Phi,\,t\in\mathcal{T}}.
$$

When applied, [`add_s_to_schedules()`](src/distopf/distributed/spatial/enapp_agents.py:336) writes

$$
P_{a,\phi,t}^{\mathrm{schedule},(k+1)}
\leftarrow \operatorname{Re}\left(S_{a,\mathrm{up},\phi,t}^{(k)}\right),
$$

$$
Q_{a,\phi,t}^{\mathrm{schedule},(k+1)}
\leftarrow \operatorname{Im}\left(S_{a,\mathrm{up},\phi,t}^{(k)}\right).
$$

The schedule column names are of the form `sending_area.phase.p` and `sending_area.phase.q`.

### 4.2 Upstream voltage message

An area can send

$$
\mathcal{M}_{a\rightarrow r}^{V_{\mathrm{up}},(k)}
= \left\{V_{a,\mathrm{up},\phi,t}^{(k)}\right\}_{\phi,t}.
$$

However, [`AreaAgent.apply_messages()`](src/distopf/distributed/spatial/enapp_agents.py:463) currently does not apply $V_{\mathrm{up}}$ messages; the relevant schedule update is commented out. Thus, operationally,

$$
\text{effect of }V_{\mathrm{up}}\text{ message on schedules}=0.
$$

### 4.3 Downstream voltage message

For a child receiving an upstream area's downstream voltage, the intended message is

$$
\mathcal{M}_{a\rightarrow b}^{V_{\mathrm{down}},(k)}
= \left\{V_{a\rightarrow b,\phi,t}^{(k)}\right\}_{\phi,t}.
$$

The recipient uses [`add_v_swing_to_schedules()`](src/distopf/distributed/spatial/enapp_agents.py:267) to impose the received values as its swing-bus schedule:

$$
V_{b,\phi,t}^{\mathrm{swing\ schedule},(k+1)}
\leftarrow V_{a\rightarrow b,\phi,t}^{(k)}.
$$

This is the principal voltage coupling that is active in the current implementation.

### 4.4 Downstream power message

The downstream power message is

$$
\mathcal{M}_{a\rightarrow b}^{S_{\mathrm{down}},(k)}
= \left\{S_{a\rightarrow b,\phi,t}^{(k)}\right\}_{\phi,t}.
$$

Although the message is routed, [`AreaAgent.apply_messages()`](src/distopf/distributed/spatial/enapp_agents.py:463) intentionally skips its schedule update. Therefore,

$$
\text{effect of }S_{\mathrm{down}}\text{ message on schedules}=0.
$$

The function that would write downstream-area power columns is present but commented out.

## 5. Local Area Solve

For a fixed iteration $k$, each area solves its local OPF/PF problem:

$$
\mathbf{z}_a^{(k)}
\in
\arg\min_{\mathbf{z}_a}
\; f_a\left(\mathbf{z}_a;\,\text{objective}\right)
$$

subject to the area-local network model, operating limits, phase equations, time-coupling constraints, and boundary values currently present in the area schedule.

The exact objective and constraints are delegated to `case.run_opf()` through [`safe_area_solve()`](src/distopf/distributed/spatial/enapp_agents.py:503). The ENAPP file therefore defines the coordination equations, but not the detailed local OPF equations.

At each iteration, the implementation obtains

$$
\left\{\mathbf{z}_a^{(k)}\right\}_{a\in\mathcal{A}}
= \operatorname{SolveLocalProblems}
\left(\left\{\text{schedules}_a^{(k)}\right\}_{a\in\mathcal{A}}\right).
$$

The solve options forcibly include

$$
\texttt{free\_swing\_voltage}=\texttt{True},
$$

and the specified swing-voltage slack penalty $\rho_V$:

$$
\rho_V
= \texttt{swing\_voltage\_slack\_penalty}.
$$

The penalty is passed into the local model, but its algebraic form is defined outside this file.

## 6. Boundary Extraction Map

After a successful local solve, [`AreaAgent.set_result()`](src/distopf/distributed/spatial/enapp_agents.py:403) applies the extraction map

$$
\mathbf{B}_a^{(k)}
= \mathcal{H}_a\left(\mathbf{z}_a^{(k)}\right),
$$

where

$$
\mathbf{B}_a^{(k)}
= \left(
\mathbf{S}_{a,\mathrm{up}}^{(k)},
\mathbf{V}_{a,\mathrm{down}}^{(k)},
\mathbf{V}_{a,\mathrm{up}}^{(k)},
\mathbf{S}_{a,\mathrm{down}}^{(k)}
\right).
$$

The extraction map consists of:

$$
\mathbf{S}_{a,\mathrm{up}}^{(k)}=\texttt{parse\_s\_up}(\text{case}_a,\text{result}_a^{(k)}),
$$

$$
\mathbf{V}_{a,\mathrm{up}}^{(k)}=\texttt{parse\_v\_up}(\text{case}_a,\text{result}_a^{(k)}),
$$

$$
\mathbf{S}_{a,\mathrm{down}}^{(k)}=\texttt{parse\_s\_dn}(\text{case}_a,\text{result}_a^{(k)},\mathcal{C}(a)),
$$

$$
\mathbf{V}_{a,\mathrm{down}}^{(k)}=\texttt{parse\_v\_dn}(\text{case}_a,\text{result}_a^{(k)},\mathcal{C}(a)).
$$

## 7. Damping / Relaxation

The generic damping function [`dampen_boundaries()`](src/distopf/distributed/spatial/enapp_agents.py:784) implements the relaxed update

$$
\widetilde{B}_{a,j}^{(k)}
= \alpha B_{a,j}^{(k)}
+ (1-\alpha)B_{a,j}^{(k-1)},
$$

for each boundary type $j\in\{S_{\mathrm{up}},V_{\mathrm{down}},S_{\mathrm{down}},V_{\mathrm{up}}\}$ and each phase/time component.

The current call site uses

$$
\alpha=1.0,
$$

so the actual update is

$$
\widetilde{B}_{a,j}^{(k)}=B_{a,j}^{(k)}.
$$

Thus damping is currently a no-op. A conventional under-relaxed ENAPP update would use $0<\alpha<1$.

## 8. Iteration as a Fixed-Point Map

Let $\mathbf{u}^{(k)}$ denote the collection of boundary values inserted into all area schedules. The overall coordination process can be written as

$$
\mathbf{u}^{(k+1)}
= \mathcal{G}\left(\mathbf{u}^{(k)}\right),
$$

where $\mathcal{G}$ consists of:

1. solving all local area problems;
2. extracting boundary powers and voltages;
3. routing messages;
4. applying the active message types to local schedules.

With the current message application logic, $\mathcal{G}$ actively updates upstream power schedules and downstream/swing voltage schedules, while the downstream power and upstream voltage message paths have no schedule effect.

The first iteration performs a solve, boundary extraction, and message exchange, but skips convergence checking. Starting with iteration $k=2$, the code computes the boundary residual.

## 9. Boundary Convergence Metric

For each area $a$, define the boundary difference

$$
\Delta\mathbf{B}_a^{(k)}
= \mathbf{B}_a^{(k)}-\mathbf{B}_a^{(k-1)}.
$$

The implementation selects three quantities from this difference:

- downstream voltage difference $\Delta V_{a,\mathrm{down}}$;
- upstream active-power difference, as obtained from the real part of the absolute complex-power frame;
- upstream reactive-power difference, as obtained from the imaginary part of the absolute complex-power frame.

In intended reference notation, the apparent residual would commonly be written as

$$
\epsilon_a^{(k)}
= \max\left\{
\max_{b\in\mathcal{C}(a),\phi,t}
\left|V_{a\rightarrow b,\phi,t}^{(k)}-V_{a\rightarrow b,\phi,t}^{(k-1)}\right|,
\max_{\phi,t}\left|P_{a,\mathrm{up},\phi,t}^{(k)}-P_{a,\mathrm{up},\phi,t}^{(k-1)}\right|,
\max_{\phi,t}\left|Q_{a,\mathrm{up},\phi,t}^{(k)}-Q_{a,\mathrm{up},\phi,t}^{(k-1)}\right|
\right\}.
$$

The global residual is

$$
\epsilon^{(k)}=\max_{a\in\mathcal{A}}\epsilon_a^{(k)}.
$$

This corresponds to [`calculate_boundary_deviation()`](src/distopf/distributed/spatial/enapp_agents.py:722), subject to the complex absolute-value caveat in Section 3. The convergence condition is

$$
\epsilon^{(k)}<\tau
\quad\text{and no local area solve failed},
$$

where $\tau=\texttt{tol}$.

The default tolerance is

$$
\tau=10^{-6}.
$$

## 10. Swing-Voltage Diagnostic

The diagnostic function [`_calculate_swing_voltage_errors()`](src/distopf/distributed/spatial/enapp_agents.py:747) compares the solved swing voltage against the voltage values stored in the local schedule.

For each area, phase, and time, it computes

$$
E_{a,\phi,t}^{V,\mathrm{swing}}
= \left|
V_{a,\mathrm{swing},\phi,t}^{\mathrm{result}}
- V_{a,\phi,t}^{\mathrm{schedule}}
\right|.
$$

The reported per-area, per-phase diagnostic is

$$
E_{a,\phi}^{V,\mathrm{swing},\max}
= \max_{t\in\mathcal{T}}
E_{a,\phi,t}^{V,\mathrm{swing}}.
$$

This diagnostic is logged but is not included in the convergence condition. Therefore, the actual stopping test is not

$$
\max\left(\epsilon^{(k)},E^{V,\mathrm{swing},\max}\right)<\tau;
$$

it is only the boundary residual test $\epsilon^{(k)}<\tau$, together with the no-failure condition.

## 11. Failure and Partial-Solve Behavior

If an area solve fails at iteration $k$, the result for that area is `None`. The implementation retains the area's previous successful result, if one exists, and records the area in the failed-area set.

If any area has never solved successfully, the algorithm returns immediately with a partial result. In mathematical terms, if

$$
\left|\mathcal{A}_{\mathrm{solved}}\right|<|\mathcal{A}|,
$$

then the returned result is marked as an incomplete solve rather than a converged ENAPP solution.

Even when all areas have previous results, an iteration containing a new solve failure cannot satisfy the convergence condition because the code requires

$$
\texttt{iteration\_solve\_failed}=\texttt{False}.
$$

## 12. Global Objective Aggregation

After the area results are combined, the reported objective is the sum of objective values from root areas only. Let $\mathcal{R}\subseteq\mathcal{A}$ be the set of areas without upstream areas. Then

$$
F_{\mathrm{global}}
= \sum_{a\in\mathcal{R}}F_a,
$$

where $F_a$ is the local result's objective value and only non-null objective values are included.

This is implemented by [`_aggregate_root_objective()`](src/distopf/distributed/spatial/enapp_agents.py:835). It is not a sum over all areas unless the area decomposition and objective definitions make that equivalent.

## 13. High-Level Algorithm

The complete implemented algorithm can be summarized as follows.

### Initialization

1. Select each area's source bus.
2. Decompose the original case into local cases.
3. Construct one agent per area.
4. Set

$$
\texttt{free\_swing\_voltage}=\texttt{True}
$$

and pass the swing-voltage slack penalty to every local solve.

### Iteration $k=1,\ldots,k_{\max}$

For every area $a$:

$$
\mathbf{z}_a^{(k)}
\leftarrow
\operatorname{LocalSolve}_a\left(\text{schedules}_a^{(k)}\right).
$$

For every successful solve:

$$
\mathbf{B}_a^{(k)}
\leftarrow \mathcal{H}_a\left(\mathbf{z}_a^{(k)}\right).
$$

Apply damping:

$$
\widetilde{\mathbf{B}}_a^{(k)}
\leftarrow
\alpha\mathbf{B}_a^{(k)}
+(1-\alpha)\mathbf{B}_a^{(k-1)},
$$

with the current implementation using $\alpha=1$.

Exchange messages in the fixed order

$$
S_{\mathrm{up}}
\rightarrow V_{\mathrm{up}}
\rightarrow S_{\mathrm{down}}
\rightarrow V_{\mathrm{down}}.
$$

Apply the active schedule updates, principally

$$
S_{\mathrm{up}}\text{ into power schedule columns},
$$

and

$$
V_{\mathrm{down}}\text{ into receiving-area swing-voltage schedule columns}.
$$

For $k\ge 2$, calculate $\epsilon^{(k)}$. Stop if

$$
\epsilon^{(k)}<\tau
\quad\text{and no area solve failed on iteration }k.
$$

Otherwise continue until convergence or until

$$
 k=k_{\max}.
$$

## 14. Cross-Check Findings

The following points should be checked against the reference mathematics:

1. **Boundary orientation:** `parse_s_up()` selects flows leaving the swing bus, while `parse_s_dn()` selects flows entering downstream boundary buses. Confirm that these signs and orientations match the reference definition of interface power.
2. **Complex power aggregation:** The code forms $S=P+\mathrm{i}Q$ per branch and sums by boundary name and time. Confirm whether phase-wise summation is intended and whether any transformer or phase-mapping corrections are required.
3. **Active coupling paths:** $S_{\mathrm{up}}$ and $V_{\mathrm{down}}$ are applied to schedules. $S_{\mathrm{down}}$ and $V_{\mathrm{up}}$ are routed but currently ignored by `apply_messages()`.
4. **Residual definition:** The current residual examines `s_up` and `v_down`, not all four boundary fields. In particular, $S_{\mathrm{down}}$ and $V_{\mathrm{up}}$ do not directly enter the convergence test.
5. **Complex absolute value:** The current sequence `abs(complex power difference)` followed by real/imaginary splitting computes $|\Delta S|$ in the real channel and zero in the imaginary channel, not independent $|\Delta P|$ and $|\Delta Q|$ residuals.
6. **Damping:** The mathematical relaxation parameter exists, but the production call fixes $\alpha=1$, so no damping is currently performed.
7. **Swing-voltage diagnostic:** Swing-voltage schedule mismatch is calculated and logged, but it is not part of the stopping criterion.
8. **Objective aggregation:** The final objective is the sum over root-area objective values only. Confirm that this agrees with the decomposition's accounting convention and avoids double counting.
9. **Failure semantics:** A failed local solve prevents convergence for that iteration, and an area that has never solved causes an immediate partial-result return.
10. **First-iteration behavior:** The algorithm always performs one complete exchange before beginning residual checks on the second iteration.

## Conclusion

The file implements a synchronous, area-decomposed fixed-point coordination method. In compact form, its intended mathematical structure is

$$
\text{local solves}
\;\longrightarrow\;
\text{boundary extraction}
\;\longrightarrow\;
\text{message exchange}
\;\longrightarrow\;
\text{schedule update}
\;\longrightarrow\;
\text{residual test}.
$$

The most consequential differences to verify against the reference formulation are the inactive $S_{\mathrm{down}}$ and $V_{\mathrm{up}}$ schedule updates, the use of $\alpha=1$, and the treatment of complex-power differences in the convergence metric. These determine the actual fixed-point map and stopping condition implemented by the code.
