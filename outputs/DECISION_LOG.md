# 📔 Research Decision Log: Malaysia HANK

This document records key methodological choices, deviations from standard literature, and calibration adjustments made during the development of the Malaysia HANK model.

## 📅 Session: Calibration of Preferences (Beta)

### 1. The Issue: The "Savings Glut"
*   **Observation:** After introducing the "Segmented Labor Market" with high income risk for the informal sector ($\sigma_{informal} = 0.75$), the model's equilibrium interest rate ($r$) collapsed to **2.26%**.
*   **Cause:** The high risk induced excessive "precautionary savings." Households accumulated too much capital ($K/Y = 4.55$) relative to the demand, driving returns down.
*   **Target:** Malaysia's historical real interest rate is approx. **3.0%**.

### 2. The Methodological Fork
We considered two approaches to fix this:

*   **Path A: Homogeneous Beta (Selected)**
    *   *Method:* Keep a single discount factor ($\beta$) for all agents but lower it.
    *   *Precedent:* Standard HANK (Kaplan, Moll, Violante, 2018) uses homogeneous preferences. Differences in wealth arise purely from income shocks and frictions.
    *   *Adjustment:* Lowered $\beta$ from 0.95 to **0.935**.
    *   *Rationale:* This maintains the "Canon" structure while acknowledging that in a high-risk environment (Emerging Market), the *average* agent must be more impatient to explain why the savings rate isn't infinite.

*   **Path B: Heterogeneous Beta (Deferred)**
    *   *Method:* Assign $\beta_{low}$ to informal workers and $\beta_{high}$ to formal workers.
    *   *Precedent:* Krusell & Smith (1998), "Behavioral HANK."
    *   *Rationale:* Captures the specific "short-termism" of the poor.
    *   *Decision:* Deferred to Phase 2 to avoid over-complicating the baseline identification.

### 3. Implications for Report
When writing the final report, we must explicitly state:
> "We calibrate the discount factor $\beta = 0.935$ internally to match the real interest rate target of 3.0%, given the heightened idiosyncratic risk parameters derived from the informal sector data."

---
*Log updated automatically by Gemini Conductor.*
