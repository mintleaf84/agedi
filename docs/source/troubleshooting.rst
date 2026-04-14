Trouble Shooting
================

Below you find problems that you might face while training or sampling
a diffusion model using AGeDi. 

NaN mean (probably position) values
----------------------------------

This error is raised when the z-coordinate mean passed to the
``TruncatedNormal`` distribution contains ``NaN`` values during a
reverse-diffusion (sampling) step.

**What causes it?**

When confinement is active (e.g. ``--confinement z_min z_max``),
mobile-atom z-coordinates are kept inside ``[z_min, z_max]`` by
sampling from a truncated normal distribution at each step.  If the
Euler-Maruyama update pushes a coordinate outside the confinement
bounds, the mean supplied to the next truncated-normal sample becomes
invalid, eventually producing ``NaN`` values.

Common triggers:

* **Step-size too large** — using a small number of reverse steps
  (``--steps``) means each step is large and more likely to overshoot
  the confinement boundary.  Try increasing ``--steps``.
* **Confinement mismatch** — if the ``--confinement`` bounds used
  during *sampling* differ from those used during *training*, the score
  model is evaluated outside its training distribution and can produce
  scores that drive atoms out of bounds.  Ensure training and sampling
  use the same confinement bounds.

**How is it handled in the current code?**

As of the current version, positions are clamped back to
``[z_min, z_max]`` after every reverse step, and the mean passed to the
truncated-normal distribution is also clamped before constructing the
distribution.  These two safeguards together mean this error should
only be triggered if a ``NaN`` arises from a different source (e.g. the
score model itself returning ``NaN`` gradients).  If you still see this
error, check the model output for numerical instabilities.

