"""Diffusion: pure sampling logic without Lightning dependency.

This module provides :class:`Diffusion`, a plain Python class that
holds the score model, noisers, and an optional regressor model and exposes
the full sampling pipeline — including predictor-corrector sampling.

It is designed to be used standalone (e.g. for inference) or as a mixin base
for :class:`~agedi.diffusion.Agedi` (the Lightning training wrapper).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser

from .guidance import (
    BatchedLBFGSStepSizer,
    ForcefieldGuidanceConfig,
    force_field_guidance_step,
    post_diffusion_relaxation_step,
)


class Diffusion:
    """Pure-Python sampling core for diffusion models.

    Holds the score model, noisers, and an optional regressor and provides
    the full forward / reverse / sampling pipeline.  This class does **not**
    inherit from :class:`torch.nn.Module` or
    :class:`lightning.LightningModule` and therefore has no training hooks.

    When used through :class:`~agedi.diffusion.Agedi` (which inherits
    from both this class and :class:`lightning.LightningModule`), the
    Lightning infrastructure manages device placement and module registration.
    When used standalone, device information is derived from the score model's
    parameters via the :attr:`device` property.

    Parameters
    ----------
    score_model : ScoreModel
        The score model.
    noisers : List[Noiser]
        A list of noisers.
    regressor_model : torch.nn.Module, optional
        An optional regressor model used for force-field guidance during
        sampling.
    eps : float, optional
        Minimum value for the diffusion time step (used in
        :meth:`sample_time`).
    """

    def __init__(
        self,
        score_model: "ScoreModel",
        noisers: List[Noiser],
        regressor_model: Optional["torch.nn.Module"] = None,
        eps: float = 1e-5,
    ) -> None:
        self.score_model = score_model
        self.noisers = noisers
        self.regressor_model = regressor_model
        self.eps = eps
        self.lbfgs_step_sizer: Optional[BatchedLBFGSStepSizer] = None
        self.zeta: float = 3.0

        self.noiser_keys = [noiser.key for noiser in noisers]
        self.score_keys = [head.key for head in score_model.heads]

        if not set(self.noiser_keys) == set(self.score_keys):
            raise ValueError("Keys of noisers and score model heads do not match")

    @property
    def device(self) -> torch.device:
        """Infer the computation device from the score model's parameters.

        When used through :class:`~agedi.diffusion.Agedi` (which also
        inherits :class:`lightning.LightningModule`), Lightning's own
        ``device`` property takes precedence.
        """
        try:
            return next(self.score_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Core forward / reverse steps
    # ------------------------------------------------------------------

    def sample_time(self, batch: AtomsGraph) -> None:
        """Sample a random diffusion time for each graph in *batch*.

        Draws times uniformly from ``[eps, 1]`` and assigns them to
        ``batch.time`` at atom resolution.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data; modified in-place.
        """
        batch_size = batch.batch_size
        time = torch.rand(batch_size) * (1.0 - self.eps) + self.eps
        batch.time = time.to(self.device)[batch.batch].unsqueeze(1)

    def forward_step(self, batch: AtomsGraph) -> AtomsGraph:
        """Forward diffusion step (corruption).

        Applies each noiser in order to corrupt the batch.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.

        Returns
        -------
        AtomsGraph
            The corrupted batch.
        """
        for noiser in self.noisers:
            batch = noiser.noise(batch)

        batch.update_graph()
        return batch

    def reverse_step(
        self,
        batch: AtomsGraph,
        delta_t: float,
        force_field_guidance: float,
        last: bool = False,
    ) -> AtomsGraph:
        """Reverse diffusion step (denoising).

        Evaluates the score model and applies one reverse-SDE step through
        all noisers.  Optionally applies force-field guidance afterwards.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        delta_t : float
            The time step.
        force_field_guidance : float
            Scale of the force-field guidance (``0.0`` disables it).
        last : bool, optional
            Whether this is the final denoising step.

        Returns
        -------
        AtomsGraph
            The denoised batch.
        """
        batch = self.score_model(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.denoise(batch, delta_t, last=last)

        batch.wrap_positions()
        batch.update_graph()

        if self.regressor_model is not None and force_field_guidance > 0.0:
            batch = force_field_guidance_step(
                batch,
                self.regressor_model,
                self.lbfgs_step_sizer,
                scale=force_field_guidance * delta_t,
                zeta=self.zeta,
            )
            batch.wrap_positions()
            batch.update_graph()

        return batch

    def corrector_step(
        self,
        batch: AtomsGraph,
        corrector_dt: float,
    ) -> AtomsGraph:
        """Langevin corrector step at constant time.

        Evaluates the score model and applies one Langevin corrector step
        through all noisers (in reverse order).

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        corrector_dt : float
            Step size for the Langevin corrector.

        Returns
        -------
        AtomsGraph
            The corrected batch.
        """
        batch = self.score_model(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.langevin_step(batch, corrector_dt)
        batch.wrap_positions()
        batch.update_graph()
        return batch

    # ------------------------------------------------------------------
    # Guidance helpers (thin wrappers around module-level functions)
    # ------------------------------------------------------------------

    def force_field_guidance_step(
        self,
        batch: AtomsGraph,
        scale: float,
        max_step_size: float = 0.1,
    ) -> AtomsGraph:
        """Apply one force-field guidance step.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        scale : float
            Base scale of the force field guidance.
        max_step_size : float, optional
            Maximum allowed step size magnitude.

        Returns
        -------
        AtomsGraph
            Updated batch.
        """
        return force_field_guidance_step(
            batch,
            self.regressor_model,
            self.lbfgs_step_sizer,
            scale=scale,
            zeta=self.zeta,
            max_step_size=max_step_size,
        )

    def post_diffusion_relaxation_step(
        self,
        batch: AtomsGraph,
        scale: float = 0.1,
    ) -> AtomsGraph:
        """Perform a pure force-based relaxation step.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        scale : float, optional
            Step size scaling factor.

        Returns
        -------
        AtomsGraph
            Updated batch.
        """
        return post_diffusion_relaxation_step(
            batch,
            self.regressor_model,
            self.lbfgs_step_sizer,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # Graph initialisation
    # ------------------------------------------------------------------

    def _initialize_graph(self, cutoff: float, **kwargs) -> AtomsGraph:
        """Initialise a single graph from noiser priors.

        Parameters
        ----------
        cutoff : float
            Cutoff radius for the neighbour list.
        **kwargs
            Additional keyword arguments passed to the graph (e.g. ``cell``,
            ``template``).

        Returns
        -------
        AtomsGraph
            The initialised graph.
        """
        graph = AtomsGraph.empty(cutoff=cutoff)
        if "template" in kwargs:
            template = kwargs.pop("template")
        else:
            template = None

        if "cell" in kwargs:
            cell = kwargs.pop("cell")
            setattr(graph, "cell", cell)

        for k, v in kwargs.items():
            setattr(graph, k, v)

        for noiser in self.noisers[::-1]:
            noiser.initialize_graph(graph)

        if template is not None:
            new_graph = template.clone()

            setattr(
                new_graph,
                "x",
                torch.cat([template.x, graph.x]),
            )

            setattr(
                new_graph,
                "pos",
                torch.cat([template.pos, graph.pos]),
            )

            setattr(
                new_graph,
                "mask",
                torch.cat([
                    torch.ones_like(template.x, dtype=torch.bool),
                    torch.zeros_like(graph.x, dtype=torch.bool),
                ]),
            )

            setattr(new_graph, "n_atoms", template.n_atoms + graph.n_atoms)
        else:
            new_graph = graph
            setattr(new_graph, "mask", torch.zeros_like(graph.x, dtype=torch.bool))

        return new_graph

    # ------------------------------------------------------------------
    # Internal sampling loop
    # ------------------------------------------------------------------

    def _sample_batch(
        self,
        batch: Batch,
        steps: int,
        eps: float,
        force_field_guidance: float,
        save_path: bool,
        progress_bar: bool,
        force_threshold: float,
        max_extra_steps: int,
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
    ) -> List[AtomsGraph]:
        """Run the reverse-diffusion loop for a pre-built batch.

        Parameters
        ----------
        batch : Batch
            A batch of :class:`~agedi.data.AtomsGraph` data at ``t=1``.
        steps : int
            Number of reverse-diffusion steps.
        eps : float
            Minimum time value (end of trajectory).
        force_field_guidance : float
            Scale of the force-field guidance (``0.0`` disables it).
        save_path : bool
            Whether to collect and return all intermediate states.
        progress_bar : bool
            Whether to display a tqdm progress bar.
        force_threshold : float
            Maximum per-atom force for terminating post-diffusion relaxation.
        max_extra_steps : int
            Maximum extra relaxation steps after the main trajectory.
        corrector_steps : int, optional
            Number of Langevin corrector passes after each predictor step.
            ``0`` (default) disables the corrector (standard DDPM/EM sampling).
        corrector_step_size : float, optional
            Step size used for each Langevin corrector step.  Defaults to
            ``1e-3``.

        Returns
        -------
        List[AtomsGraph]
            Final structures, or (when *save_path* is ``True``) a list of
            trajectories (one per graph).
        """
        if steps < 2:
            return batch.to_data_list()

        if force_field_guidance > 0 and self.regressor_model is not None:
            self.lbfgs_step_sizer = BatchedLBFGSStepSizer(
                batch_size=batch.batch_size
            )

        ts = torch.linspace(1, eps, steps, device=self.device)
        dt = ts[0] - ts[1]

        # Pre-create corrector delta_t tensor once to avoid per-call allocation.
        corrector_dt: Optional[torch.Tensor] = None
        if corrector_steps > 0:
            corrector_dt = torch.tensor(
                corrector_step_size, dtype=dt.dtype, device=self.device
            )

        if save_path:
            path = []

        if progress_bar:
            iterator = tqdm(range(steps))
        else:
            iterator = range(steps)

        for i in iterator:
            if save_path:
                path.append(batch.to_data_list())

            batch.add_batch_attr("time", ts[i].repeat(batch.x.shape[0], 1), type="node")

            # Predictor step
            if i < steps - 1:
                batch = self.reverse_step(batch, dt, force_field_guidance)
            else:
                batch = self.reverse_step(batch, dt, force_field_guidance, last=True)

            # Corrector steps at constant time t_i
            for _ in range(corrector_steps):
                assert corrector_dt is not None  # guaranteed by pre-loop initialisation
                batch.add_batch_attr(
                    "time", ts[i].repeat(batch.x.shape[0], 1), type="node"
                )
                batch = self.corrector_step(batch, corrector_dt)

        # Optional post-diffusion relaxation
        if force_field_guidance > 0 and self.regressor_model is not None:
            batch = self.regressor_model(batch)
            max_forces = torch.norm(batch.forces_prediction, dim=1).max(dim=0)[0]

            if max_forces > force_threshold and max_extra_steps > 0:
                if progress_bar:
                    print(
                        f"Max force after diffusion: {max_forces:.4f}, "
                        "continuing relaxation..."
                    )
                    extra_iterator = tqdm(
                        range(max_extra_steps), desc="Post-diffusion relaxation"
                    )
                else:
                    extra_iterator = range(max_extra_steps)

                batch.add_batch_attr(
                    "time", torch.zeros_like(batch.time), type="node"
                )

                for i in extra_iterator:
                    batch = self.post_diffusion_relaxation_step(batch, scale=0.1)

                    batch = self.regressor_model(batch)
                    max_forces = torch.norm(batch.forces_prediction, dim=1).max(
                        dim=0
                    )[0]

                    if save_path:
                        path.append(batch.to_data_list())

                    if max_forces <= force_threshold:
                        if progress_bar:
                            print(
                                f"Relaxation converged after {i+1} steps, "
                                f"max force: {max_forces:.4f}"
                            )
                        break

                if progress_bar and max_forces > force_threshold:
                    print(
                        f"Relaxation did not converge, "
                        f"final max force: {max_forces:.4f}"
                    )

        if save_path:
            path.append(batch.to_data_list())
            return list(map(list, zip(*path)))

        return batch.to_data_list()

    def _sample(
        self,
        N: int,
        steps: int,
        cutoff: float,
        eps: float,
        force_field_guidance: float,
        force_threshold: float,
        max_extra_steps: int,
        progress_bar: bool,
        save_path: bool,
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
        **kwargs,
    ) -> List[AtomsGraph]:
        """Build *N* graphs from priors and run the sampling loop.

        Parameters
        ----------
        N : int
            Number of structures to generate.
        steps : int
            Number of reverse-diffusion steps.
        cutoff : float
            Cutoff radius for the neighbour list.
        eps : float
            Minimum time value (end of trajectory).
        force_field_guidance : float
            Scale of the force-field guidance.
        force_threshold : float
            Maximum per-atom force for post-diffusion relaxation.
        max_extra_steps : int
            Maximum extra relaxation steps.
        progress_bar : bool
            Show tqdm progress bar.
        save_path : bool
            Collect all intermediate states.
        corrector_steps : int, optional
            Langevin corrector passes per predictor step.
        corrector_step_size : float, optional
            Step size for each corrector pass.
        **kwargs
            Keyword arguments forwarded to :meth:`_initialize_graph`.

        Returns
        -------
        List[AtomsGraph]
            Sampled structures (or trajectories when *save_path* is ``True``).
        """
        data = []
        for _ in range(N):
            data.append(self._initialize_graph(cutoff, **kwargs))

        batch = Batch.from_data_list(data).to(self.device)
        batch.update_graph()

        return self._sample_batch(
            batch,
            steps,
            eps,
            force_field_guidance,
            save_path,
            progress_bar,
            force_threshold,
            max_extra_steps,
            corrector_steps=corrector_steps,
            corrector_step_size=corrector_step_size,
        )

    # ------------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------------

    def sample(
        self,
        N: int,
        template: Optional[AtomsGraph] = None,
        batch_size: Optional[int] = 64,
        steps: Optional[int] = 500,
        cutoff: Optional[float] = 6.0,
        eps: Optional[float] = 1e-3,
        n_atoms: Optional[int] = None,
        atomic_numbers: Optional[List[int]] = None,
        formula: Optional[str] = None,
        positions: Optional[np.ndarray] = None,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
        confinement: Optional[Tuple[float, float]] = None,
        ff_guidance: Optional[ForcefieldGuidanceConfig] = None,
        property: Optional[Dict] = None,
        progress_bar: Optional[bool] = False,
        save_path: Optional[bool] = False,
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
    ) -> List[AtomsGraph]:
        """Sample structures from the diffusion model.

        The minimum required arguments depend on the configured noisers and
        whether a template is provided:

        * ``n_atoms`` – always required unless derivable from
          ``atomic_numbers`` or ``formula``.
        * ``atomic_numbers`` – required unless a types-noiser is configured
          (key ``"x"``), or derivable from ``formula``.
        * ``positions`` – required when no positions-noiser is configured
          (type-only diffusion).
        * ``cell`` – required when no ``template`` is given.
        * ``pbc`` – optional; defaults to ``[True, True, True]``.

        Parameters
        ----------
        N : int
            Number of structures to generate.
        template : AtomsGraph, optional
            Template structure.  ``cell`` and ``pbc`` are taken from the
            template when not explicitly provided.
        batch_size : int, optional
            Internal batch size for splitting large *N*.
        steps : int, optional
            Number of reverse-diffusion steps.
        cutoff : float, optional
            Cutoff radius for the neighbour list.
        eps : float, optional
            Minimum time value at the end of the trajectory.
        n_atoms : int, optional
            Number of atoms per structure.
        atomic_numbers : List[int], optional
            Atomic numbers of the atoms to generate.
        formula : str, optional
            Chemical formula (e.g. ``"H2O"``).
        positions : np.ndarray, optional
            Fixed atom positions (shape ``(n_atoms, 3)``).
        cell : np.ndarray, optional
            Unit-cell matrix (3×3).
        pbc : np.ndarray, optional
            Periodic boundary conditions.
        confinement : Tuple[float, float], optional
            Z-directional confinement ``(z_min, z_max)``.
        ff_guidance : ForcefieldGuidanceConfig, optional
            Force-field guidance configuration.
        property : dict, optional
            Conditioning properties (key → scalar tensor).
        progress_bar : bool, optional
            Show a tqdm progress bar.
        save_path : bool, optional
            Return full trajectories instead of final structures.
        corrector_steps : int, optional
            Number of Langevin corrector passes after each predictor step.
            ``0`` (default) gives standard (predictor-only) sampling.
        corrector_step_size : float, optional
            Step size for each corrector pass.  Defaults to ``1e-3``.

        Returns
        -------
        List[AtomsGraph]
            Sampled structures, or trajectories when *save_path* is ``True``.
        """
        if ff_guidance is None:
            ff_guidance = ForcefieldGuidanceConfig()

        self.score_model.sample_mode()

        # Derive n_atoms / atomic_numbers from a molecular formula if given.
        if formula is not None:
            from ase import Atoms as _AseAtoms

            _formula_atoms = _AseAtoms(formula)
            if n_atoms is None:
                n_atoms = len(_formula_atoms)
            if atomic_numbers is None and "x" not in self.noiser_keys:
                atomic_numbers = _formula_atoms.get_atomic_numbers().tolist()

        # When a template is provided but no cell is given, borrow the
        # template's cell so noiser priors (e.g. UniformCell) can use it.
        if template is not None and cell is None:
            cell = template.cell.detach().cpu().numpy()

        kwargs: Dict = {}
        # Sampling-control parameters passed explicitly to _sample.
        sample_kwargs: Dict = {
            "progress_bar": progress_bar,
            "save_path": save_path,
            "force_threshold": ff_guidance.force_threshold,
            "max_extra_steps": ff_guidance.max_extra_steps,
            "corrector_steps": corrector_steps,
            "corrector_step_size": corrector_step_size,
        }
        self.zeta = ff_guidance.zeta

        if n_atoms is not None:
            kwargs["n_atoms"] = torch.tensor([n_atoms]).reshape(1, 1)
        if positions is not None:
            kwargs["pos"] = torch.tensor(
                np.array(positions), dtype=torch.float
            ).reshape(-1, 3)
            if "n_atoms" not in kwargs:
                kwargs["n_atoms"] = torch.tensor(
                    [kwargs["pos"].shape[0]]
                ).reshape(1, 1)
        if atomic_numbers is not None:
            kwargs["x"] = torch.tensor(atomic_numbers, dtype=torch.long).reshape(-1)
            if "n_atoms" not in kwargs:
                kwargs["n_atoms"] = torch.tensor([len(atomic_numbers)]).reshape(1, 1)

        if cell is not None:
            kwargs["cell"] = torch.tensor(
                np.array(cell), dtype=torch.float
            ).reshape(3, 3)

        if property is not None:
            for k, v in property.items():
                kwargs[k] = torch.tensor(v, dtype=torch.float)

        for key in ["pos", "x", "cell", "n_atoms"]:
            if key not in kwargs and key not in self.noiser_keys:
                if key == "pos" and "frac" in self.noiser_keys:
                    continue
                raise ValueError(
                    f"Missing default values for key {key} in kwargs."
                )

        if confinement is not None:
            kwargs["confinement"] = torch.tensor(
                confinement, dtype=torch.float
            ).reshape(1, 2)

        if template is not None:
            kwargs["template"] = template
        else:
            n_atoms = kwargs["n_atoms"].item()

        if pbc is not None:
            kwargs["pbc"] = torch.tensor(pbc, dtype=torch.bool).reshape(3)

        if N > batch_size:
            n_full = N // batch_size
            n_remainder = N % batch_size
            n_batches = n_full + (1 if n_remainder > 0 else 0)
            out = []
            for i in range(n_full):
                print(f"Sampling batch {i + 1}/{n_batches}...")
                out += self._sample(
                    batch_size, steps, cutoff, eps, ff_guidance.guidance,
                    **sample_kwargs, **kwargs,
                )
            if n_remainder > 0:
                print(f"Sampling batch {n_batches}/{n_batches}...")
                out += self._sample(
                    n_remainder, steps, cutoff, eps, ff_guidance.guidance,
                    **sample_kwargs, **kwargs,
                )
            return out
        else:
            return self._sample(
                N, steps, cutoff, eps, ff_guidance.guidance,
                **sample_kwargs, **kwargs,
            )
