from .base import Noiser
from .sde import SDENoiser
from .pos import PositionsNoiser, Positions, CellPositions, ConfinedCellPositions
from .types import Types, TypesNoiser
from .weighted_pos import WeightedPositionsNoiser

# ---------------------------------------------------------------------------
# Built-in noiser registry entries
# ---------------------------------------------------------------------------
# Register the built-in noisers so they can be referenced by name in
# :func:`~agedi.functional.create_diffusion`.  The factory callable receives
# ``sde`` as a keyword argument; noisers that do not use an SDE (like
# :class:`Types`) can simply ignore it.

# CamelCase names (primary)
Noiser.register("Positions", lambda sde: Positions(sde=sde))
Noiser.register("CellPositions", lambda sde: CellPositions(sde=sde))
Noiser.register("ConfinedCellPositions", lambda sde: ConfinedCellPositions(sde=sde))
Noiser.register("Types", lambda sde: Types())
Noiser.register("WeightedPositions", lambda sde: WeightedPositionsNoiser(sde=sde))

# snake_case aliases for backwards compatibility
Noiser.register("positions", lambda sde: Positions(sde=sde))
Noiser.register("cell_positions", lambda sde: CellPositions(sde=sde))
Noiser.register("confined_cell_positions", lambda sde: ConfinedCellPositions(sde=sde))
Noiser.register("types", lambda sde: Types())
Noiser.register("weighted_positions", lambda sde: WeightedPositionsNoiser(sde=sde))

__all__ = [
    "Noiser",
    "SDENoiser",
    "PositionsNoiser",
    "Positions",
    "CellPositions",
    "ConfinedCellPositions",
    "Types",
    "TypesNoiser",
]
