import torch
from .base import Conditioning

class IntegerConditioning(Conditioning):
    """Conditioning module for integer-valued properties.

    Embeds an integer property (e.g. number of atoms) into a fixed-size
    representation using :class:`torch.nn.Embedding`.
    """

    def __init__(self, max_int: int = 200, *args, **kwargs) -> None:
        """Initialize the integer conditioning module.

        Parameters
        ----------
        max_int : int, optional
            Maximum integer value supported by the embedding table.
        *args
            Positional arguments forwarded to :class:`~agedi.models.conditionings.base.Conditioning`.
        **kwargs
            Keyword arguments forwarded to :class:`~agedi.models.conditionings.base.Conditioning`.
        """
        super().__init__(input_dim=1, output_dim=64, *args, **kwargs)
        self.embedder = torch.nn.Embedding(max_int, self.output_dim)


    def get_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Get the conditioning tensor for x

        Parameters
        ----------
        x : torch.Tensor
            Time tensor of shape (Nodes, 1).

        Returns
        -------
        torch.Tensor
            Conditioning tensor of shape (Nodes, 2).

        """
        x = x.long()  # Ensure x is of type long for embedding
        c = self.embedder(x)

        return c

    def get_empty_conditioning(self, n: int) -> torch.Tensor:
        """Get an empty conditioning tensor.

        Returns
        -------
        torch.Tensor
            Empty conditioning tensor of shape (n, 2).

        """
        return torch.zeros(n, self.output_dim, device=self.device)


