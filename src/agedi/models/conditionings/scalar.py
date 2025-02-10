import torch
from .base import Conditioning

class ScalarConditioning(Conditioning):

    def __init__(self, *args, **kwargs):
        super().__init__(input_dim=1, output_dim=2, *args, **kwargs)

        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim),
        )

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
        x = x.view(-1, 1)
        c = self.embedder(x)
        c = torch.cat([torch.cos(c), torch.sin(c)], dim=-1)

        return c

    def get_empty_conditioning(self, n: int) -> torch.Tensor:
        """Get an empty conditioning tensor.

        Returns
        -------
        torch.Tensor
            Empty conditioning tensor of shape (n, 2).

        """
        return torch.zeros(n, self.output_dim, device=self.device)


