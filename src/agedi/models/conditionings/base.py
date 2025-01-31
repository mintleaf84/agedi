from abc import ABC, abstractmethod

import torch


class Conditioning(ABC, torch.nn.Module):
    """Conditioning Base Class

    Parameters
    ----------
    property: str
        The property of the batch to condition on
    input_dim: int
        The dimension of the input conditioning
    output_dim: int
        The dimension of the output conditioning
    concatenation_type: str
        The type of concatenation to use. Default is "scalar"
    probability: float
        The probability of conditioning. Default is 0.5. Only used in training mode

    Returns
    -------
    Conditioning
    
    """

    def __init__(
        self,
        property: str,
        input_dim: int,
        output_dim: int,
        concatenation_type: str = "scalar",
        probability: float = 0.5,
        **kwargs,
    ) -> None:
        """Constructor for the Conditioning class
        """
        super().__init__(**kwargs)
        self.property = property
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.concatenation_type = concatenation_type
        self.probability = probability

    @abstractmethod
    def get_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract method to get the conditioning from the input

        Must be implemented by the subclass

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The conditioning tensor
        
        """
        pass

    @abstractmethod
    def get_emtpy_conditioning(self, n: int) -> torch.Tensor:
        """Abstract method to get an empty conditioning tensor

        Must be implemented by the subclass

        Parameters
        ----------
        n: int
            The number of nodes in the batch

        Returns
        torch.Tensor
            The empty conditioning tensor
        
        """
        
    def forward(self, batch: "AtomsGraph", empty: bool=False) -> "AtomsGraph":
        """Forward method to get the conditioning from the input

        Parameters
        ----------
        batch: AtomsGraph
            The input batch
        empty: bool
            If True, return an empty conditioning tensor

        Returns
        -------
        AtomsGraph
            The batch with the conditioning added to the representation
        
        """
        if self.training:
            print('in training mode!')
            n = batch.batch_size
            
            cond_idx = torch.rand(n) < self.probability
            cond_idx = cond_idx[batch.batch]
            
            c = self.get_emtpy_conditioning(batch[self.property])
            c[cond_idx] = self.get_conditioning(batch[self.property])[cond_idx]
        else:
            print('in eval mode!')
            if empty:
                c = self.get_empty_conditioning(batch[self.property].shape[0])
            else:
                c = self.get_conditioning(batch[self.property])

        self.concatenate(batch, c)

        return batch

    def concatenate(self, batch: "AtomsGraph", c: torch.Tensor) -> None:
        """Concatenate the conditioning to the batch

        Parameters
        ----------
        batch: AtomsGraph
            The input batch
        c: torch.Tensor
            The conditioning tensor

        Returns
        -------
        None
        
        """
        if self.concatenation_type == "scalar":
            rep = batch.representation
            scalar = rep.scalar

            if scalar.shape[0] != c.shape[0]:
                raise ValueError(
                    "Scalar and conditioning have different number of nodes"
                )

            new_scalar = torch.cat((scalar, c), dim=1)
            rep.scalar = new_scalar
            batch.representation = rep

        else:
            raise ValueError(
                f"Concatenation type {self.concatenation_type} not supported"
            )
