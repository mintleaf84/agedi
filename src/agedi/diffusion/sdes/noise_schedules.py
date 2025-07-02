from abc import ABC, abstractmethod
import math


class NoiseSchedule(ABC):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max
    
    @abstractmethod
    def f(self, t: float) -> float:
        """Returns the noise schedule value at time t."""
        pass

    @abstractmethod
    def fprime(self, t: float) -> float:
        """Returns the derivative of the noise schedule at time t."""
        pass

    @abstractmethod
    def fint(self, t: float) -> float:
        """Return the integral of the noise schedule at time t"""
        pass

    def df2dt(self, t: float) -> float:
        return 2 * self.f(t) * self.fprime(t)
    

class Linear(NoiseSchedule):
    def f(self, t: float) -> float:
        return self.min + (self.max - self.min) * t

    def fprime(self, t: float) -> float:
        return self.max - self.min

    def fint(self, t: float) -> float:
        return self.min * t + 0.5 * (self.max - self.min) * t **2

class Exponential(NoiseSchedule):
    def f(self, t: float) -> float:
        return self.min * (self.max / self.min) ** t

    def fprime(self, t: float) -> float:
        return self.min * (self.max / self.min) ** t * torch.log(self.max / self.min)

    def fint(self, t: float) -> float:
        return self.min * (self.max / self.min) ** t / torch.log(self.max / self.min)


class Cosine(NoiseSchedule):
        def f(self, t: float) -> float:
                return self.min + (self.max - self.min) * (1 - torch.cos(t * math.pi)) / 2
        
        def fprime(self, t: float) -> float:
                return (self.max - self.min) * math.pi * torch.sin(t * math.pi) / 2
        
        def fint(self, t: float) -> float:
                return (self.max - self.min) * (t / 2 - torch.sin(2 * t * math.pi) / (4 * math.pi))
        


    


