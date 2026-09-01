"""GPU-native coupling between Newton rigid bodies and DEME mesh owners."""

from .contact_coupler import NewtonDEMEContactCoupler, NewtonDEMEOwnerMap

__all__ = ["NewtonDEMEContactCoupler", "NewtonDEMEOwnerMap"]
