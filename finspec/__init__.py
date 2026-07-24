"""FinSpec core package."""

from finspec.decoding import finspec_forward
from finspec.schema_fsm import SchemaFSM

__all__ = ["SchemaFSM", "finspec_forward"]

__version__ = "0.1.0"
