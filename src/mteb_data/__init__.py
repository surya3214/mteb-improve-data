"""mteb-data: generate training views from MTEB Multilingual v2."""

from mteb_data.catalog import Catalog, load_catalog
from mteb_data.schema import CanonicalRecord, Provenance

__all__ = ["Catalog", "CanonicalRecord", "Provenance", "load_catalog"]
__version__ = "0.1.0"
