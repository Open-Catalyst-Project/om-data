from typing import Dict


def validate_metadata_file(metadata: Dict):
    """
    Validates the metadata file to ensure that it contains the necessary fields.
    Args:
        metadata: Dictionary containing metadata for the system.
    """
    required_fields = [
        "residue",
        "species",
        "solute_or_solvent",
        "partial_charges",
    ]
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Metadata file is missing required field: {field}")
