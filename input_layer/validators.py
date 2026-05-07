from __future__ import annotations

from input_layer.contracts import InputManifest, ValidationIssue, ValidationReport
from input_layer.dom import validate_dom_input_contract
from input_layer.height import validate_dem_sources, validate_height_rasters
from input_layer.prior_data import validate_prior_data_knowledge_items, validate_prior_data_tables
from input_layer.public_dataset import validate_public_datasets
from input_layer.remote_sensing import validate_remote_sensing_sources
from input_layer.vector import validate_vector_sources


def validate_input_manifest(manifest: InputManifest) -> ValidationReport:
    issues: list[ValidationIssue] = []

    validate_remote_sensing_sources(manifest.remote_sensing, issues)
    validate_dom_input_contract(manifest.dom_input_contract, issues)
    validate_dem_sources(manifest.terrain_dem, issues)
    validate_height_rasters(manifest.canopy_height, issues)
    validate_height_rasters(manifest.surface_models, issues)
    validate_prior_data_tables(manifest.survey_tables, issues)
    validate_vector_sources(manifest.industry_vectors, issues)
    validate_prior_data_knowledge_items(manifest.domain_knowledge_items, issues)
    validate_public_datasets(manifest.public_datasets, issues)

    status = "ok"
    if any(item.level == "error" for item in issues):
        status = "invalid"
    elif issues:
        status = "warning"
    return ValidationReport(status=status, issues=issues)
