"""Configuration file for the project."""

from typing import Any, List, Dict

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML.

    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """

    num_features: List[str]
    cat_features: List[str]
    binary_features: List[str]
    target: str
    secondary_target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]
    high_cardinality_features: List[str]
    data_quality: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    # ADD THESE THREE LINES - they're in your YAML but missing from the class
    experiment_name_basic: str
    experiment_name_custom: str
    experiment_name_fe: str

    @classmethod
    def from_yaml(cls, config_path: str = "project_config.yml", env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file (default: project_config.yml)
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]

            return cls(**config_dict)


class Tags(BaseModel):
    """Represents a set of tags for a Git commit.

    Contains information about the Git SHA, branch, and job run ID.
    """

    git_sha: str
    branch: str
    experiment_name: str = "financial_complaints_mlops"
    model_name: str = "financial_complaints_classifier"