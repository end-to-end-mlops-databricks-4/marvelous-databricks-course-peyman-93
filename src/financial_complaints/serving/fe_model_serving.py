"""Feature Lookup Serving module for financial complaints."""

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class FeatureLookupServing:
    """Manage Feature Lookup Serving operations for financial complaints."""

    def __init__(self, model_name: str, endpoint_name: str, company_table: str, state_table: str) -> None:
        """Initialize the Feature Lookup Serving Manager.

        :param model_name: Name of the model
        :param endpoint_name: Name of the endpoint
        :param company_table: Name of the company features table
        :param state_table: Name of the state features table
        """
        self.workspace = WorkspaceClient()
        self.company_table = company_table
        self.state_table = state_table
        self.model_name = model_name
        self.endpoint_name = endpoint_name
        self.fe = FeatureEngineeringClient()

    def get_latest_model_version(self) -> str:
        """Get the latest version of the model.

        :return: Latest model version
        """
        client = mlflow.MlflowClient()
        try:
            latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-fe-model").version
        except:
            # Fallback to latest-model if latest-fe-model doesn't exist
            latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def create_or_update_online_table(self, online_store, table_name: str) -> None:
        """Create or update an online table for features.
        
        :param online_store: Online store instance
        :param table_name: Source table name to publish
        """
        # Publish the feature table to the online store
        self.fe.publish_table(
            online_store=online_store,
            source_table_name=table_name,
            online_table_name=f"{table_name}_online",
        )

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True, wait: bool = False
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks.

        :param version: Version of the model to deploy
        :param workload_size: Workload size (number of concurrent requests)
        :param scale_to_zero: If True, endpoint scales to 0 when unused
        :param wait: If True, wait for endpoint creation/update to finish
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            if wait:
                self.workspace.serving_endpoints.create_and_wait(
                    name=self.endpoint_name,
                    config=EndpointCoreConfigInput(
                        served_entities=served_entities,
                    ),
                )
            else:
                self.workspace.serving_endpoints.create(
                    name=self.endpoint_name,
                    config=EndpointCoreConfigInput(
                        served_entities=served_entities,
                    ),
                )
        else:
            if wait:
                self.workspace.serving_endpoints.update_config_and_wait(
                    name=self.endpoint_name,
                    served_entities=served_entities,
                )
            else:
                self.workspace.serving_endpoints.update_config(
                    name=self.endpoint_name,
                    served_entities=served_entities,
                )