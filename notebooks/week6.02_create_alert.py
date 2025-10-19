# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the percentage of F1 Score being lower than threshold

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()

# Alert query for financial complaints model
# Monitor F1 score performance
alert_query = """
SELECT
  (COUNT(CASE WHEN f1_score < 0.7 THEN 1 END) * 100.0 /
   COUNT(CASE WHEN f1_score IS NOT NULL AND NOT isnan(f1_score) THEN 1 END))
   AS percentage_below_threshold
FROM mlops_dev.peyman21.model_monitoring_profile_metrics
"""

query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f'financial-complaints-alert-query-{time.time_ns()}',
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on financial complaints model F1 score",
        query_text=alert_query
    )
)

alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(
                column=sql.AlertOperandColumn(name="percentage_below_threshold")
            ),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(
                value=sql.AlertOperandValue(double_value=30)
            )
        ),
        display_name=f'financial-complaints-f1-alert-{time.time_ns()}',
        query_id=query.id
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative: Monitor Prediction Drift

# COMMAND ----------

# Alert query for prediction drift
# Monitors if the average prediction probability is drifting
drift_alert_query = """
SELECT
  AVG(prediction) AS avg_prediction,
  STDDEV(prediction) AS stddev_prediction
FROM mlops_dev.peyman21.model_monitoring
WHERE timestamp >= current_timestamp() - INTERVAL 1 HOUR
"""

drift_query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f'financial-complaints-drift-query-{time.time_ns()}',
        warehouse_id=srcs[0].warehouse_id,
        description="Monitor prediction drift in complaints model",
        query_text=drift_alert_query
    )
)

drift_alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(
                column=sql.AlertOperandColumn(name="avg_prediction")
            ),
            op=sql.AlertOperator.LESS_THAN,
            threshold=sql.AlertConditionThreshold(
                value=sql.AlertOperandValue(double_value=0.3)
            )
        ),
        display_name=f'financial-complaints-drift-alert-{time.time_ns()}',
        query_id=drift_query.id
    )
)

# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)
w.queries.delete(id=drift_query.id)
w.alerts.delete(id=drift_alert.id)
