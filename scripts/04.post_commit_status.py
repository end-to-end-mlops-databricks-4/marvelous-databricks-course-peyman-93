"""Post Commit Status to GitHub.

This script posts the status of a Databricks job run back to GitHub as a commit status check.
It's typically used in CI/CD pipelines to report integration test results back to GitHub PRs.

The script posts a status to GitHub's Status API which appears in pull requests and commits.

Usage:
    python 04.post_commit_status.py \
        success \
        --git_sha abc123def456 \
        --job_id 123456789 \
        --job_run_id 987654321 \
        --org your-org \
        --repo your-repo

Environment Variables Required:
    TOKEN_STATUS_CHECK: GitHub personal access token with repo:status scope
"""

import argparse
import os

import requests
from loguru import logger
from pyspark.sql import SparkSession


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post commit status to GitHub from Databricks job"
    )

    # Positional argument for status
    parser.add_argument(
        "status",
        type=str,
        choices=["success", "failure", "pending", "error"],
        help="Status to post (success, failure, pending, or error)",
    )

    parser.add_argument(
        "--git_sha",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Git commit SHA to post status to",
    )

    parser.add_argument(
        "--job_id",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Databricks job ID",
    )

    parser.add_argument(
        "--job_run_id",
        action="store",
        default=None,
        type=str,
        required=True,
        help="Databricks job run ID",
    )

    parser.add_argument(
        "--org",
        action="store",
        default=None,
        type=str,
        required=True,
        help="GitHub organization or user name",
    )

    parser.add_argument(
        "--repo",
        action="store",
        default=None,
        type=str,
        required=True,
        help="GitHub repository name",
    )

    parser.add_argument(
        "--context",
        action="store",
        default="integration-testing/databricks",
        type=str,
        help="Context label for the status check (default: integration-testing/databricks)",
    )

    return parser.parse_args()


def get_status_description(status: str) -> str:
    """Get human-readable description for each status.

    :param status: Status type (success, failure, pending, error)
    :return: Human-readable description
    """
    descriptions = {
        "success": "Integration test completed successfully!",
        "failure": "Integration test failed. Check Databricks logs for details.",
        "pending": "Integration test is running...",
        "error": "Integration test encountered an error.",
    }
    return descriptions.get(status, "Integration test status update")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("POSTING COMMIT STATUS TO GITHUB")
    logger.info("=" * 70)
    logger.info(f"Status: {args.status}")
    logger.info(f"Git SHA: {args.git_sha}")
    logger.info(f"Repository: {args.org}/{args.repo}")
    logger.info(f"Job ID: {args.job_id}")
    logger.info(f"Run ID: {args.job_run_id}")
    logger.info("=" * 70)

    # Get Databricks workspace URL
    spark = SparkSession.builder.getOrCreate()
    host = spark.conf.get("spark.databricks.workspaceUrl")
    logger.info(f"Databricks host: {host}")

    # Extract arguments
    org = args.org
    repo = args.repo
    git_sha = args.git_sha
    job_id = args.job_id
    run_id = args.job_run_id
    status = args.status
    context = args.context

    # Get GitHub token from environment
    try:
        token = os.environ["TOKEN_STATUS_CHECK"]
        logger.info("GitHub token found in environment")
    except KeyError:
        logger.error("TOKEN_STATUS_CHECK environment variable not found!")
        logger.error("Please set TOKEN_STATUS_CHECK with a GitHub personal access token")
        raise

    # Build GitHub API URL
    url = f"https://api.github.com/repos/{org}/{repo}/statuses/{git_sha}"
    logger.info(f"GitHub API URL: {url}")

    # Build link to Databricks run
    link_to_databricks_run = f"https://{host}/jobs/{job_id}/runs/{run_id}"
    logger.info(f"Databricks run URL: {link_to_databricks_run}")

    # Prepare request headers
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Prepare payload
    payload = {
        "state": status,
        "target_url": link_to_databricks_run,
        "description": get_status_description(status),
        "context": context,
    }

    logger.info("\nPayload:")
    logger.info(f"  State: {payload['state']}")
    logger.info(f"  Description: {payload['description']}")
    logger.info(f"  Context: {payload['context']}")
    logger.info(f"  Target URL: {payload['target_url']}")

    # Post status to GitHub
    logger.info("\n" + "=" * 70)
    logger.info("POSTING TO GITHUB API")
    logger.info("=" * 70)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        logger.info(f"✓ Status posted successfully!")
        logger.info(f"  HTTP Status Code: {response.status_code}")
        logger.info(f"  Response: {response.json()}")

        logger.info("\n" + "=" * 70)
        logger.info("COMMIT STATUS POSTED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Status '{status}' posted to GitHub commit {git_sha[:7]}")
        logger.info(f"View in GitHub: https://github.com/{org}/{repo}/commit/{git_sha}")

    except requests.exceptions.HTTPError as e:
        logger.error(f"✗ HTTP Error posting status: {e}")
        logger.error(f"  Status Code: {response.status_code}")
        logger.error(f"  Response: {response.text}")
        raise

    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Request error posting status: {e}")
        raise

    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
