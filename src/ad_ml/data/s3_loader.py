"""S3 data loading utilities for partitioned Parquet datasets."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import boto3
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from botocore.config import Config

from ad_ml.config.settings import S3Config

logger = logging.getLogger(__name__)


class S3DataLoader:
    """Load partitioned Parquet data from S3 with date range filtering and incremental support."""

    def __init__(self, config: S3Config) -> None:
        self.config = config
        self._client = self._build_client()

    def _build_client(self) -> boto3.client:  # type: ignore[type-arg]
        kwargs: Dict[str, object] = {
            "region_name": self.config.region,
            "config": Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        }
        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url
        if self.config.access_key_id and self.config.secret_access_key:
            kwargs["aws_access_key_id"] = self.config.access_key_id
            kwargs["aws_secret_access_key"] = self.config.secret_access_key
        return boto3.client("s3", **kwargs)  # type: ignore[call-overload]

    def _date_partitions(self, start: date, end: date) -> List[str]:
        """Generate s3 partition path suffixes for a date range."""
        partitions: List[str] = []
        current = start
        while current <= end:
            partitions.append(f"year={current.year}/month={current.month:02d}/day={current.day:02d}")
            current += timedelta(days=1)
        return partitions

    def _list_partition_keys(self, prefix: str, partition: str) -> List[str]:
        """List all Parquet file keys under a partition prefix."""
        full_prefix = f"{prefix}{partition}/"
        paginator = self._client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith(".parquet"):
                    keys.append(key)
        return keys

    def load_events(
        self,
        start_date: date,
        end_date: date,
        columns: Optional[List[str]] = None,
        filters: Optional[List[object]] = None,
    ) -> pd.DataFrame:
        """Load user event data for a date range from S3."""
        s3_paths = self._collect_s3_paths(self.config.event_prefix, start_date, end_date)
        if not s3_paths:
            logger.warning("No event partitions found for range %s to %s", start_date, end_date)
            return pd.DataFrame()
        return self._read_parquet_files(s3_paths, columns=columns, filters=filters)

    def load_campaigns(
        self,
        start_date: date,
        end_date: date,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load campaign performance data for a date range from S3."""
        s3_paths = self._collect_s3_paths(self.config.campaign_prefix, start_date, end_date)
        if not s3_paths:
            logger.warning("No campaign partitions found for range %s to %s", start_date, end_date)
            return pd.DataFrame()
        return self._read_parquet_files(s3_paths, columns=columns)

    def _collect_s3_paths(self, prefix: str, start_date: date, end_date: date) -> List[str]:
        partitions = self._date_partitions(start_date, end_date)
        paths: List[str] = []
        for partition in partitions:
            keys = self._list_partition_keys(prefix, partition)
            paths.extend(f"s3://{self.config.bucket}/{k}" for k in keys)
        return paths

    def _read_parquet_files(
        self,
        s3_paths: List[str],
        columns: Optional[List[str]] = None,
        filters: Optional[List[object]] = None,
    ) -> pd.DataFrame:
        """Read multiple Parquet files into a single DataFrame using PyArrow."""
        dataset = ds.dataset(
            s3_paths,
            format="parquet",
            filesystem=self._get_pyarrow_fs(),
        )
        table = dataset.to_table(columns=columns, filter=filters)
        return table.to_pandas()  # type: ignore[no-any-return]

    def _get_pyarrow_fs(self) -> object:
        """Return a PyArrow S3FileSystem with the current credentials."""
        import pyarrow.fs as pafs

        kwargs: Dict[str, object] = {"region": self.config.region}
        if self.config.endpoint_url:
            kwargs["endpoint_override"] = self.config.endpoint_url
        if self.config.access_key_id:
            kwargs["access_key"] = self.config.access_key_id
        if self.config.secret_access_key:
            kwargs["secret_key"] = self.config.secret_access_key
        return pafs.S3FileSystem(**kwargs)  # type: ignore[call-overload]

    def stream_events_daily(
        self,
        start_date: date,
        end_date: date,
        columns: Optional[List[str]] = None,
    ) -> Iterator[tuple[date, pd.DataFrame]]:
        """Yield (date, DataFrame) tuples one day at a time for memory-efficient processing."""
        current = start_date
        while current <= end_date:
            df = self.load_events(current, current, columns=columns)
            yield current, df
            current += timedelta(days=1)

    def upload_artifact(self, local_path: Path, s3_key: str) -> None:
        """Upload a local file to S3."""
        self._client.upload_file(str(local_path), self.config.bucket, s3_key)
        logger.info("Uploaded %s to s3://%s/%s", local_path, self.config.bucket, s3_key)

    def download_artifact(self, s3_key: str, local_path: Path) -> None:
        """Download a file from S3 to local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.config.bucket, s3_key, str(local_path))
        logger.info("Downloaded s3://%s/%s to %s", self.config.bucket, s3_key, local_path)
