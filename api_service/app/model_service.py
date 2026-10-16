from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from api_service.app.config import Settings
from api_service.app.family import (
    FAMILY_KEY_DEFINITION,
    FAMILY_RULE_NAME,
    FAMILY_RULE_VERSION,
    build_family_alerts,
)
from api_service.app.feature_store import FeatureDataNotReadyError, FeatureStore


STATE_NAMES = {0: "normal", 1: "watch", 2: "abnormal", 3: "high_risk"}


class ModelDataError(RuntimeError):
    pass


class ObjectNotFoundError(ModelDataError):
    pass


class AssessmentDateNotFoundError(ModelDataError):
    pass


@dataclass
class LoadedModel:
    model: torch.nn.Module
    preprocessor: Any
    numeric_features: list[str]
    categorical_features: list[str]
    component_keys: list[str]
    component_names: dict[str, str]
    name: str
    version: str
    metrics: dict[str, Any]


class ModelService:
    def __init__(self, settings: Settings, feature_store: FeatureStore) -> None:
        self.settings = settings
        self.feature_store = feature_store
        source_root = str(settings.project_root / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        from griddx.features import build_device_features, build_station_features
        from griddx.model_zoo import DeviceAdaptiveMultiDiscriminator, resolve_torch_device
        from griddx.station_fusion import StationHierarchicalFusionNetwork

        self._build_device_features = build_device_features
        self._build_station_features = build_station_features
        self.device = resolve_torch_device(settings.torch_device, settings.gpu_id)
        self._lock = RLock()
        self._feature_cache_lock = RLock()
        self._feature_cache_version: str | None = None
        self._device_feature_cache: pd.DataFrame | None = None
        self._station_feature_cache: pd.DataFrame | None = None
        self.device_model = self._load_device_model(
            settings.device_model_path, DeviceAdaptiveMultiDiscriminator
        )
        self.station_model = self._load_station_model(
            settings.station_model_path, StationHierarchicalFusionNetwork
        )

    def _load_device_model(self, path: Path, model_class: type[torch.nn.Module]) -> LoadedModel:
        artifact = self._read_artifact(path)
        model = model_class(
            n_features=len(artifact["transformed_feature_names"]),
            n_classes=int(artifact["n_classes"]),
            feature_groups=artifact["feature_groups"],
        )
        model.load_state_dict(artifact["model_state_dict"])
        model.to(self.device).eval()
        keys = list(artifact["feature_groups"])
        return LoadedModel(
            model=model,
            preprocessor=artifact["preprocessor"],
            numeric_features=list(artifact["numeric_features"]),
            categorical_features=list(artifact["categorical_features"]),
            component_keys=keys,
            component_names=dict(artifact.get("discriminator_names", {})),
            name=str(artifact.get("model_name", "DAMD-Net")),
            version=self._hash_file(path),
            metrics=dict(artifact.get("metrics", {})),
        )

    def _load_station_model(self, path: Path, model_class: type[torch.nn.Module]) -> LoadedModel:
        artifact = self._read_artifact(path)
        model = model_class(
            n_features=len(artifact["transformed_feature_names"]),
            n_classes=int(artifact["n_classes"]),
            feature_views=artifact["feature_views"],
        )
        model.load_state_dict(artifact["model_state_dict"])
        model.to(self.device).eval()
        keys = list(artifact["feature_views"])
        return LoadedModel(
            model=model,
            preprocessor=artifact["preprocessor"],
            numeric_features=list(artifact["numeric_features"]),
            categorical_features=list(artifact["categorical_features"]),
            component_keys=keys,
            component_names=dict(artifact.get("view_names", {})),
            name=str(artifact.get("model_name", "HSF-Net")),
            version=self._hash_file(path),
            metrics=dict(artifact.get("metrics", {})),
        )

    @staticmethod
    def _read_artifact(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Trained model artifact does not exist: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:12]

    def assess_devices(
        self,
        assessment_date: date,
        device_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        frame, manifest = self.feature_store.read_device_features()
        features = self._engineered_device_features(frame, str(manifest["version"]))
        predictions = self._device_predictions(
            features,
            start_date=assessment_date,
            end_date=assessment_date,
            object_ids=device_ids,
        )
        records = predictions.drop(columns=["date"]).to_dict(orient="records")
        for record in records:
            record["confidence"] = self._round(record["confidence"])
            record["high_risk_probability"] = self._round(
                record["high_risk_probability"]
            )
            record["discriminator_weights"] = {
                key: self._round(value)
                for key, value in record.pop("component_weights").items()
            }
        return records

    def assess_stations(
        self,
        assessment_date: date,
        station_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        frame, manifest = self.feature_store.read_station_features()
        features = self._engineered_station_features(frame, str(manifest["version"]))
        predictions = self._station_predictions(
            features,
            start_date=assessment_date,
            end_date=assessment_date,
            object_ids=station_ids,
        )
        records = predictions.drop(columns=["date"]).to_dict(orient="records")
        for record in records:
            record["risk_score"] = self._round(record["risk_score"])
            record["confidence"] = self._round(record["confidence"])
            record["high_risk_probability"] = self._round(
                record["high_risk_probability"]
            )
            record["view_weights"] = {
                key: self._round(value)
                for key, value in record.pop("component_weights").items()
            }
        return records

    def device_profile(self, device_id: str, start_date: date, end_date: date) -> dict[str, Any]:
        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date.")
        frame, manifest = self.feature_store.read_device_features()
        features = self._engineered_device_features(frame, str(manifest["version"]))
        values = self._device_predictions(
            features, start_date=start_date, end_date=end_date, object_ids=[device_id]
        )
        weights = self._average_weight_columns(values, "weight_")
        dominant_key = max(weights, key=weights.get)
        return {
            "device_id": device_id,
            "station_id": self._first_text(values["station_id"]),
            "start_date": start_date,
            "end_date": end_date,
            "sample_count": int(len(values)),
            "average_high_risk_probability": self._round(values["high_risk_probability"].mean()),
            "average_confidence": self._round(values["confidence"].mean()),
            "dominant_discriminator": self.device_model.component_names.get(
                dominant_key, dominant_key
            ),
            "average_discriminator_weights": {
                key: self._round(value) for key, value in weights.items()
            },
        }

    def station_profile(self, station_id: str, start_date: date, end_date: date) -> dict[str, Any]:
        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date.")
        frame, manifest = self.feature_store.read_station_features()
        features = self._engineered_station_features(frame, str(manifest["version"]))
        values = self._station_predictions(
            features, start_date=start_date, end_date=end_date, object_ids=[station_id]
        )
        weights = self._average_weight_columns(values, "weight_")
        dominant_key = max(weights, key=weights.get)
        return {
            "station_id": station_id,
            "station_name": self._first_text(values["station_name"]),
            "start_date": start_date,
            "end_date": end_date,
            "sample_count": int(len(values)),
            "average_risk_score": self._round(values["risk_score"].mean()),
            "average_high_risk_probability": self._round(values["high_risk_probability"].mean()),
            "average_confidence": self._round(values["confidence"].mean()),
            "dominant_view": self.station_model.component_names.get(dominant_key, dominant_key),
            "average_view_weights": {
                key: self._round(value) for key, value in weights.items()
            },
        }

    def family_alerts(
        self,
        assessment_date: date,
        *,
        level: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        frame, _manifest = self.feature_store.read_device_features()
        required = {
            "family_id",
            "family_manufacturer",
            "family_model",
            "family_batch",
            "family_batch_source",
            "family_device_count",
            "family_defect_device_count",
            "family_history_defect_count",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise FeatureDataNotReadyError(
                "Family alerts require a full-mode feature version with family identity "
                f"columns; missing: {missing}"
            )
        selected = frame[frame["date"] == pd.Timestamp(assessment_date)].copy()
        if selected.empty:
            raise AssessmentDateNotFoundError(
                f"No device data exists for assessment_date={assessment_date}."
            )
        predictions = self.assess_devices(assessment_date)
        data = build_family_alerts(
            selected,
            predictions,
            assessment_date=assessment_date,
            level=level,
            limit=limit,
        )
        data["device_model_name"] = self.device_model.name
        data["device_model_version"] = self.device_model.version
        return data

    def model_info(self) -> dict[str, Any]:
        try:
            manifest = self.feature_store.get_active_manifest()
        except RuntimeError:
            manifest = None
        active_feature = None
        if manifest is not None:
            active_feature = {
                key: manifest[key]
                for key in (
                    "version",
                    "business_date",
                    "run_mode",
                    "device_rows",
                    "station_rows",
                    "published_at",
                )
            }
        return {
            "inference_device": str(self.device),
            "training_enabled": False,
            "active_feature_version": active_feature,
            "models": [
                {
                    "name": self.device_model.name,
                    "version": self.device_model.version,
                    "purpose": "device_state_assessment",
                    "components": self.device_model.component_names,
                },
                {
                    "name": self.station_model.name,
                    "version": self.station_model.version,
                    "purpose": "station_state_and_risk_assessment",
                    "components": self.station_model.component_names,
                },
                {
                    "name": "RA-MOD",
                    "version": "rule-lp-1",
                    "purpose": "risk_aware_economic_dispatch",
                    "parameter_source": "request_and_config",
                },
                {
                    "name": FAMILY_RULE_NAME,
                    "version": FAMILY_RULE_VERSION,
                    "purpose": "family_defect_alert",
                    "family_key_definition": FAMILY_KEY_DEFINITION,
                },
            ],
        }

    def _device_predictions(
        self,
        features: pd.DataFrame,
        *,
        start_date: date,
        end_date: date,
        object_ids: list[str] | None,
    ) -> pd.DataFrame:
        id_column = "unified_device_id"
        self._validate_raw(features, id_column)
        selected = self._select_rows(
            features, id_column, start_date, end_date, object_ids, "device"
        )
        matrix, quality = self._transform(selected, self.device_model)
        with self._lock, torch.inference_mode():
            logits, weights = self.device_model.model(
                torch.from_numpy(matrix).to(self.device)
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            weights_np = weights.cpu().numpy()
        labels = probabilities.argmax(axis=1)
        result = pd.DataFrame(
            {
                "date": selected["date"].dt.date,
                "device_id": selected[id_column].astype(str).to_numpy(),
                "station_id": self._optional_text_array(selected, "station_id"),
                "state_level": labels.astype(int),
                "state_name": [STATE_NAMES[int(label)] for label in labels],
                "confidence": probabilities.max(axis=1),
                "high_risk_probability": probabilities[:, 2:].sum(axis=1),
                "data_quality": quality,
            }
        )
        self._attach_weights(result, weights_np, self.device_model)
        result["dominant_discriminator"] = [
            self.device_model.component_names.get(key, key)
            for key in result.pop("dominant_component")
        ]
        return self._round_prediction_frame(result)

    def _station_predictions(
        self,
        features: pd.DataFrame,
        *,
        start_date: date,
        end_date: date,
        object_ids: list[str] | None,
    ) -> pd.DataFrame:
        id_column = "station_id"
        self._validate_raw(features, id_column)
        selected = self._select_rows(
            features, id_column, start_date, end_date, object_ids, "station"
        )
        matrix, quality = self._transform(selected, self.station_model)
        with self._lock, torch.inference_mode():
            logits, risk, weights = self.station_model.model(
                torch.from_numpy(matrix).to(self.device)
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            risks = risk.cpu().numpy()
            weights_np = weights.cpu().numpy()
        labels = probabilities.argmax(axis=1)
        result = pd.DataFrame(
            {
                "date": selected["date"].dt.date,
                "station_id": selected[id_column].astype(str).to_numpy(),
                "station_name": self._optional_text_array(selected, "station_name"),
                "state_level": labels.astype(int),
                "state_name": [STATE_NAMES[int(label)] for label in labels],
                "risk_score": 100.0 * risks,
                "confidence": probabilities.max(axis=1),
                "high_risk_probability": probabilities[:, 2:].sum(axis=1),
                "data_quality": quality,
            }
        )
        self._attach_weights(result, weights_np, self.station_model)
        result["dominant_view"] = [
            self.station_model.component_names.get(key, key)
            for key in result.pop("dominant_component")
        ]
        return self._round_prediction_frame(result)

    @staticmethod
    def _validate_raw(frame: pd.DataFrame, id_column: str) -> None:
        missing = {"date", id_column} - set(frame.columns)
        if missing:
            raise ModelDataError(f"Published feature data is missing columns: {sorted(missing)}")

    def _engineered_device_features(
        self, raw: pd.DataFrame, version: str
    ) -> pd.DataFrame:
        with self._feature_cache_lock:
            self._reset_engineered_cache(version)
            if self._device_feature_cache is None:
                self._device_feature_cache = self._build_device_features(raw.copy())
            return self._device_feature_cache

    def _engineered_station_features(
        self, raw: pd.DataFrame, version: str
    ) -> pd.DataFrame:
        with self._feature_cache_lock:
            self._reset_engineered_cache(version)
            if self._station_feature_cache is None:
                self._station_feature_cache = self._build_station_features(raw.copy())
            return self._station_feature_cache

    def _reset_engineered_cache(self, version: str) -> None:
        if self._feature_cache_version == version:
            return
        self._feature_cache_version = version
        self._device_feature_cache = None
        self._station_feature_cache = None

    @staticmethod
    def _select_rows(
        frame: pd.DataFrame,
        id_column: str,
        start_date: date,
        end_date: date,
        object_ids: list[str] | None,
        kind: str,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        selected = frame[frame["date"].between(start, end)].copy()
        if selected.empty:
            raise AssessmentDateNotFoundError(
                f"No {kind} feature rows found from {start_date} to {end_date}."
            )
        if object_ids is not None:
            available = set(selected[id_column].dropna().astype(str))
            missing = [value for value in object_ids if value not in available]
            if missing:
                raise ObjectNotFoundError(
                    f"Unknown {kind} IDs for the requested date range: {missing}"
                )
            selected = selected[selected[id_column].astype(str).isin(object_ids)].copy()
        return selected.sort_values(["date", id_column]).reset_index(drop=True)

    @staticmethod
    def _transform(frame: pd.DataFrame, artifact: LoadedModel) -> tuple[np.ndarray, np.ndarray]:
        expected = artifact.numeric_features + artifact.categorical_features
        values = frame.copy()
        for column in expected:
            if column not in values.columns:
                values[column] = np.nan
        values[artifact.numeric_features] = values[artifact.numeric_features].replace(
            [np.inf, -np.inf], np.nan
        )
        completeness = values[expected].notna().mean(axis=1).to_numpy()
        transformed = artifact.preprocessor.transform(values[expected])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        matrix = np.asarray(transformed, dtype=np.float32)
        quality = np.where(completeness >= 0.60, "pass", "warning")
        return matrix, quality

    @staticmethod
    def _attach_weights(
        result: pd.DataFrame, weights: np.ndarray, artifact: LoadedModel
    ) -> None:
        for index, key in enumerate(artifact.component_keys):
            result[f"weight_{key}"] = weights[:, index]
        dominant_indices = weights.argmax(axis=1)
        result["dominant_component"] = [artifact.component_keys[index] for index in dominant_indices]

    @staticmethod
    def _round_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = frame.select_dtypes(include="number").columns
        float_columns = [column for column in numeric_columns if column != "state_level"]
        frame[float_columns] = frame[float_columns].round(6)
        weight_columns = [column for column in frame.columns if column.startswith("weight_")]
        if weight_columns:
            frame["component_weights"] = frame[weight_columns].apply(
                lambda row: {
                    column.removeprefix("weight_"): float(row[column])
                    for column in weight_columns
                },
                axis=1,
            )
            frame = frame.drop(columns=weight_columns)
        return frame

    @staticmethod
    def _average_weight_columns(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
        # Prediction output stores all component weights in one JSON-like column.
        del prefix
        keys = list(frame.iloc[0]["component_weights"])
        return {
            key: float(np.mean([item[key] for item in frame["component_weights"]]))
            for key in keys
        }

    @staticmethod
    def _optional_text_array(frame: pd.DataFrame, column: str) -> np.ndarray:
        if column not in frame.columns:
            return np.full(len(frame), None, dtype=object)
        return frame[column].map(lambda value: None if pd.isna(value) else str(value)).to_numpy()

    @staticmethod
    def _first_text(values: Iterable[Any]) -> str | None:
        for value in values:
            if value is not None and not pd.isna(value):
                return str(value)
        return None

    @staticmethod
    def _round(value: float) -> float:
        return round(float(value), 6)
