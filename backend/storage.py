from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .csv_io import read_cues, write_cues
from .models import (
    AppConfig,
    Cue,
    ProjectCreate,
    ProjectManifest,
    ProjectUpdate,
    utc_now,
)


class ProjectNotFoundError(KeyError):
    pass


class CueNotFoundError(KeyError):
    pass


class DuplicateCueError(ValueError):
    pass


class Storage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self.projects_dir = self.data_dir / "projects"
        self.config_path = self.data_dir / "config.json"
        self._lock = RLock()
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        if not project_id or any(char not in "0123456789abcdef-" for char in project_id):
            raise ProjectNotFoundError(project_id)
        path = (self.projects_dir / project_id).resolve()
        if self.projects_dir not in path.parents:
            raise ProjectNotFoundError(project_id)
        return path

    def _manifest_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "manifest.json"

    def _cues_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "source.csv"

    def _translated_cues_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "translated.csv"

    def _ocr_cues_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "ocr.csv"

    def _speech_cues_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "speech.csv"

    def source_csv_path(self, project_id: str) -> Path:
        self.get_project(project_id)
        return self._cues_path(project_id)

    def translated_csv_path(self, project_id: str) -> Path:
        self.get_project(project_id)
        return self._translated_cues_path(project_id)

    def ocr_csv_path(self, project_id: str) -> Path:
        self.get_project(project_id)
        return self._ocr_cues_path(project_id)

    def speech_csv_path(self, project_id: str) -> Path:
        self.get_project(project_id)
        return self._speech_cues_path(project_id)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)

    def list_projects(self) -> list[ProjectManifest]:
        projects: list[ProjectManifest] = []
        for manifest_path in self.projects_dir.glob("*/manifest.json"):
            try:
                projects.append(self._read_manifest(manifest_path))
            except (ValueError, OSError):
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def create_project(self, data: ProjectCreate) -> ProjectManifest:
        with self._lock:
            project_id = str(uuid4())
            manifest = ProjectManifest(project_id=project_id, **data.model_dump())
            self._write_json(
                self._manifest_path(project_id), manifest.model_dump(mode="json")
            )
            write_cues(self._cues_path(project_id), [])
            return manifest

    def _read_manifest(self, path: Path) -> ProjectManifest:
        with path.open("r", encoding="utf-8") as handle:
            return ProjectManifest.model_validate(json.load(handle))

    def get_project(self, project_id: str) -> ProjectManifest:
        path = self._manifest_path(project_id)
        if not path.exists():
            raise ProjectNotFoundError(project_id)
        return self._read_manifest(path)

    def update_project(
        self, project_id: str, data: ProjectUpdate
    ) -> ProjectManifest:
        with self._lock:
            current = self.get_project(project_id)
            updates = data.model_dump(exclude_unset=True)
            updates["updated_at"] = utc_now()
            updated = current.model_copy(update=updates)
            self._write_json(
                self._manifest_path(project_id), updated.model_dump(mode="json")
            )
            return updated

    def delete_project(self, project_id: str) -> None:
        with self._lock:
            path = self._project_dir(project_id)
            if not (path / "manifest.json").exists():
                raise ProjectNotFoundError(project_id)
            shutil.rmtree(path)

    def reset_project_workspace(self, project_id: str) -> ProjectManifest:
        """Remove generated artifacts while preserving the registered source video."""
        with self._lock:
            project = self.get_project(project_id)
            project_dir = self._project_dir(project_id)
            for directory_name in ("cache", "exports"):
                directory = project_dir / directory_name
                if directory.is_dir():
                    shutil.rmtree(directory)
            for path in (
                self._cues_path(project_id),
                self._translated_cues_path(project_id),
                self._ocr_cues_path(project_id),
                self._speech_cues_path(project_id),
            ):
                path.unlink(missing_ok=True)
            write_cues(self._cues_path(project_id), [])
            project.updated_at = utc_now()
            self._write_json(
                self._manifest_path(project_id), project.model_dump(mode="json")
            )
            return project

    def list_cues(self, project_id: str) -> list[Cue]:
        self.get_project(project_id)
        return read_cues(self._cues_path(project_id))

    def list_translated_cues(self, project_id: str) -> list[Cue]:
        self.get_project(project_id)
        path = self._translated_cues_path(project_id)
        return read_cues(path) if path.exists() else []

    def list_ocr_cues(self, project_id: str) -> list[Cue]:
        self.get_project(project_id)
        path = self._ocr_cues_path(project_id)
        return read_cues(path) if path.exists() else []

    def list_speech_cues(self, project_id: str) -> list[Cue]:
        self.get_project(project_id)
        path = self._speech_cues_path(project_id)
        return read_cues(path) if path.exists() else []

    @staticmethod
    def _validate_unique_cues(cues: list[Cue]) -> None:
        cue_ids = [cue.cue_id for cue in cues]
        if len(cue_ids) != len(set(cue_ids)):
            raise DuplicateCueError("cue_id values must be unique")

    def save_ocr_cues(
        self, project_id: str, cues: list[Cue], *, promote: bool = True
    ) -> list[Cue]:
        with self._lock:
            self.get_project(project_id)
            self._validate_unique_cues(cues)
            write_cues(self._ocr_cues_path(project_id), cues)
            if promote:
                write_cues(self._cues_path(project_id), cues)
                self._translated_cues_path(project_id).unlink(missing_ok=True)
            self._touch_project(project_id)
            return read_cues(self._ocr_cues_path(project_id))

    def save_speech_cues(
        self, project_id: str, cues: list[Cue], *, promote: bool = True
    ) -> list[Cue]:
        with self._lock:
            self.get_project(project_id)
            self._validate_unique_cues(cues)
            write_cues(self._speech_cues_path(project_id), cues)
            if promote:
                write_cues(self._cues_path(project_id), cues)
                self._translated_cues_path(project_id).unlink(missing_ok=True)
            self._touch_project(project_id)
            return read_cues(self._speech_cues_path(project_id))

    def save_translated_cues(self, project_id: str, cues: list[Cue]) -> list[Cue]:
        with self._lock:
            self.get_project(project_id)
            source_ids = {cue.cue_id for cue in self.list_cues(project_id)}
            translated_ids = {cue.cue_id for cue in cues}
            if source_ids != translated_ids:
                raise ValueError("translated cues must match source cue IDs")
            write_cues(self._translated_cues_path(project_id), cues)
            self._touch_project(project_id)
            return read_cues(self._translated_cues_path(project_id))

    def replace_cues(self, project_id: str, cues: list[Cue]) -> list[Cue]:
        with self._lock:
            self.get_project(project_id)
            self._validate_unique_cues(cues)
            write_cues(self._cues_path(project_id), cues)
            self._translated_cues_path(project_id).unlink(missing_ok=True)
            self._touch_project(project_id)
            return read_cues(self._cues_path(project_id))

    def create_cue(self, project_id: str, cue: Cue) -> Cue:
        with self._lock:
            cues = self.list_cues(project_id)
            if any(item.cue_id == cue.cue_id for item in cues):
                raise DuplicateCueError(cue.cue_id)
            translated_path = self._translated_cues_path(project_id)
            translated_cues = read_cues(translated_path) if translated_path.exists() else []
            source_cue = (
                cue.model_copy(update={"target_text": ""})
                if translated_path.exists()
                else cue
            )
            cues.append(source_cue)
            write_cues(self._cues_path(project_id), cues)
            if translated_path.exists():
                translated_cues.append(cue)
                write_cues(translated_path, translated_cues)
            self._touch_project(project_id)
            return cue

    def update_cue(self, project_id: str, cue_id: str, cue: Cue) -> Cue:
        if cue.cue_id != cue_id:
            raise ValueError("path cue_id must match body cue_id")
        with self._lock:
            cues = self.list_cues(project_id)
            for index, current in enumerate(cues):
                if current.cue_id == cue_id:
                    translated_path = self._translated_cues_path(project_id)
                    translated_cues = read_cues(translated_path) if translated_path.exists() else []
                    cues[index] = (
                        cue.model_copy(update={"target_text": current.target_text})
                        if translated_path.exists()
                        else cue
                    )
                    write_cues(self._cues_path(project_id), cues)
                    if translated_path.exists():
                        translated_index = next(
                            (
                                translated_index
                                for translated_index, translated in enumerate(translated_cues)
                                if translated.cue_id == cue_id
                            ),
                            None,
                        )
                        if translated_index is not None:
                            translated_cues[translated_index] = cue
                            write_cues(translated_path, translated_cues)
                    self._touch_project(project_id)
                    return cue
            raise CueNotFoundError(cue_id)

    def update_cue_color(self, project_id: str, cue_id: str, color: str) -> Cue:
        with self._lock:
            source_cues = self.list_cues(project_id)
            source_index = next(
                (index for index, cue in enumerate(source_cues) if cue.cue_id == cue_id),
                None,
            )
            if source_index is None:
                raise CueNotFoundError(cue_id)

            source_cues[source_index] = source_cues[source_index].model_copy(
                update={"speaker_color": color}
            )
            visible_cue = source_cues[source_index]

            translated_path = self._translated_cues_path(project_id)
            translated_cues = read_cues(translated_path) if translated_path.exists() else []
            translated_index = next(
                (
                    index
                    for index, cue in enumerate(translated_cues)
                    if cue.cue_id == cue_id
                ),
                None,
            )
            if translated_index is not None:
                translated_cues[translated_index] = translated_cues[
                    translated_index
                ].model_copy(update={"speaker_color": color})
                visible_cue = translated_cues[translated_index]

            write_cues(self._cues_path(project_id), source_cues)
            if translated_path.exists():
                write_cues(translated_path, translated_cues)
            self._touch_project(project_id)
            return visible_cue

    def update_speaker_color(
        self,
        project_id: str,
        *,
        speaker_id: str,
        speaker_name: str,
        color: str,
    ) -> list[Cue]:
        normalized_id = speaker_id.strip()
        normalized_name = speaker_name.strip().casefold()

        def matches(cue: Cue) -> bool:
            if normalized_id:
                return cue.speaker_id == normalized_id
            return bool(normalized_name and cue.speaker_name.strip().casefold() == normalized_name)

        with self._lock:
            source_cues = self.list_cues(project_id)
            matched_ids = {cue.cue_id for cue in source_cues if matches(cue)}
            if not matched_ids:
                raise CueNotFoundError(normalized_id or speaker_name)
            source_cues = [
                cue.model_copy(update={"speaker_color": color})
                if cue.cue_id in matched_ids
                else cue
                for cue in source_cues
            ]
            write_cues(self._cues_path(project_id), source_cues)

            translated_path = self._translated_cues_path(project_id)
            translated_cues = read_cues(translated_path) if translated_path.exists() else []
            if translated_path.exists():
                translated_cues = [
                    cue.model_copy(update={"speaker_color": color})
                    if cue.cue_id in matched_ids
                    else cue
                    for cue in translated_cues
                ]
                write_cues(translated_path, translated_cues)
            self._touch_project(project_id)
            visible = translated_cues if translated_path.exists() else source_cues
            return [cue for cue in visible if cue.cue_id in matched_ids]

    def delete_cue(self, project_id: str, cue_id: str) -> None:
        with self._lock:
            cues = self.list_cues(project_id)
            remaining = [cue for cue in cues if cue.cue_id != cue_id]
            if len(remaining) == len(cues):
                raise CueNotFoundError(cue_id)
            write_cues(self._cues_path(project_id), remaining)
            translated_path = self._translated_cues_path(project_id)
            if translated_path.exists():
                translated = [
                    cue for cue in read_cues(translated_path) if cue.cue_id != cue_id
                ]
                write_cues(translated_path, translated)
            self._touch_project(project_id)

    def _touch_project(self, project_id: str) -> None:
        manifest = self.get_project(project_id)
        manifest.updated_at = utc_now()
        self._write_json(
            self._manifest_path(project_id), manifest.model_dump(mode="json")
        )

    def get_config(self) -> AppConfig:
        if not self.config_path.exists():
            return AppConfig()
        with self.config_path.open("r", encoding="utf-8") as handle:
            return AppConfig.model_validate(json.load(handle))

    def set_config(self, config: AppConfig) -> AppConfig:
        with self._lock:
            self._write_json(self.config_path, config.model_dump(mode="json"))
            return config
