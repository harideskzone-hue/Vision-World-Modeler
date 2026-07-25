# object_tracker/entity_registry.py
import uuid
from typing import Dict, Optional, Set

class EntityRegistry:
    """
    Single source of truth for stable entity IDs across frames.
    """
    def __init__(self):
        self._name_to_id: Dict[str, str] = {}
        self._id_to_name: Dict[str, str] = {}
        self._archived_ids: Set[str] = set()

    def generate_id(self, base_name: str) -> str:
        """Creates a new stable ID for an entity."""
        unique_id = f"{base_name}_{uuid.uuid4().hex[:6]}"
        self._name_to_id[base_name] = unique_id
        self._id_to_name[unique_id] = base_name
        return unique_id

    def register_id(self, base_name: str, specific_id: str) -> None:
        """Explicitly register an ID (useful for disambiguation)."""
        self._name_to_id[base_name] = specific_id
        self._id_to_name[specific_id] = base_name

    def get_id(self, base_name: str) -> Optional[str]:
        """Looks up the stable ID for a given name."""
        return self._name_to_id.get(base_name)

    def archive(self, entity_id: str) -> None:
        """Archives an ID so it isn't matched by default."""
        self._archived_ids.add(entity_id)

    def reactivate(self, entity_id: str) -> None:
        """Reactivates an archived ID."""
        if entity_id in self._archived_ids:
            self._archived_ids.remove(entity_id)

    def is_archived(self, entity_id: str) -> bool:
        """Checks if an ID is currently archived."""
        return entity_id in self._archived_ids

    def get_name(self, entity_id: str) -> Optional[str]:
        """Looks up the base name for a given stable ID."""
        return self._id_to_name.get(entity_id)
