# tests/test_vision_extractor.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision_extractor.scene_parser import SceneParser
from shared.config import DEFAULT_CONFIG

def test_scene_parser():
    parser = SceneParser()
    frame_id = 1
    
    print("Running test_vision_extractor.py...")

    # Case 1: Valid JSON response
    valid_json = """
    {
      "scene": "kitchen",
      "entities": [
        {"name": "wooden_table", "category": "furniture", "state": null, "confidence": 0.95},
        {"name": "red_mug", "category": "container", "state": "empty", "confidence": 0.85}
      ]
    }
    """
    obs = parser.parse(valid_json, frame_id)
    assert obs is not None
    assert obs.scene == "kitchen"
    assert len(obs.entities) == 2
    assert obs.entities[0]["name"] == "wooden_table"
    assert obs.entities[1]["state"] == "empty"
    print("✔ Valid JSON response parsed")

    # Case 2: JSON inside Markdown code fences
    markdown_json = """
    Here is the scene information you requested:
    ```json
    {
      "scene": "office",
      "entities": [
        {"name": "laptop", "category": "electronics", "state": "on", "confidence": 0.99}
      ]
    }
    ```
    I hope this helps!
    """
    obs = parser.parse(markdown_json, frame_id)
    assert obs is not None
    assert obs.scene == "office"
    assert len(obs.entities) == 1
    print("✔ JSON inside Markdown code fences parsed")

    # Case 3: Malformed JSON recovered by regex
    malformed_json = """
    {
      "scene": "garden",
      "entities": [
        {name: "tree", category: "plant", state: null, confidence: 0.90},
        {"name": "bench", "category": "furniture", "state": "dirty", confidence: 0.8}
    """
    obs = parser.parse(malformed_json, frame_id)
    assert obs is not None
    assert obs.scene == "garden"
    assert len(obs.entities) >= 1  # Should salvage at least the bench which has quoted keys
    print("✔ Malformed JSON recovered by regex parsed")

    # Case 4: Empty/invalid model output returning None
    invalid_output = "I'm sorry, I cannot see the image."
    obs = parser.parse(invalid_output, frame_id)
    assert obs is None
    print("✔ Empty/invalid model output returned None")

    # Case 5: Scene with no detectable entities
    no_entities = """
    {
      "scene": "empty_room",
      "entities": []
    }
    """
    obs = parser.parse(no_entities, frame_id)
    assert obs is not None
    assert obs.scene == "empty_room"
    assert len(obs.entities) == 0
    print("✔ Scene with no detectable entities parsed")

    # Case 6: Ambiguous object descriptions (validate names are cleaned)
    ambiguous = """
    {
      "scene": "living room",
      "entities": [
        {"name": "A big Red Sofa", "category": "furniture"}
      ]
    }
    """
    obs = parser.parse(ambiguous, frame_id)
    assert obs is not None
    assert obs.scene == "living room"
    assert obs.entities[0]["name"] == "a_big_red_sofa"
    assert obs.entities[0]["confidence"] == DEFAULT_CONFIG.vision_extractor.base_vlm_confidence
    print("✔ Ambiguous object descriptions cleaned")

    # Case 7: Multiple objects with the same category
    mult_same_cat = """
    {
      "scene": "kitchen",
      "entities": [
        {"name": "mug_1", "category": "container"},
        {"name": "mug_2", "category": "container"}
      ]
    }
    """
    obs = parser.parse(mult_same_cat, frame_id)
    assert obs is not None
    assert len(obs.entities) == 2
    assert obs.entities[0]["category"] == "container"
    assert obs.entities[1]["category"] == "container"
    print("✔ Multiple objects with the same category parsed")

    # Case 8: Missing optional fields
    missing_fields = """
    {
      "scene": "hallway",
      "entities": [
        {"name": "door"}
      ]
    }
    """
    obs = parser.parse(missing_fields, frame_id)
    assert obs is not None
    assert obs.entities[0]["category"] == "unknown"
    assert obs.entities[0]["state"] is None
    assert obs.entities[0]["confidence"] == DEFAULT_CONFIG.vision_extractor.base_vlm_confidence
    print("✔ Missing optional fields handled")

    # Case 9: Large scene (10-20 entities)
    large_scene_entities = ",\n".join([f'{{"name": "item_{i}", "category": "misc"}}' for i in range(15)])
    large_scene = f"""
    {{
      "scene": "warehouse",
      "entities": [
        {large_scene_entities}
      ]
    }}
    """
    obs = parser.parse(large_scene, frame_id)
    assert obs is not None
    assert len(obs.entities) == 15
    print("✔ Large scene parsed")

    print("\nAll Vision Extractor Parser tests passed!")

if __name__ == "__main__":
    test_scene_parser()
