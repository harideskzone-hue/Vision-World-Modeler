# object_tracker/candidate_converter.py
from typing import List
from shared.models import SceneObservation, CandidateFact
from shared.enums import RelationType, ExtractionMethod
from shared.config import DEFAULT_CONFIG

class CandidateConverter:
    """
    Transforms a matched SceneObservation into CandidateFacts for the Updater.
    Does not mutate the graph itself.
    """
    def convert(self, observation: SceneObservation) -> List[CandidateFact]:
        facts = []
        
        # 1. Camera location fact
        facts.append(CandidateFact(
            subject="camera",
            relation=RelationType.LOCATED_IN,
            object=observation.scene,
            confidence=1.0,  # Scene classification is usually high conf
            source_frame_id=observation.frame,
            extraction_method=ExtractionMethod.VLM
        ))

        # 2. Entity facts
        for ent in observation.entities:
            stable_id = ent["name"]
            conf = ent["confidence"]
            
            # Entity located in scene
            facts.append(CandidateFact(
                subject=stable_id,
                relation=RelationType.LOCATED_IN,
                object=observation.scene,
                confidence=conf,
                source_frame_id=observation.frame,
                extraction_method=ExtractionMethod.FUSED if "bbox" in ent else ExtractionMethod.VLM
            ))
            
            # Entity has type
            facts.append(CandidateFact(
                subject=stable_id,
                relation=RelationType.IS_TYPE,
                object=ent["category"],
                confidence=conf,
                source_frame_id=observation.frame,
                extraction_method=ExtractionMethod.VLM
            ))
            
            # Entity state (if present)
            if ent.get("state"):
                facts.append(CandidateFact(
                    subject=stable_id,
                    relation=RelationType.HAS_STATE,
                    object=ent["state"],
                    confidence=conf,
                    source_frame_id=observation.frame,
                    extraction_method=ExtractionMethod.VLM
                ))

        return facts
