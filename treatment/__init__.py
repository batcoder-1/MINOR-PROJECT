"""
Rule-based treatment recommendation engine backed by knowledge_base.json.

    from treatment import TreatmentEngine
    engine = TreatmentEngine()
    advice = engine.recommend('Tomato___Late_blight', confidence=94.3, severity_category='Moderate')
"""

from .engine import (
    CHEM_ADVISORY, CHEM_NO_CURATIVE, CHEM_NOT_INDICATED, CHEM_SUPPRESSED, KB_PATH,
    RULE_HEALTHY, RULE_LOW_CONFIDENCE, RULE_MILD, RULE_MODERATE, RULE_SEVERE, RULE_SEVERITY_UNKNOWN,
    RULE_SYSTEMIC, RULE_UNKNOWN_DISEASE, KnowledgeBaseError, TreatmentEngine, load_knowledge_base,
    normalise_severity,
)

__all__ = [
    'TreatmentEngine', 'load_knowledge_base', 'normalise_severity', 'KnowledgeBaseError', 'KB_PATH',
    'CHEM_ADVISORY', 'CHEM_NO_CURATIVE', 'CHEM_NOT_INDICATED', 'CHEM_SUPPRESSED',
    'RULE_HEALTHY', 'RULE_LOW_CONFIDENCE', 'RULE_MILD', 'RULE_MODERATE', 'RULE_SEVERE',
    'RULE_SEVERITY_UNKNOWN', 'RULE_SYSTEMIC', 'RULE_UNKNOWN_DISEASE',
]
