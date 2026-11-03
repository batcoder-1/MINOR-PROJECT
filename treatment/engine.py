"""
Rule-based treatment recommendation engine (decision support, not prescription).

All advisory text lives in ``knowledge_base.json``; this module only contains
the decision logic that selects which parts of a knowledge-base entry are
returned for a given (disease, confidence, severity) input:

    unknown class           -> generic fallback, expert consultation
    healthy                 -> preventive practices only
    confidence < threshold  -> no disease-specific management, no chemical
                               guidance; ask for a clearer image / expert
    systemic disease        -> (viral, HLB, esca) no curative chemical; vector
                               management, removal of infected plants
    Mild                    -> cultural practices + monitoring + prevention
    Moderate                -> + disease-specific management guidance
                               + generic chemical guidance (expert review)
    Severe                  -> + removal of heavily infected material
                               + expert consultation
    severity unknown        -> moderate-level guidance with a field-inspection caveat

The engine never produces product doses, rates or spray intervals; the
knowledge base does not contain any.
"""

import json
import os

from labels import display_crop_name, display_disease_name, is_healthy

KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_base.json')

SEVERITY_LEVELS = ('mild', 'moderate', 'severe')

# rule identifiers reported in the response so every decision is traceable
RULE_UNKNOWN_DISEASE = 'unknown_disease_fallback'
RULE_HEALTHY = 'healthy_preventive_only'
RULE_LOW_CONFIDENCE = 'low_confidence_suppression'
RULE_SYSTEMIC = 'systemic_no_curative_chemical'
RULE_MILD = 'severity_mild'
RULE_MODERATE = 'severity_moderate'
RULE_SEVERE = 'severity_severe'
RULE_SEVERITY_UNKNOWN = 'severity_unknown'

CHEM_NOT_INDICATED = 'not_indicated'
CHEM_SUPPRESSED = 'suppressed_low_confidence'
CHEM_NO_CURATIVE = 'no_curative_chemical'
CHEM_ADVISORY = 'advisory_requires_expert_review'


class KnowledgeBaseError(ValueError):
    pass


def load_knowledge_base(path=KB_PATH):
    with open(path, 'r', encoding='utf-8') as fh:
        kb = json.load(fh)
    for key in ('metadata', 'disclaimer', 'decision_rules', 'diseases'):
        if key not in kb:
            raise KnowledgeBaseError('knowledge base is missing top-level key %r' % key)
    return kb


def normalise_severity(category):
    """'Moderate' -> 'moderate'; anything that is not a band ('Unknown', None, 'Not applicable') -> None."""
    if not category:
        return None
    value = str(category).strip().lower()
    return value if value in SEVERITY_LEVELS else None


class TreatmentEngine:
    def __init__(self, kb_path=KB_PATH, low_confidence_threshold=None):
        self.kb = load_knowledge_base(kb_path)
        rules = self.kb['decision_rules']
        self.messages = rules['messages']
        self.generic = rules['generic_practices']
        self.low_confidence_threshold = float(
            low_confidence_threshold if low_confidence_threshold is not None
            else rules['low_confidence_threshold'])

    # ------------------------------------------------------------------ lookup
    def lookup(self, class_label):
        """Knowledge-base entry for a class label, or None."""
        return self.kb['diseases'].get(class_label)

    def coverage(self, class_labels):
        """(covered, missing) lists for the given labels."""
        covered = [c for c in class_labels if c in self.kb['diseases']]
        missing = [c for c in class_labels if c not in self.kb['diseases']]
        return covered, missing

    # --------------------------------------------------------------- recommend
    def recommend(self, class_label, confidence=None, severity_category=None, severity_status=None):
        """Build the advisory for one prediction.

        ``confidence``        : classifier confidence in percent (0-100)
        ``severity_category`` : 'Mild' | 'Moderate' | 'Severe' | 'Unknown' | 'Not applicable' | None
        ``severity_status``   : status string from the severity estimator (informational)
        """
        entry = self.lookup(class_label)
        low_conf = confidence is not None and float(confidence) < self.low_confidence_threshold
        severity = normalise_severity(severity_category)
        out = self._skeleton(class_label, entry, confidence, low_conf, severity_category, severity_status)

        if entry is None:
            return self._unknown(out)
        if is_healthy(class_label) or entry.get('pathogen_type') == 'none':
            return self._healthy(out, entry, low_conf)
        if low_conf:
            return self._low_confidence(out, entry)
        if entry.get('no_curative_chemical') or entry.get('pathogen_type') == 'viral':
            return self._systemic(out, entry)
        return self._by_severity(out, entry, severity)

    # ----------------------------------------------------------------- helpers
    def _skeleton(self, class_label, entry, confidence, low_conf, severity_category, severity_status):
        return {
            'class_label': class_label,
            'crop': entry['crop'] if entry else display_crop_name(class_label),
            'disease': entry['disease_name'] if entry else display_disease_name(class_label),
            'found_in_knowledge_base': entry is not None,
            'disease_info': None,
            'decision': {
                'confidence': confidence,
                'low_confidence': low_conf,
                'low_confidence_threshold': self.low_confidence_threshold,
                'severity_category': severity_category,
                'severity_status': severity_status,
                'rules_applied': [],
            },
            'cultural_practices': [],
            'preventive_measures': [],
            'monitoring': [],
            'management_guidance': [],
            'vector_management': [],
            'chemical_guidance': None,
            'expert_consultation': {'recommended': False, 'reason': None},
            'sources': list(entry.get('sources', [])) if entry else [],
            'review_status': (entry or {}).get('review_status', self.kb['metadata'].get('review_status')),
            'disclaimer': self.kb['disclaimer'],
            'warnings': [],
        }

    def _disease_info(self, entry, tentative=False):
        return {
            'pathogen': entry.get('pathogen'),
            'pathogen_type': entry.get('pathogen_type'),
            'symptoms': list(entry.get('symptoms', [])),
            'vector': entry.get('vector'),
            'tentative': tentative,
        }

    def _chemical(self, status, summary_key, entry=None, extra_notes=None):
        chem = {
            'status': status,
            'summary': self.messages[summary_key],
            'generic_guidance': [],
            'active_ingredients_named_by_source': [],
            'dosage': None,
            'interval': None,
            'requires_expert_review': True,
            'label_note': self.messages['label_compliance'],
            'notes': list(extra_notes or []),
        }
        if entry and status == CHEM_ADVISORY:
            cg = entry.get('chemical_guidance') or {}
            chem['generic_guidance'] = list(cg.get('generic_guidance', []))
            chem['active_ingredients_named_by_source'] = list(cg.get('active_ingredients_named_by_source', []))
            if cg.get('note'):
                chem['notes'].append(cg['note'])
            if chem['active_ingredients_named_by_source']:
                chem['notes'].append(self.messages['active_ingredient_caveat'])
        elif entry and status == CHEM_NO_CURATIVE:
            cg = entry.get('chemical_guidance') or {}
            chem['generic_guidance'] = list(cg.get('generic_guidance', []))
        return chem

    def _unknown(self, out):
        out['decision']['rules_applied'].append(RULE_UNKNOWN_DISEASE)
        out['cultural_practices'] = list(self.generic['unknown_disease'])
        out['monitoring'] = list(self.generic['monitoring'])
        out['expert_consultation'] = {'recommended': True, 'reason': self.messages['unknown_disease']}
        out['warnings'].append(self.messages['unknown_disease'])
        return out

    def _healthy(self, out, entry, low_conf):
        out['decision']['rules_applied'].append(RULE_HEALTHY)
        out['preventive_measures'] = list(entry.get('preventive_practices', []))
        out['monitoring'] = list(self.generic['monitoring'])
        out['chemical_guidance'] = self._chemical(CHEM_NOT_INDICATED, 'chemical_not_indicated_healthy')
        if low_conf:
            out['decision']['rules_applied'].append(RULE_LOW_CONFIDENCE)
            out['warnings'].append(self.messages['low_confidence_healthy'])
            out['expert_consultation'] = {'recommended': True, 'reason': self.messages['low_confidence']}
        return out

    def _low_confidence(self, out, entry):
        out['decision']['rules_applied'].append(RULE_LOW_CONFIDENCE)
        out['disease_info'] = self._disease_info(entry, tentative=True)
        out['cultural_practices'] = list(self.generic['low_confidence'])
        out['monitoring'] = list(self.generic['monitoring'])
        out['chemical_guidance'] = self._chemical(CHEM_SUPPRESSED, 'chemical_suppressed_low_confidence')
        out['expert_consultation'] = {'recommended': True, 'reason': self.messages['low_confidence']}
        out['warnings'].append(self.messages['low_confidence'])
        return out

    def _systemic(self, out, entry):
        out['decision']['rules_applied'].append(RULE_SYSTEMIC)
        out['disease_info'] = self._disease_info(entry)
        out['cultural_practices'] = list(entry.get('cultural_practices', []))
        out['preventive_measures'] = list(entry.get('preventive_practices', []))
        out['vector_management'] = list(entry.get('vector_management', []))
        out['management_guidance'] = list(entry.get('management', []))
        out['monitoring'] = list(self.generic['monitoring'])
        out['chemical_guidance'] = self._chemical(CHEM_NO_CURATIVE, 'no_curative_chemical', entry)
        out['expert_consultation'] = {'recommended': True, 'reason': self.messages['expert_systemic']}
        return out

    def _by_severity(self, out, entry, severity):
        out['disease_info'] = self._disease_info(entry)
        out['cultural_practices'] = list(entry.get('cultural_practices', []))
        out['preventive_measures'] = list(entry.get('preventive_practices', []))
        out['monitoring'] = list(self.generic['monitoring'])
        recs = entry.get('severity_recommendations', {})

        if severity == 'mild':
            out['decision']['rules_applied'].append(RULE_MILD)
            out['management_guidance'] = list(recs.get('mild', []))
            out['chemical_guidance'] = self._chemical(CHEM_NOT_INDICATED, 'chemical_not_indicated_mild')
        elif severity == 'moderate':
            out['decision']['rules_applied'].append(RULE_MODERATE)
            out['management_guidance'] = list(recs.get('moderate', []))
            out['chemical_guidance'] = self._chemical(CHEM_ADVISORY, 'chemical_advisory', entry)
        elif severity == 'severe':
            out['decision']['rules_applied'].append(RULE_SEVERE)
            out['management_guidance'] = list(recs.get('severe', [])) + list(self.generic['severe_removal'])
            out['chemical_guidance'] = self._chemical(CHEM_ADVISORY, 'chemical_advisory', entry)
            out['expert_consultation'] = {'recommended': True, 'reason': self.messages['expert_severe']}
        else:
            out['decision']['rules_applied'].append(RULE_SEVERITY_UNKNOWN)
            out['management_guidance'] = list(recs.get('moderate', []))
            out['chemical_guidance'] = self._chemical(CHEM_ADVISORY, 'chemical_advisory', entry,
                                                      extra_notes=[self.messages['severity_unknown']])
            out['expert_consultation'] = {'recommended': True, 'reason': self.messages['severity_unknown']}
            out['warnings'].append(self.messages['severity_unknown'])
        return out
