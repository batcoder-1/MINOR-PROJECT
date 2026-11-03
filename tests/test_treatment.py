import json
import re

import pytest

from labels import CLASS_LABELS, is_healthy
from treatment import (CHEM_ADVISORY, CHEM_NO_CURATIVE, CHEM_NOT_INDICATED, CHEM_SUPPRESSED, KB_PATH, RULE_HEALTHY,
                       RULE_LOW_CONFIDENCE, RULE_MILD, RULE_MODERATE, RULE_SEVERE, RULE_SEVERITY_UNKNOWN,
                       RULE_SYSTEMIC, RULE_UNKNOWN_DISEASE, KnowledgeBaseError, TreatmentEngine, load_knowledge_base,
                       normalise_severity)

VIRAL = ['Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus']


@pytest.fixture(scope='module')
def engine():
    return TreatmentEngine()


@pytest.fixture(scope='module')
def kb():
    return load_knowledge_base()


# ---------------------------------------------------------------- knowledge base content

def test_every_class_has_an_entry(engine):
    covered, missing = engine.coverage(CLASS_LABELS)
    assert missing == [] and len(covered) == 38


def test_entry_schema(kb):
    for label, entry in kb['diseases'].items():
        assert entry['class_id'] == CLASS_LABELS.index(label), label
        for key in ('crop', 'disease_name', 'pathogen_type', 'symptoms', 'preventive_practices', 'sources'):
            assert key in entry, (label, key)
        assert entry['pathogen_type'] in ('fungal', 'oomycete', 'bacterial', 'viral', 'arthropod_pest', 'none')
        if is_healthy(label):
            assert entry['pathogen_type'] == 'none'
            continue
        assert entry['symptoms'] and entry['cultural_practices'] and entry['preventive_practices']
        if entry.get('no_curative_chemical'):
            assert entry['management'], label
        else:
            assert set(entry['severity_recommendations']) == {'mild', 'moderate', 'severe'}, label
            assert all(entry['severity_recommendations'][k] for k in ('mild', 'moderate', 'severe')), label


def test_every_disease_cites_a_verified_https_source(kb):
    for label, entry in kb['diseases'].items():
        if is_healthy(label) and entry.get('review_status'):
            continue  # generic healthy entries for crops without a disease class are flagged instead
        assert entry['sources'], label
        for src in entry['sources']:
            assert src['url'].startswith('https://') and src['title'] and src['organization'], (label, src)


def test_no_dosages_or_intervals_anywhere(kb):
    blob = json.dumps(kb['diseases']).lower()
    for pattern in (r'\d+\s*(g|ml|kg|l)\s*/\s*(l|ha|acre)', r'\d+\s*ppm', r'per (acre|hectare)', r'\d+\s*%\s*w/v'):
        assert not re.search(pattern, blob), pattern
    for entry in kb['diseases'].values():
        cg = entry.get('chemical_guidance')
        if cg:
            assert cg['dosage'] is None and cg['interval'] is None
            assert cg['verification_status'] == 'requires_expert_review'


def test_viral_entries_have_no_fungicides(kb):
    for label in VIRAL + ['Orange___Haunglongbing_(Citrus_greening)']:
        entry = kb['diseases'][label]
        assert entry['no_curative_chemical'] is True
        names = ' '.join(entry['chemical_guidance']['active_ingredients_named_by_source']).lower()
        for fungicide in ('mancozeb', 'chlorothalonil', 'azoxystrobin', 'copper'):
            assert fungicide not in names, (label, fungicide)


def test_missing_top_level_key_rejected(tmp_path):
    bad = tmp_path / 'kb.json'
    bad.write_text(json.dumps({'metadata': {}, 'diseases': {}}))
    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(str(bad))


# ---------------------------------------------------------------- decision logic

def test_lookup(engine):
    entry = engine.lookup('Potato___Late_blight')
    assert entry['pathogen_type'] == 'oomycete' and entry['class_id'] == 21
    assert engine.lookup('Banana___Sigatoka') is None


@pytest.mark.parametrize('value,expected', [('Mild', 'mild'), (' SEVERE ', 'severe'), ('Unknown', None),
                                            ('Not applicable', None), (None, None)])
def test_normalise_severity(value, expected):
    assert normalise_severity(value) == expected


def test_healthy_gets_preventive_only(engine):
    r = engine.recommend('Tomato___healthy', confidence=97, severity_category='Not applicable')
    assert r['decision']['rules_applied'] == [RULE_HEALTHY]
    assert r['preventive_measures'] and not r['management_guidance'] and not r['cultural_practices']
    assert r['chemical_guidance']['status'] == CHEM_NOT_INDICATED
    assert not r['expert_consultation']['recommended']


def test_low_confidence_healthy_asks_for_confirmation(engine):
    r = engine.recommend('Tomato___healthy', confidence=40)
    assert RULE_LOW_CONFIDENCE in r['decision']['rules_applied'] and r['expert_consultation']['recommended']


@pytest.mark.parametrize('severity', ['Mild', 'Moderate', 'Severe', 'Unknown'])
def test_low_confidence_suppresses_chemical_and_specific_advice(engine, severity):
    r = engine.recommend('Tomato___Late_blight', confidence=45.0, severity_category=severity)
    assert r['decision']['rules_applied'] == [RULE_LOW_CONFIDENCE]
    assert r['chemical_guidance']['status'] == CHEM_SUPPRESSED
    assert r['chemical_guidance']['active_ingredients_named_by_source'] == []
    assert r['management_guidance'] == []
    assert r['disease_info']['tentative'] is True
    assert r['expert_consultation']['recommended']
    assert any('clearer' in w.lower() or 'retake' in w.lower() for w in r['warnings'] + r['cultural_practices'])


def test_low_confidence_threshold_boundary(engine):
    assert engine.recommend('Tomato___Late_blight', 59.99, 'Moderate')['decision']['low_confidence'] is True
    assert engine.recommend('Tomato___Late_blight', 60.0, 'Moderate')['decision']['low_confidence'] is False


def test_custom_threshold():
    strict = TreatmentEngine(low_confidence_threshold=90)
    assert strict.recommend('Tomato___Late_blight', 85, 'Moderate')['chemical_guidance']['status'] == CHEM_SUPPRESSED


def test_mild_is_cultural_and_monitoring_without_chemical(engine):
    r = engine.recommend('Tomato___Early_blight', confidence=95, severity_category='Mild')
    assert r['decision']['rules_applied'] == [RULE_MILD]
    assert r['cultural_practices'] and r['monitoring'] and r['preventive_measures'] and r['management_guidance']
    assert r['chemical_guidance']['status'] == CHEM_NOT_INDICATED
    assert not r['expert_consultation']['recommended']


def test_moderate_adds_disease_specific_management(engine, kb):
    r = engine.recommend('Tomato___Early_blight', confidence=95, severity_category='Moderate')
    assert r['decision']['rules_applied'] == [RULE_MODERATE]
    assert r['management_guidance'] == kb['diseases']['Tomato___Early_blight']['severity_recommendations']['moderate']
    assert r['chemical_guidance']['status'] == CHEM_ADVISORY
    assert r['chemical_guidance']['requires_expert_review'] is True and r['chemical_guidance']['dosage'] is None


def test_severe_adds_removal_and_expert(engine, kb):
    r = engine.recommend('Tomato___Early_blight', confidence=95, severity_category='Severe')
    assert r['decision']['rules_applied'] == [RULE_SEVERE]
    removal = kb['decision_rules']['generic_practices']['severe_removal']
    assert all(item in r['management_guidance'] for item in removal)
    assert r['expert_consultation']['recommended']


def test_unknown_severity_falls_back_to_field_inspection(engine):
    r = engine.recommend('Potato___Early_blight', confidence=95, severity_category='Unknown', severity_status='unreliable')
    assert r['decision']['rules_applied'] == [RULE_SEVERITY_UNKNOWN]
    assert r['expert_consultation']['recommended'] and r['warnings']


@pytest.mark.parametrize('label', VIRAL)
def test_viral_no_curative_fungicide_and_vector_guidance(engine, label):
    r = engine.recommend(label, confidence=99, severity_category='Not applicable')
    assert r['decision']['rules_applied'] == [RULE_SYSTEMIC]
    assert r['chemical_guidance']['status'] == CHEM_NO_CURATIVE
    assert r['chemical_guidance']['active_ingredients_named_by_source'] == []
    assert r['vector_management'] and r['management_guidance']


def test_hlb_and_esca_treated_as_systemic(engine):
    for label in ('Orange___Haunglongbing_(Citrus_greening)', 'Grape___Esca_(Black_Measles)'):
        assert engine.recommend(label, 95, 'Severe')['chemical_guidance']['status'] == CHEM_NO_CURATIVE


def test_unknown_disease_fallback(engine):
    r = engine.recommend('Banana___Sigatoka', confidence=95, severity_category='Moderate')
    assert r['found_in_knowledge_base'] is False
    assert r['decision']['rules_applied'] == [RULE_UNKNOWN_DISEASE]
    assert r['chemical_guidance'] is None and r['sources'] == []
    assert r['expert_consultation']['recommended'] and r['cultural_practices']


def test_spider_mite_is_a_pest(engine):
    r = engine.recommend('Tomato___Spider_mites Two-spotted_spider_mite', 95, 'Severe')
    assert r['disease_info']['pathogen_type'] == 'arthropod_pest'


def test_every_class_produces_a_response(engine):
    for label in CLASS_LABELS:
        for sev in ('Mild', 'Moderate', 'Severe', 'Unknown', 'Not applicable'):
            r = engine.recommend(label, confidence=90, severity_category=sev)
            assert r['disclaimer'] and r['found_in_knowledge_base']
            json.dumps(r)


def test_kb_file_path_exists():
    import os
    assert os.path.exists(KB_PATH)
