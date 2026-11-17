import json

import pytest
from typer.testing import CliRunner

from bench import cli, segments, templates
from bench.config import BenchError
from bench.templates import (
    active,
    assigned,
    coverage,
    fact_intents,
    styles,
    subject_names,
    template_fact_problem,
    template_question_problem,
)
from bench.templates import templates as load_templates
from bench.verify import SHAPES, not_english, uid

REPO = load_templates()  # the committed templates/ files
NO_TOPICS = {"switches": {}, "sections": {}}

# Sentences written for these tests (not copied from the PDFs): one that speaks to each intent and one that doesn't.
FIXTURES = {
    ("animal_care", "symptoms"): ("Affected cows show high fever, swelling of the joints and a nasal discharge.",
                                  "Cows should be milked twice a day at fixed times."),
    ("animal_care", "cause_and_spread"): ("The disease is caused by a virus and spreads through biting flies.",
                                          "Store the feed bags on a raised platform."),
    ("animal_care", "prevention"): ("To prevent the infection, isolate new animals for three weeks.",
                                    "Milk yield peaks in the second month of lactation."),
    ("animal_care", "vaccination_schedule"): ("Calves should be vaccinated at 4 months of age, with a booster every year.",
                                              "Vaccination protects the whole herd."),  # no age or interval: not a figure
    ("animal_care", "treatment_first_aid"): ("Treat bloat by drenching the animal with vegetable oil and call a veterinarian.",
                                             "Green fodder is cheaper in the monsoon."),
    ("govt_scheme", "objective"): ("The objective of the mission is to improve the productivity of indigenous cattle.",
                                   "Buffalo milk has more fat than cow milk."),
    ("govt_scheme", "eligibility"): ("Individual farmers, self help groups and cooperatives are eligible to apply.",
                                     "The shed should face east."),
    ("govt_scheme", "benefit"): ("The scheme gives a capital subsidy of 50%, up to Rs 25 lakh per unit.",
                                 "A subsidy may be given to farmers."),  # no amount
    ("govt_scheme", "funding_pattern"): ("The funding pattern between the Centre and the States is 60:40.",
                                         "Funding comes from the Centre."),
    ("govt_scheme", "implementing_agency"): ("The scheme is implemented through NABARD in every state.",
                                             "The scheme was implemented in many states."),
    ("statistical_report", "value_lookup"): ("Total milk production reached 247.87 million tonnes in 2024-25.",
                                             "Production depends on feeding."),
    ("statistical_report", "growth_change"): ("Egg production grew by 6.8% over the previous year.",
                                              "Egg production grew steadily."),
    ("statistical_report", "share_contribution"): ("Buffaloes contribute 31.2% of the total milk production.",
                                                   "Buffaloes contribute a lot of milk."),
    ("statistical_report", "per_unit_productivity"): ("The per capita availability of milk is 485 grams per day.",
                                                      "Per capita availability has improved."),
    ("statistical_report", "ranking_top_n"): ("Uttar Pradesh is the largest milk producing state, followed by Rajasthan.",
                                              "Many states produce milk."),
}


def _intents(template: str, sentence: str) -> list[str]:
    text = f"Some earlier text.\n\n{sentence}"
    fact = {"fact_id": "s:f01", "quote": sentence, "start": text.index(sentence), "end": len(text)}
    return fact_intents(fact, text, REPO[template])


@pytest.mark.parametrize("key", list(FIXTURES))
def test_cue_and_shape_tagging_on_written_fixtures(key):
    template, intent = key
    hit, miss = FIXTURES[key]
    assert intent in _intents(template, hit)
    assert intent not in _intents(template, miss)


def test_a_heading_just_above_a_quote_counts_but_earlier_sentences_do_not():
    text = "The calf had a mild fever last week.\n\nSYMPTOMS\nAnimals stop eating and stand apart."
    fact = {"fact_id": "s:f01", "quote": "Animals stop eating and stand apart.", "start": text.index("Animals"), "end": len(text)}
    assert "symptoms" in fact_intents(fact, text, REPO["animal_care"])
    text = "The calf had a mild fever last week.\n\nAnimals stop eating and stand apart."
    fact = {**fact, "start": text.index("Animals"), "end": len(text)}
    assert "symptoms" not in fact_intents(fact, text, REPO["animal_care"])


def test_shapes():
    assert all(SHAPES["money"].search(t) for t in ["Rs. 25 lakh", "₹1,500", "2 crore", "INR 5,00,000"])
    assert not SHAPES["money"].search("in 2024-25")
    assert SHAPES["ratio"].search("60:40") and not SHAPES["ratio"].search("between 10:30 am and noon")
    assert SHAPES["percent"].search("4.97 per cent") and SHAPES["age"].search("between 4-8 months of age")
    assert SHAPES["rate"].search("9.05 Kg/day") and SHAPES["rate"].search("485 grams per day")


def test_every_committed_example_is_plain_english_and_names_its_subject():
    for name, tmpl in REPO.items():
        for intent in tmpl.intents.values():
            for example in intent.examples:
                question = example.replace("{subject}", "lumpy skin disease")
                assert not_english(question) is None, (name, example)
                assert "{subject}" in example or not tmpl.subject


def test_templates_are_checked_strictly(tmp_path):
    (tmp_path / "ok.toml").write_text('description = "d"\nsignals = ["a"]\n[intents.i]\nkind = "explanation"\n'
                                      'description = "d"\ncues = ["c"]\nexamples = ["e"]\n')
    assert list(load_templates(tmp_path)) == ["ok"]
    for bad, message in [('signal = ["a"]\n', "signal"),  # a typo for signals
                         ('[intents.i]\nkind = "figure"\ndescription = "d"\ncues = ["c"]\nexamples = ["e"]\n', "needs a shape"),
                         ('[intents.i]\nkind = "figure"\ndescription = "d"\ncues = ["c"]\nshape = ["cash"]\nexamples = ["e"]\n', "unknown shape")]:
        (tmp_path / "ok.toml").write_text('description = "d"\nsignals = ["a"]\n' + bad
                                          + ('' if "intents" in bad else '[intents.i]\nkind = "explanation"\ndescription = "d"\ncues = ["c"]\nexamples = ["e"]\n'))
        with pytest.raises(BenchError, match=message):
            load_templates(tmp_path)


def _segment(sid: str, title: str, text: str) -> dict:
    return {"segment_id": sid, "file_name": "book.pdf", "title": title, "headings": [], "use_for_questions": True, "text": text}


DISEASE = _segment("book-01", "Health", ("Foot and mouth disease is a viral disease. Symptoms include fever and blisters. "
                                         "Vaccinate calves at 4 months of age and repeat every six months. Treat sick animals "
                                         "with a veterinarian's help, and isolate them to prevent infection. ") * 4)
STATS = _segment("stats-01", "Highlights", "Milk production 2023-24 239.30 2024-25 247.87 growth 3.58% share 31.2% "
                                           "Uttar Pradesh 15.66% Rajasthan 14.82% Madhya Pradesh 8.91% " * 6)
SCHEME = _segment("dairy-01", "Schemes", ("The centrally sponsored scheme gives a subsidy to eligible beneficiaries through "
                                          "the implementing agency, with grant-in-aid from the Central Government. ") * 4)
GENERAL = _segment("dairy-05", "Housing", "A shed should face east and have a sloping floor so that it stays dry. " * 10)


def test_the_orchestrator_categorises_the_rules_are_its_fallback_and_you_can_override():
    assert assigned(DISEASE, REPO, NO_TOPICS) == (["animal_care"], "rules")
    assert assigned(STATS, REPO, NO_TOPICS) == (["statistical_report"], "rules")
    assert assigned(SCHEME, REPO, NO_TOPICS) == (["govt_scheme"], "rules")
    assert assigned(GENERAL, REPO, NO_TOPICS) == ([], "rules")
    saved = {"dairy-05": {"templates": ["govt_scheme"], "by": "orchestrator"}}  # the decision saved on the PDF's row
    assert assigned(GENERAL, REPO, NO_TOPICS, saved) == (["govt_scheme"], "orchestrator")
    assert assigned(GENERAL, REPO, {"switches": {}, "sections": {"dairy-05": ["animal_care"]}}, saved) == (["animal_care"], "you")
    assert active(DISEASE, REPO, {"switches": {"animal_care.symptoms": False}, "sections": {}}) == {
        "animal_care": ["cause_and_spread", "prevention", "vaccination_schedule", "treatment_first_aid"]}


def test_coverage_statuses():
    text = DISEASE["text"]
    quote = "Vaccinate calves at 4 months of age and repeat every six months."
    fact = {"fact_id": "book-01:f01", "uid": uid("x"), "quote": quote, "start": text.index(quote), "end": text.index(quote) + len(quote)}
    rows = coverage([DISEASE], {"book-01": {"facts": [fact]}}, {"book-01": {"questions": [{"fact_ids": ["book-01:f01"]}]}},
                    {"animal_care": REPO["animal_care"]}, NO_TOPICS)
    status = {r["intent"]: r["status"] for r in rows}
    assert status["vaccination_schedule"] == "covered" and status["symptoms"] == "stated but missing"
    rows = coverage([DISEASE], {"book-01": {"facts": [fact]}}, {}, {"animal_care": REPO["animal_care"]}, NO_TOPICS)
    assert {r["intent"]: r["status"] for r in rows}["vaccination_schedule"] == "facts only"
    switched = {"switches": {"animal_care.symptoms": False}, "sections": {}}
    assert "symptoms" not in {r["intent"] for r in coverage([DISEASE], {}, {}, {"animal_care": REPO["animal_care"]}, switched)}
    general = coverage([GENERAL], {}, {}, {"animal_care": REPO["animal_care"]}, {"switches": {}, "sections": {"dairy-05": ["animal_care"]}})
    assert {r["status"] for r in general} == {"not stated"}


def test_cli_assigns_switches_and_checks(isolated, monkeypatch):
    monkeypatch.setattr(segments, "load_segments", lambda: [DISEASE, GENERAL])
    runner = CliRunner()
    assert runner.invoke(cli.app, ["templates", "assign", "dairy-05", "animal_care"]).exit_code == 0
    assert runner.invoke(cli.app, ["templates", "off", "animal_care", "symptoms"]).exit_code == 0
    topics = json.loads(templates.TOPICS.read_text())
    assert topics == {"sections": {"dairy-05": ["animal_care"]}, "switches": {"animal_care.symptoms": False}, "groups": {}}
    result = runner.invoke(cli.app, ["templates", "assign", "dairy-05", "no_such_template"])
    assert result.exit_code == 1 and "unknown template" in result.output
    assert runner.invoke(cli.app, ["templates", "assign", "dairy-05", "auto"]).exit_code == 0
    assert json.loads(templates.TOPICS.read_text())["sections"] == {}
    listing = runner.invoke(cli.app, ["templates", "list"])
    assert "symptoms (off)" in listing.output and "book-01: animal_care (rules)" in listing.output


def _fact(text: str, quote: str, **extra) -> dict:
    return {"fact_id": "book-01:tf01", "quote": quote, "start": text.index(quote), "end": text.index(quote) + len(quote), **extra}


def test_the_intent_audit_needs_a_cue_a_shape_and_the_subject():
    text = "Foot and Mouth Disease\n\nVaccinate calves at 4 months of age, with a booster every six months."
    seg = {**DISEASE, "text": text, "headings": [], "page_spans": [{"page": 1, "start": 0, "end": len(text)}], "pages": [1, 1]}
    quote = "Vaccinate calves at 4 months of age, with a booster every six months."
    fact = _fact(text, quote, template="animal_care", intent="vaccination_schedule", subject="Foot and Mouth Disease (FMD)")
    assert template_fact_problem(fact, seg, REPO) is None  # the subject is named in the heading line above
    assert "isn't named" in template_fact_problem({**fact, "subject": "Lumpy Skin Disease"}, seg, REPO)
    assert template_fact_problem({**fact, "intent": "symptoms"}, seg, REPO).startswith("no cue word")
    shapeless = "Foot and Mouth Disease\n\nVaccination protects the whole herd from the disease."
    fact = _fact(shapeless, "Vaccination protects the whole herd from the disease.", template="animal_care",
                 intent="vaccination_schedule", subject="Foot and Mouth Disease")
    assert template_fact_problem(fact, {**seg, "text": shapeless}, REPO) == "no age or interval in the quote"
    assert subject_names("Foot and Mouth Disease (FMD)") == ["Foot and Mouth Disease (FMD)", "Foot and Mouth Disease", "FMD"]


def test_a_table_row_takes_its_unit_from_the_header():
    text = "Milk production by year\n\nYear | Milk production (million tonnes)\n2024-25 | 247.87"
    fact = _fact(text, "2024-25 | 247.87", header="Year | Milk production (million tonnes)",
                 header_start=text.index("Year |"), template="statistical_report", intent="value_lookup", subject="milk production")
    seg = {**STATS, "text": text, "headings": [], "page_spans": [{"page": 1, "start": 0, "end": len(text)}], "pages": [1, 1]}
    assert template_fact_problem(fact, seg, REPO) is None


def test_a_template_question_names_its_subject_and_is_not_a_general_question():
    fact = {"fact_id": "s:tf01", "quote": "A dry floor prevents mastitis.", "subject": "mastitis"}
    kinds = {"direct": {"rules": {"facts": [1, 1]}}}
    q = {"type": "direct", "question": "How can a farmer prevent mastitis?", "answer": "A dry floor prevents mastitis", "fact_ids": ["s:tf01"]}
    assert template_question_problem(q, {"s:tf01": fact}, kinds, "", set()) is None
    assert template_question_problem({**q, "question": "How can udder infection be prevented?"}, {"s:tf01": fact}, kinds, "", set()).startswith("doesn't name")
    assert template_question_problem(q, {"s:tf01": fact}, kinds, "", {"how can a farmer prevent mastitis"}) == "same as a general question"
    with pytest.raises(BenchError, match="aren't question types"):
        styles({"x": REPO["animal_care"].model_copy(update={"styles": {"unanswerable": 1}})})  # a type, but not one written from facts


def test_a_subject_is_named_in_any_case_or_number():
    from bench.templates import names
    assert names("Aflatoxins", "to prevent aflatoxin contamination") and names("milk production", "Milk Production (MT)")
    assert names("Foot and Mouth Disease (FMD)", "FMD vaccine") and names("Lumpy skin disease", "lumpy-skin disease")
    assert not names("Mastitis", "the udder was swollen") and not names("cow", "cowpea fodder")


def test_a_list_under_a_disease_and_symptoms_heading_counts_but_a_distant_heading_does_not():
    bullets = "\n".join(f"• Sign {n}: the animal stands apart from the herd." for n in range(8))
    text = f"A. RUMEN ACIDOSIS\n\nSymptoms:\n{bullets}\n• Sudden drop in feed intake and milk yield."
    quote = "• Sudden drop in feed intake and milk yield."
    seg = {**DISEASE, "title": "Metabolic disorders", "text": text, "headings": [], "pages": [1, 1],
           "page_spans": [{"page": 1, "start": 0, "end": len(text)}]}
    fact = _fact(text, quote, template="animal_care", intent="symptoms", subject="Rumen acidosis")
    assert template_fact_problem(fact, seg, REPO) is None
    far = "A. RUMEN ACIDOSIS\n\n" + "Cows were fed hay and grain on most farms in the district. " * 30 + f"\n\n{quote}"
    fact = _fact(far, quote, template="animal_care", intent="symptoms", subject="Rumen acidosis")
    assert template_fact_problem(fact, {**seg, "text": far, "page_spans": [{"page": 1, "start": 0, "end": len(far)}]}, REPO).startswith("no cue")


def test_a_question_may_name_its_subject_by_one_of_its_words():
    from bench.templates import mentions
    assert mentions("wool production", "By what percentage did India's wool output grow in 2024-25?")
    assert mentions("JOHNE’S DISEASE", "What are the signs of Johne's disease in cattle?")
    assert not mentions("Aflatoxins", "What are the two main stages of mycotoxin control?")
    assert not mentions("DYSTOCIA", "When do I need to call the vet during a difficult birth?")
