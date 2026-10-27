from bench import segments
from bench.segments import BOILERPLATE, TocEntry, heading_path, keep_valid, label, locate, normalize, parts, segment_pdf

SENT = "Give the calf colostrum within two hours of birth for good immunity. "  # 69 chars


def _pages(n=30, headings=None):
    headings = headings or {}
    return [{"doc_id": "d" * 64, "file_name": "book.pdf", "page": i, "printed": i, "garbled": False,
             "raw_text": f"{headings.get(i, 'body')}\n{SENT * 3}", "text": f"{headings.get(i, 'body')}\n\n{SENT * 3}"}
            for i in range(1, n + 1)]


def test_titles_must_be_in_the_toc_text_and_pages_never_go_back():
    toc = "Contents\n1 Breeds of Cattle ........ 1\n2 Housing ..... 14\n3 Feeding ..... 9"
    entries = [TocEntry(number="1", title="Breeds of Cattle", printed_page=1, level=1),
               TocEntry(number="x", title="Invented chapter", printed_page=5, level=1),
               TocEntry(number="2", title="Housing", printed_page=14, level=0),
               TocEntry(number="1", title="Breeds of Cattle", printed_page=1, level=1),  # repeats a used line
               TocEntry(number="3", title="Feeding", printed_page=9, level=1)]  # page goes backwards
    kept = keep_valid(entries, toc)
    assert [e.title for e in kept] == ["Breeds of Cattle", "Housing"] and kept[1].level == 1
    assert normalize("Maize……… 6") == "maize 6"


def test_offset_is_found_from_titles_in_the_body():
    pages = _pages(headings={9: "Breeds of Cattle", 16: "Feeding Pregnant Animals", 22: "Housing of Dairy Animals"})
    entries = [TocEntry(number="1", title="Breeds of Cattle", printed_page=1, level=1),
               TocEntry(number="2", title="Feeding Pregnant Animals", printed_page=8, level=1),
               TocEntry(number="3", title="Housing of Dairy Animals", printed_page=14, level=1)]
    agreement, placed = locate(entries, pages, after=3)
    assert agreement == 1.0 and [start for _, start in placed] == [9, 16, 22]


def test_moved_entries_return_to_toc_order_and_unpaged_ones_are_found_by_their_heading():
    toc = ("Contents\nPART – I : PREVALENCE\n1.0 Executive summary 4\nPART – II : CONTROL\n9.0 Use of binders 16\n"
           "IV LIST OF TABLES\nV LIST OF FIGURES\nGLOSSARY\nPART-III : Frequently asked questions")
    as_parsed = [TocEntry(number="1.0", title="Executive summary", printed_page=4, level=2),  # the model moved the Part labels
                 TocEntry(number="PART – I : PREVALENCE", title="PREVALENCE", printed_page=None, level=1),
                 TocEntry(number="9.0", title="Use of binders", printed_page=16, level=2),
                 TocEntry(number="PART – II : CONTROL", title="CONTROL", printed_page=None, level=1),
                 TocEntry(number="IV", title="LIST OF TABLES", printed_page=None, level=1),  # no heading: follows 9.0
                 TocEntry(number="V", title="LIST OF FIGURES", printed_page=None, level=1),  # no heading: dropped
                 TocEntry(number="", title="GLOSSARY", printed_page=None, level=1),
                 TocEntry(number="PART-III", title="Frequently asked questions", printed_page=None, level=1)]
    kept = keep_valid(as_parsed, toc)
    assert [label(e) for e in kept] == ["PART – I : PREVALENCE", "1.0 Executive summary", "PART – II : CONTROL",
                                        "9.0 Use of binders", "IV LIST OF TABLES", "V LIST OF FIGURES", "GLOSSARY",
                                        "PART-III Frequently asked questions"]
    pages = _pages(headings={4: "PART – I\nPREVALENCE", 5: "Executive summary", 16: "PART – II\nCONTROL",
                             17: "Use of binders", 25: "GLOSSARY", 28: "PART-III\nFREQUENTLY ASKED\nQUESTIONS (FAQs)"})
    _, placed = locate(kept, pages, after=3)
    assert [(e.title, start) for e, start in placed] == [
        ("PREVALENCE", 4), ("Executive summary", 5), ("CONTROL", 16), ("Use of binders", 17), ("LIST OF TABLES", 18),
        ("GLOSSARY", 25), ("Frequently asked questions", 28)]


def test_segments_follow_top_level_entries_and_mark_boilerplate(monkeypatch):
    pages = _pages(headings={3: "Contents", 9: "Breeds of Cattle", 11: "Indigenous breeds", 16: "Annexure-I Semen Banks",
                             20: "Milk tables"})
    pages[2]["raw_text"] = "Contents\n" + "\n".join(f"Entry {i} ........ {i}" for i in range(10))
    pages[28]["printed"] = pages[29]["printed"] = None  # unnumbered covers at the end
    toc = [TocEntry(number="1", title="Breeds of Cattle", printed_page=1, level=1),
           TocEntry(number="1.1", title="Indigenous breeds", printed_page=3, level=2),
           TocEntry(number="Annexure-I", title="Semen Banks", printed_page=8, level=1),
           TocEntry(number="SECTION-I", title="Milk tables", printed_page=12, level=1)]  # after an annexure: back matter
    monkeypatch.setattr(segments, "read_toc", lambda name, text: toc)
    found = segment_pdf(pages)
    assert [(s["title"], s["pages"], s["use_for_questions"]) for s in found] == [
        ("Front matter", [1, 8], False), ("1 Breeds of Cattle", [9, 15], True), ("Annexure-I Semen Banks", [16, 19], False),
        ("SECTION-I Milk tables", [20, 28], False), ("Covers", [29, 30], False)]
    assert heading_path(found[1], 12) == "1 Breeds of Cattle > 1.1 Indigenous breeds"
    for s in found:  # page_spans point at each page's text inside the segment
        assert all(s["text"][ps["start"] : ps["end"]] == pages[ps["page"] - 1]["text"] for ps in s["page_spans"])


def test_big_segments_split_at_paragraphs_with_context_and_headings(monkeypatch):
    pages = _pages(headings={9: "Breeds of Cattle", 11: "Indigenous breeds"})
    toc = [TocEntry(number="1", title="Breeds of Cattle", printed_page=1, level=1),
           TocEntry(number="1.1", title="Indigenous breeds", printed_page=3, level=2),
           TocEntry(number="2", title="Housing", printed_page=40, level=1)]
    monkeypatch.setattr(segments, "read_toc", lambda name, text: toc)
    pages[2]["raw_text"] = "Contents\n" + "\n".join(f"Entry {i} ........ {i}" for i in range(10))
    seg = next(s for s in segment_pdf(pages) if s["title"] == "1 Breeds of Cattle")
    cut = parts(seg, max_chars=700)
    assert len(cut) > 3 and cut[0]["context"] == "" and cut[1]["context"]
    assert all(len(p["text"]) <= 700 for p in cut)
    assert cut[-1]["heading"] == "1 Breeds of Cattle > 1.1 Indigenous breeds"
    assert parts(seg, max_chars=10**9)[0]["text"] == seg["text"].strip()  # a small segment is one part


def test_an_unpaged_entry_is_never_placed_past_the_next_paged_one():
    entries = [TocEntry(number="1", title="Breeds of Cattle", printed_page=1, level=1),
               TocEntry(number="", title="Summary", printed_page=None, level=1),  # its title line only appears much later
               TocEntry(number="2", title="Housing of Dairy Animals", printed_page=8, level=1)]
    pages = _pages(headings={9: "Breeds of Cattle", 16: "Housing of Dairy Animals", 25: "Summary"})
    _, placed = locate(entries, pages, after=3)
    assert [(e.title, start) for e, start in placed] == [("Breeds of Cattle", 9), ("Housing of Dairy Animals", 16)]


def test_boilerplate_titles_are_matched_at_their_start():
    assert all(BOILERPLATE.match(t) for t in ("VI GLOSSARY", "8. References", "Annexure-I Semen Banks", "IV LIST OF TABLES"))
    assert not any(BOILERPLATE.match(t) for t in ("5 Temperature Humidity Index", "Mineral contents of feeds"))


def test_a_title_wrapped_around_its_page_number_is_kept():
    toc = "3. Managing Dairy Animals for Optimum Productivity 46\nand Health\n4. Housing 58"
    entry = TocEntry(number="3", title="Managing Dairy Animals for Optimum Productivity and Health", printed_page=46, level=1)
    assert keep_valid([entry], toc) == [entry]
