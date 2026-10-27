import pymupdf

from bench.pdf import extract, garbled


def _pdf(tmp_path, pages=5):
    doc = pymupdf.open()
    for n in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 40), "Dairy Farming")  # running header
        page.insert_textbox(pymupdf.Rect(72, 100, 520, 160), f"Paragraph one of page {n}.\nFeed {n + 1} kg of hay.")
        page.insert_text((72, 220), "yy")  # stray bullet glyph
        page.insert_textbox(pymupdf.Rect(72, 260, 520, 320), f"Give {n * 10} g of mineral mixture.\nSplit over {n} meals.")
        page.insert_textbox(pymupdf.Rect(72, 400, 520, 440), f"60-{70 + n}")  # a table value at the page edge
        page.insert_text((300, 800), str(n + 8))  # printed page number = PDF page + 8
    path = tmp_path / "book.pdf"
    doc.save(path)
    return path


def test_headers_page_numbers_and_glyphs_go_but_paragraphs_and_table_values_stay(tmp_path):
    page = extract(_pdf(tmp_path))[2]
    assert page["text"] == ("Paragraph one of page 3.\nFeed 4 kg of hay.\n\n"
                            "Give 30 g of mineral mixture.\nSplit over 3 meals.\n\n60-73")
    assert "Dairy Farming" in page["raw_text"] and not page["garbled"] and page["printed"] == 11


def test_garbled_flags_ocr_junk_but_not_tables_or_prose():
    assert garbled("0 CL o c — Z3 0 o -O C 3 l— 0 o c 0 0 p - d .3 0 .E " * 5)
    assert not garbled("34 Delhi - 0.00 NIL 0.00 35 Lakshadweep 3188 2865 - 13 " * 8)
    assert not garbled("Give the calf colostrum within two hours of birth for good immunity. " * 6)
