from pathlib import Path


TEMPLATE = Path("plex_generate_previews/web/templates/index.html")


def test_index_template_renders_priority_reason_chips():
    html = TEMPLATE.read_text()

    assert "priorityReasonChips(item)" in html
    assert "priorityReasonTitle(chip)" in html
    assert "Next up" in html
    assert "On deck" in html
    assert "Hub" in html
    assert "Multi-user" in html


def test_hub_priority_chip_uses_specific_hub_name():
    html = TEMPLATE.read_text()

    assert "if (chip.type === 'hub' && chip.hub) return chip.hub;" in html
