import asyncio

from starlette.requests import Request

from plex_generate_previews.web import main


class StrictTemplates:
    def __init__(self):
        self.calls = []

    def TemplateResponse(self, request, name, context=None, *args, **kwargs):
        assert isinstance(request, Request)
        assert isinstance(name, str)
        self.calls.append((name, context or {}))
        return {"template": name, "context": context or {}}


def make_request(path="/"):
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
            "client": ("testclient", 50000),
        }
    )


def test_template_routes_use_request_first_signature(monkeypatch):
    strict_templates = StrictTemplates()
    monkeypatch.setattr(main, "templates", strict_templates)
    monkeypatch.setattr(main.scheduler, "config", object())

    assert asyncio.run(main.login_page(make_request("/login")))["template"] == "login.html"
    assert asyncio.run(main.settings_page(make_request("/settings"), user="test"))["template"] == "settings.html"
    assert asyncio.run(main.setup_page(make_request("/setup"), user="test"))["template"] == "setup.html"
    assert asyncio.run(main.index(make_request("/"), user="test"))["template"] == "index.html"

    assert strict_templates.calls == [
        ("login.html", {}),
        ("settings.html", {"user": "test"}),
        ("setup.html", {}),
        ("index.html", {"user": "test"}),
    ]
