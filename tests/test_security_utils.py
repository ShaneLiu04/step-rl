"""Tests for security utilities."""

from step_rl.utils.security_utils import (
    escape_css_string,
    escape_xpath_string,
    validate_url,
)


class TestEscapeCssString:
    def test_basic(self):
        assert escape_css_string("hello") == "hello"

    def test_single_quote(self):
        assert escape_css_string("it's") == "it\\'s"

    def test_double_quote(self):
        assert escape_css_string('say "hi"') == 'say \\"hi\\"'

    def test_backslash(self):
        assert escape_css_string("a\\b") == "a\\\\b"

    def test_newline(self):
        assert escape_css_string("line1\nline2") == "line1\\n line2"

    def test_null_byte(self):
        assert escape_css_string("a\x00b") == "ab"

    def test_non_string(self):
        assert escape_css_string(123) == ""


class TestEscapeXpathString:
    def test_no_quotes(self):
        assert escape_xpath_string("hello") == "'hello'"

    def test_single_quote_only(self):
        result = escape_xpath_string("it's")
        # When only single quotes present, wrap in double quotes
        assert result == '"it\'s"'

    def test_both_quotes(self):
        result = escape_xpath_string('it\'s a "test"')
        assert "concat(" in result

    def test_non_string(self):
        assert escape_xpath_string(123) == ""


class TestValidateUrl:
    def test_allowed_default(self):
        assert validate_url("https://example.com", set(), set()) is True

    def test_blocked_exact(self):
        assert validate_url("https://localhost/foo", {"localhost"}, set()) is False

    def test_blocked_subdomain(self):
        assert validate_url("https://api.localhost/foo", {"localhost"}, set()) is False

    def test_blocked_substring_bypass(self):
        # Ensure exact/subdomain matching, not substring
        assert validate_url("https://not-example.com", {"example.com"}, set()) is True

    def test_allowed_list(self):
        assert validate_url("https://example.com", set(), {"example.com"}) is True
        assert validate_url("https://other.com", set(), {"example.com"}) is False

    def test_invalid_url(self):
        assert validate_url("", set(), set()) is False
        assert validate_url(None, set(), set()) is False
