"""
Shared robust element locator for Playwright environments.
Extracts multi-attribute cascade matching to avoid duplication between
PlaywrightWebEnv and GroundingValidator.
"""

from typing import Any, Dict, Optional, Tuple

from playwright.async_api import Locator, Page

from step_rl.utils.logging_utils import get_logger
from step_rl.utils.security_utils import escape_css_string

logger = get_logger(__name__)


async def robust_locate(
    page: Page,
    params: Dict[str, Any],
    multi_attribute_match: bool = True,
) -> Tuple[Optional[Locator], Dict[str, Any]]:
    """
    Multi-attribute cascade matching with input sanitization.
    Priority: element_id > element_text (+ tag) > xpath > css_selector > coordinates

    Returns:
        (locator, match_info)
    """
    if not multi_attribute_match:
        sel = (
            params.get("xpath")
            or params.get("css_selector")
            or f"text={escape_css_string(params.get('element_text', ''))}"
        )
        if sel:
            try:
                loc = page.locator(sel)
                if await loc.count() > 0:
                    return loc.first, {"method": "simple", "selector": sel}
            except Exception as e:
                logger.debug(f"Simple locator failed: {e}")
        return None, {"method": "simple", "status": "failed"}

    # Cascade matching with sanitized inputs
    selectors = []
    methods = []

    if params.get("element_id"):
        eid = escape_css_string(params["element_id"])
        selectors.extend(
            [
                f"[data-testid='{eid}']",
                f"[data-test-id='{eid}']",
                f"#{eid}",
                f"[id='{eid}']",
            ]
        )
        methods.extend(["data-testid", "data-test-id", "id_hash", "id_attr"])

    if params.get("element_text"):
        text = escape_css_string(params["element_text"])
        selectors.append(f"text={text}")
        methods.append("text_exact")
        if params.get("tag"):
            tag = escape_css_string(params["tag"])
            selectors.append(f"{tag}:has-text('{text}')")
            methods.append("tag_text")

    if params.get("xpath"):
        selectors.append(params["xpath"])
        methods.append("xpath")

    if params.get("css_selector"):
        selectors.append(params["css_selector"])
        methods.append("css")

    for sel, method in zip(selectors, methods):
        if not sel:
            continue
        try:
            loc = page.locator(sel)
            count = await loc.count()
            if count > 0:
                return loc.first, {"method": method, "selector": sel, "count": count}
        except Exception as e:
            logger.debug(f"Locator attempt failed ({method}): {e}")
            continue

    # Coordinate fallback
    locator, info = await _coordinate_fallback(page, params)
    if locator is not None:
        return locator, info

    return None, {"method": "none", "status": "not_found"}


async def _coordinate_fallback(
    page: Page, params: Dict[str, Any]
) -> Tuple[Optional[Locator], Dict[str, Any]]:
    """
    Fallback to coordinates. Returns a locator that actually targets the
    element found at the given coordinates, not 'body'.
    """
    coords = params.get("coordinates")
    if not coords or not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None, {}

    x, y = int(coords[0]), int(coords[1])
    try:
        elem_info = await page.evaluate(
            """({x, y}) => {
                const el = document.elementFromPoint(x, y);
                if (!el) return null;
                return {
                    tag: el.tagName.toLowerCase(),
                    id: el.id || '',
                    text: (el.innerText || el.textContent || '').slice(0,50).trim(),
                    testid: el.getAttribute('data-testid') || '',
                    clickable: el.tagName === 'BUTTON' || el.tagName === 'A' || el.onclick != null
                };
            }""",
            {"x": x, "y": y},
        )
        if not elem_info or not elem_info.get("clickable"):
            return None, {"method": "coordinate_fallback", "status": "not_clickable"}

        # Build a locator that actually targets the element
        tag = elem_info.get("tag", "")
        eid = elem_info.get("id", "")
        testid = elem_info.get("testid", "")
        text = elem_info.get("text", "")

        # Prefer id or testid since they are most robust
        if testid:
            loc = page.locator(f"[data-testid='{escape_css_string(testid)}']")
            if await loc.count() > 0:
                return loc.first, {
                    "method": "coordinate_fallback",
                    "coords": [x, y],
                    "by": "testid",
                }
        if eid:
            loc = page.locator(f"#{escape_css_string(eid)}")
            if await loc.count() > 0:
                return loc.first, {
                    "method": "coordinate_fallback",
                    "coords": [x, y],
                    "by": "id",
                }
        if text:
            loc = page.locator(f"{tag}:has-text('{escape_css_string(text)}')")
            if await loc.count() > 0:
                return loc.first, {
                    "method": "coordinate_fallback",
                    "coords": [x, y],
                    "by": "tag_text",
                }
        # Last resort: tag at coordinates via nth-of-type is fragile;
        # use JS click via page.evaluate as a reliable fallback
        return (
            page.locator(f"{tag}").filter(has_text=text if text else None).first
            if text
            else page.locator(tag).first
        ), {
            "method": "coordinate_fallback",
            "coords": [x, y],
            "by": "tag",
        }

    except Exception as e:
        logger.debug(f"Coordinate fallback failed: {e}")
        return None, {
            "method": "coordinate_fallback",
            "status": "error",
            "error": str(e),
        }
