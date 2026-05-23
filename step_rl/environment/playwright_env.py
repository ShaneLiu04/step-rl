"""
Playwright Web Environment for Step-RL v2.0
- Accessibility tree compression via JS DOM extraction
- Multi-attribute robust anchoring (delegated to shared locator module)
- Action execution with validation hooks
- Security sandbox enforcement with proper domain validation
"""

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from step_rl.environment.locator import robust_locate
from step_rl.utils.logging_utils import get_logger
from step_rl.utils.security_utils import validate_url

logger = get_logger(__name__)


@dataclass
class Observation:
    """Structured observation from the web environment."""

    text: str = ""
    url: str = ""
    title: str = ""
    viewport: Dict[str, int] = field(default_factory=dict)
    screenshot_b64: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Structured action output from the policy."""

    thought: str = ""
    action: str = "wait"  # click, type, scroll, goto, wait, finish
    params: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"thought": self.thought, "action": self.action, "params": self.params},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> "Action":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = cls._extract_from_text(text)
        return cls(
            thought=data.get("thought", ""),
            action=data.get("action", "wait"),
            params=data.get("params", {}),
        )

    @staticmethod
    def _extract_from_text(text: str) -> Dict[str, Any]:
        """Fallback extraction when JSON is malformed."""
        action_match = re.search(r'"action"\s*:\s*"(\w+)"', text)
        thought_match = re.search(r'"thought"\s*:\s*"([^"]+)"', text)
        return {
            "thought": thought_match.group(1) if thought_match else "",
            "action": action_match.group(1) if action_match else "wait",
            "params": {},
        }


@dataclass
class StepResult:
    """Result of a single step."""

    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class PlaywrightWebEnv:
    """
    Async Web Environment using Playwright.
    Handles browser lifecycle, observation extraction, and action execution.
    """

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: Optional[Dict[str, int]] = None,
        max_obs_tokens: int = 2048,
        timeout_ms: int = 30000,
        action_timeout_ms: int = 5000,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        sandbox_mode: bool = True,
    ):
        self.browser_type = browser_type
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.max_obs_tokens = max_obs_tokens
        self.timeout_ms = timeout_ms
        self.action_timeout_ms = action_timeout_ms
        self.allowed_domains = set(
            d.lower().strip() for d in (allowed_domains or []) if d
        )
        self.blocked_domains = set(
            d.lower().strip() for d in (blocked_domains or []) if d
        )
        self.sandbox_mode = sandbox_mode

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._step_count = 0
        self._task_goal = ""

    # -----------------------------
    # Lifecycle
    # -----------------------------

    async def start(self) -> None:
        """Start the browser with graceful error handling."""
        try:
            self._playwright = await async_playwright().start()
            browser_cls = getattr(self._playwright, self.browser_type)
            self._browser = await browser_cls.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                viewport=self.viewport,
                java_script_enabled=True,
            )
            # Block unwanted resources for speed
            await self._context.route(
                "**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2,ttf}",
                lambda route: route.abort(),
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.timeout_ms)
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            await self.stop()
            raise RuntimeError(f"Browser startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop the browser, handling partial state gracefully."""
        try:
            if self._context:
                await self._context.close()
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
        try:
            if self._browser:
                await self._browser.close()
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
        self._page = None
        self._browser = None
        self._context = None
        self._playwright = None

    async def reset(
        self, task_goal: str = "", start_url: Optional[str] = None
    ) -> Observation:
        self._step_count = 0
        self._task_goal = task_goal
        if self._page is None:
            await self.start()
        else:
            try:
                await self._page.goto("about:blank")
            except Exception as e:
                logger.warning(f"Failed to navigate to blank page: {e}")
                await self.stop()
                await self.start()
        if start_url:
            await self._safe_goto(start_url)
        await self._wait_for_page_ready()
        return await self.get_observation()

    # -----------------------------
    # Observation
    # -----------------------------

    async def get_observation(self, include_screenshot: bool = False) -> Observation:
        page = self._page
        if page is None:
            raise RuntimeError(
                "Environment not started. Call start() or reset() first."
            )

        compressed = await self._extract_page_text(page)

        screenshot_b64 = None
        if include_screenshot:
            try:
                screenshot_bytes = await page.screenshot(type="jpeg", quality=50)
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            except Exception as e:
                logger.warning(f"Screenshot failed: {e}")

        try:
            title = await page.title()
        except Exception as e:
            logger.warning(f"Failed to get page title: {e}")
            title = ""

        return Observation(
            text=compressed,
            url=page.url,
            title=title,
            viewport=self.viewport,
            screenshot_b64=screenshot_b64,
            metadata={"step": self._step_count, "task": self._task_goal},
        )

    async def _extract_page_text(self, page, max_tokens: Optional[int] = None) -> str:
        """
        Extract a compressed text representation of the page using JavaScript.
        Compatible with Playwright >= 1.60 where page.accessibility is removed.
        """
        max_tokens = max_tokens or self.max_obs_tokens
        max_chars = int(max_tokens * 2.5)

        js_code = """
        () => {
            const results = [];
            const tags = ['a', 'button', 'input', 'textarea', 'select', 'label',
                          'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'div',
                          'li', 'td', 'th'];
            tags.forEach(tag => {
                document.querySelectorAll(tag).forEach((el, idx) => {
                    if (el.offsetParent === null && tag !== 'div') return;
                    const rect = el.getBoundingClientRect();
                    const text = (el.innerText || el.textContent || el.value || el.placeholder || '').trim();
                    const role = el.getAttribute('role') || tag;
                    const id = el.id || el.getAttribute('data-testid') || '';
                    if (text.length > 0 || id.length > 0 || tag === 'input' || tag === 'button' || tag === 'a') {
                        results.push({
                            tag: tag,
                            role: role,
                            text: text.slice(0, 200),
                            id: id,
                            coords: `(${Math.round(rect.x)},${Math.round(rect.y)})`,
                            visible: el.offsetParent !== null
                        });
                    }
                });
            });
            return results;
        }
        """
        try:
            elements = await page.evaluate(js_code)
        except Exception as e:
            logger.warning(f"JS extraction failed: {e}, falling back to BeautifulSoup")
            try:
                html = await page.content()
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "lxml")
                text = soup.get_text(separator="\n", strip=True)
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n...[truncated]"
                return text
            except Exception as e2:
                logger.error(f"Fallback extraction also failed: {e2}")
                return ""

        lines: List[str] = []
        seen = set()
        for el in elements:
            key = (el.get("tag"), el.get("text"), el.get("id"))
            if key in seen:
                continue
            seen.add(key)
            parts = [el.get("tag", ""), el.get("role", "")]
            if el.get("text"):
                parts.append(f"'{el['text']}'")
            if el.get("id"):
                parts.append(f"id={el['id']}")
            parts.append(el.get("coords", ""))
            line = " ".join([p for p in parts if p])
            lines.append(line)

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text

    # -----------------------------
    # Action Execution
    # -----------------------------

    async def execute_action(self, action: Action) -> tuple[bool, Dict[str, Any]]:
        """
        Execute an action. Returns (success, info_dict).
        This is the low-level executor; GroundingValidator handles pre-validation.
        """
        page = self._page
        if page is None:
            return False, {"error": "Page not initialized"}

        self._step_count += 1
        params = action.params
        info = {"step": self._step_count, "action": action.action, "params": params}

        try:
            if action.action == "goto":
                url = params.get("url", "about:blank")
                success = await self._safe_goto(url)
                info["success"] = success

            elif action.action == "click":
                locator = await robust_locate(page, params)
                if locator[0] is None:
                    return False, {**info, "error": "Element not found for click"}
                await locator[0].click(timeout=self.action_timeout_ms)
                info["success"] = True

            elif action.action == "type":
                locator = await robust_locate(page, params)
                if locator[0] is None:
                    return False, {**info, "error": "Element not found for type"}
                text = params.get("text", "")
                await locator[0].fill(text, timeout=self.action_timeout_ms)
                info["success"] = True

            elif action.action == "scroll":
                direction = params.get("direction", "down")
                amount = params.get("amount", 500)
                if direction == "down":
                    await page.evaluate(f"window.scrollBy(0, {amount})")
                else:
                    await page.evaluate(f"window.scrollBy(0, -{amount})")
                info["success"] = True

            elif action.action == "wait":
                duration = params.get("duration_ms", 1000)
                await page.wait_for_timeout(duration)
                info["success"] = True

            elif action.action == "finish":
                info["success"] = True
                info["terminal"] = True

            else:
                return False, {**info, "error": f"Unknown action: {action.action}"}

            # Post-action wait for SPA stability
            if action.action in ("click", "type", "goto"):
                await self._wait_for_page_ready()

            return info.get("success", False), info

        except Exception as e:
            logger.warning(f"Action execution failed: {e}")
            return False, {**info, "error": str(e)}

    # -----------------------------
    # Helpers
    # -----------------------------

    async def _safe_goto(self, url: str) -> bool:
        """Navigate to URL with security validation."""
        if self.sandbox_mode:
            if not validate_url(url, self.blocked_domains, self.allowed_domains):
                logger.warning(f"Blocked navigation to: {url}")
                return False
        try:
            await self._page.goto(
                url, wait_until="domcontentloaded", timeout=self.timeout_ms
            )
            return True
        except Exception as e:
            logger.warning(f"Navigation failed: {e}")
            return False

    async def _wait_for_page_ready(self) -> None:
        """Wait for DOM ready and network idle approximated."""
        page = self._page
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def step_count(self) -> int:
        return self._step_count
