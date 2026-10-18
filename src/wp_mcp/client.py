"""Async client for the WordPress REST API with retries on transient failures."""

import asyncio
import logging
from typing import Any, Self

import httpx

from wp_mcp.errors import WordPressError
from wp_mcp.models import CreateDraftInput, DraftCreated, PostDetail, PostSummary, Term
from wp_mcp.settings import WPSettings

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
# Errors where the request never reached WordPress, so even a POST is safe to resend.
_NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class WordPressClient:
    def __init__(self, http: httpx.AsyncClient, settings: WPSettings) -> None:
        self._http = http
        self._settings = settings
        self._site = settings.base_url.rstrip("/")

    @classmethod
    def from_settings(cls, settings: WPSettings) -> Self:
        http = httpx.AsyncClient(
            base_url=f"{settings.base_url.rstrip('/')}/wp-json/wp/v2",
            auth=httpx.BasicAuth(settings.username, settings.app_password.get_secret_value()),
            timeout=settings.timeout_s,
        )
        return cls(http, settings)

    async def aclose(self) -> None:
        await self._http.aclose()

    def edit_url(self, post_id: int) -> str:
        return f"{self._site}/wp-admin/post.php?post={post_id}&action=edit"

    async def list_posts(self, search: str = "", per_page: int = 10) -> list[PostSummary]:
        params: dict[str, Any] = {
            "per_page": per_page,
            "status": "publish,draft",
            "_fields": "id,title,link,status",
        }
        if search:
            params["search"] = search
        data = await self._request("GET", "/posts", params=params)
        return [PostSummary.model_validate(item) for item in data]

    async def list_categories(self) -> list[Term]:
        params = {"per_page": 100, "_fields": "id,name,slug"}
        data = await self._request("GET", "/categories", params=params)
        return [Term.model_validate(item) for item in data]

    async def list_tags(self, search: str = "") -> list[Term]:
        params: dict[str, Any] = {"per_page": 100, "_fields": "id,name,slug"}
        if search:
            params["search"] = search
        data = await self._request("GET", "/tags", params=params)
        return [Term.model_validate(item) for item in data]

    async def create_tag(self, name: str) -> Term:
        try:
            data = await self._request("POST", "/tags", json={"name": name})
        except WordPressError as exc:
            term_id = exc.data.get("term_id")
            if exc.status_code == 400 and exc.code == "term_exists" and term_id is not None:
                data = await self._request("GET", f"/tags/{term_id}")
            else:
                raise
        return Term.model_validate(data)

    async def create_draft_post(self, draft: CreateDraftInput) -> DraftCreated:
        body: dict[str, Any] = {
            "title": draft.title,
            "content": draft.content_html,
            "status": "draft",
            "categories": draft.categories,
            "tags": draft.tags,
        }
        if draft.slug:
            body["slug"] = draft.slug
        if draft.excerpt:
            body["excerpt"] = draft.excerpt
        if draft.meta_description:
            body["meta"] = {self._settings.meta_description_key: draft.meta_description}
        data = await self._request("POST", "/posts", json=body)
        return DraftCreated(
            id=data["id"],
            status=data["status"],
            link=data["link"],
            edit_url=self.edit_url(data["id"]),
        )

    async def get_post(self, post_id: int) -> PostDetail:
        data = await self._request("GET", f"/posts/{post_id}", params={"context": "edit"})
        return PostDetail.model_validate({**data, "edit_url": self.edit_url(data["id"])})

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        # A POST that timed out or got a 5xx may still have created the post or tag,
        # so retrying it could create duplicates.
        idempotent = method != "POST"
        attempts = self._settings.max_retries + 1
        last: WordPressError | None = None
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(self._settings.backoff_s * 2 ** (attempt - 1))
            try:
                response = await self._http.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last = WordPressError(f"request timed out: {path}", code="timeout")
                last.__cause__ = exc
                if not (idempotent or isinstance(exc, _NOT_SENT_ERRORS)):
                    raise last from exc
            except httpx.TransportError as exc:
                last = WordPressError(f"connection failed: {exc}", code="connection_error")
                last.__cause__ = exc
                if not (idempotent or isinstance(exc, _NOT_SENT_ERRORS)):
                    raise last from exc
            else:
                if response.is_success:
                    return response.json()
                last = WordPressError.from_payload(response.status_code, _json_or_none(response))
                retryable = response.status_code in _TRANSIENT_STATUS and (
                    idempotent or response.status_code == 429
                )
                if not retryable:
                    raise last
            logger.warning(
                "transient failure on %s %s (attempt %d/%d): %s",
                method,
                path,
                attempt + 1,
                attempts,
                last,
            )
        assert last is not None
        raise last


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None
