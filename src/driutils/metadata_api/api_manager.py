"""Module to handle calls to the metadata API."""

import logging
from typing import Any

import requests
from httpx import HTTPError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

PAGE_SIZE = 500

_RETRYABLE_EXCEPTIONS = (HTTPError, requests.exceptions.RequestException)

api_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)


class MetadataAPIManager:
    """Manage requests to the metadata API."""

    def __init__(self, host: str, session: requests.Session | None = None) -> None:
        """
        Initialise the API Manager.

        Args:
            host: Host URL for the metadata API.
            session: Optional instance of a requests session object. If none, a default one will be created.
        """
        self.host = host
        self.session = session or requests.Session()

    @api_retry
    def make_api_call(self, url: str, params: list[tuple[str, str]] | dict[str, str] | None = None) -> dict[str, Any]:
        """Make a call to the metadata API.

        Args:
            url: The request URL.
            params: The request parameters. Defaults to None.

        Returns:
            The JSON response from the API.

        Raises:
            HTTPError: If the API request fails or returns an error.
        """
        try:
            response = self.session.get(url=url, params=params, timeout=30)
            response.raise_for_status()
            logger.debug(f"Accessed: {response.url}")
            return response.json()
        except HTTPError as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
        except ValueError:
            logger.error(f"Invalid JSON response from: {url}")
            raise

    @api_retry
    def make_paginated_api_call(
        self,
        url: str,
        params: tuple[tuple[str, str], ...] | dict[str, str] | None = None,
        page_size: int = PAGE_SIZE,
    ) -> dict[str, Any]:
        """Make a paginated call to the metadata API.

        Due to some metadata API calls not supporting pagination, an initial API call is made first. The response from
        this then provides the information required to determine if further paginated API calls are required.

        If the response indicates no pagination support by the absence of a 'limit' field in the response 'meta' data
        then the initial response is returned directly.

        Similarly, if the response indicates that pagination is supported but fewer items are present in the response
        than the provided limit (e.g. the limit is 25 but the response only contains 3 items), then the initial response
        will be returned directly.

        Args:
            url: URL to request data from
            params: Any supporting parameters to accompany the URL in the request. Defaults to None.
            page_size: The number of items to request at once when making a paginated API call. Defaults to PAGE_SIZE.

        Returns:
            The JSON response from the API. This will be the combined response if pagination is required.
        """
        # Make the initial API response to determine if further paginated API calls are required.
        initial_response = self.make_api_call(url=url, params=params)
        meta = initial_response.get("meta", {})
        items = list(initial_response.get("items", []))

        if "limit" not in meta or len(items) < meta.get("limit", 0):
            return initial_response

        # Prepare for pagination
        # Ensure params exists. Note that this isn't located at the top of the function as some API calls change their
        # behaviour if any params are provided, even if it's an empty dictionary
        if params is None:
            params = {}
        params = self._update_params(params, "_limit", page_size)
        offset = len(items)

        while True:
            params = self._update_params(params, "_offset", offset)
            next_page = self.make_api_call(url=url, params=params)
            new_items = next_page.get("items", [])
            if not new_items:
                break
            items.extend(new_items)
            offset += page_size

        return {"meta": meta, "items": items}

    @staticmethod
    def _update_params(params: dict | list[tuple], param_key: str, param_value: str | int) -> dict | list[tuple]:
        """Update query parameters, handling both dict and list-of-tuples formats."""
        if isinstance(params, dict):
            params[param_key] = param_value
        else:
            params = [(k, v) for (k, v) in params if k != param_key]
            params.append((param_key, param_value))
        return params
