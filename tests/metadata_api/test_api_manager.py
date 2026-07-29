import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from httpx import HTTPError, HTTPStatusError, Request, Response, TimeoutException

from driutils.metadata_api.api_manager import MetadataAPIManager
from driutils.testing_utils.mock_metadata_api import MockMetadataAPI


@pytest.fixture
def api() -> MetadataAPIManager:
    api = MetadataAPIManager(host="test_url.com")
    return api


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip tenacity's real backoff sleeps so retry tests run instantly."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


class TestMetadataApiManager:
    """Test the MetadataAPIManager class."""

    mock_response_data = {
        "meta": {
            "@id": "http://fdri.ceh.ac.uk/id/network/cosmos.json",
            "publisher": "UK Centre for Ecology & Hydrology",
            "license": "http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
            "licenseName": "OGL 3",
            "comment": "",
            "version": "1.0.0",
        },
        "items": [
            {
                "@id": "http://fdri.ceh.ac.uk/id/network/cosmos",
                "contains": [{"@id": "http://fdri.ceh.ac.uk/id/site/cosmos-sheep", "label": ["Sheepdrove"]}],
            }
        ],
    }

    @pytest.mark.asyncio
    def test_make_api_call_success(self, api: MetadataAPIManager) -> None:
        """Test successful API response."""
        with patch("requests.Session.get") as mock_get:
            mock_request = Request(method="get", url="http://test_url.com")
            mock_response = Response(200, json=self.mock_response_data, request=mock_request)

            mock_get.return_value = mock_response

            result = api.make_api_call("http://test_url.com")

            assert result == self.mock_response_data

    @pytest.mark.asyncio
    def test_make_api_call_general_error(self, api: MetadataAPIManager) -> None:
        """Test handling of general errors."""
        with patch("requests.Session.get") as mock_get:
            mock_error = HTTPError("API Error")
            mock_get.side_effect = mock_error

            with pytest.raises(HTTPError, match="API Error"):
                api.make_api_call("test_url.com")

    @pytest.mark.asyncio
    async def test_make_api_call_404_error(self, api: MetadataAPIManager) -> None:
        """Test handling of 404 errors."""
        with patch("requests.Session.get") as mock_get:
            mock_error = HTTPStatusError("404 Error", request="test_request", response=404)
            mock_get.side_effect = mock_error

            with pytest.raises(HTTPError, match="404 Error"):
                api.make_api_call("test_url.com")

    @pytest.mark.asyncio
    async def test_make_api_call_timeout_error(self, api: MetadataAPIManager) -> None:
        """Test handling of timeout errors."""
        with patch("requests.Session.get") as mock_get:
            mock_error = TimeoutException("timed_out")
            mock_get.side_effect = mock_error

            with pytest.raises(HTTPError, match="timed_out"):
                api.make_api_call("test_url.com")

    @pytest.mark.asyncio
    async def test_make_api_call_invalid_json(self, api: MetadataAPIManager) -> None:
        """Test handling of invalid JSON response."""
        with patch("requests.Session.get") as mock_get:
            mock_request = Request(method="get", url="test_url.com")
            mock_response = Response(200, json=self.mock_response_data, request=mock_request)
            mock_response._content = b"invalid json"
            mock_get.return_value = mock_response

            with pytest.raises(json.JSONDecodeError):
                api.make_api_call("test_url.com")

            # Invalid JSON is not a transient failure, so it should not be retried.
            assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_make_api_call_retries_then_succeeds(self, api: MetadataAPIManager) -> None:
        """Test the call succeeds once a transient connection error clears on retry."""
        with patch("requests.Session.get") as mock_get:
            mock_request = Request(method="get", url="http://test_url.com")
            mock_response = Response(200, json=self.mock_response_data, request=mock_request)

            mock_get.side_effect = [
                requests.exceptions.ConnectionError("boom"),
                requests.exceptions.ConnectionError("boom"),
                mock_response,
            ]

            result = api.make_api_call("http://test_url.com")

            assert result == self.mock_response_data
            assert mock_get.call_count == 3

    @pytest.mark.asyncio
    async def test_make_api_call_retries_exhausted(self, api: MetadataAPIManager) -> None:
        """Test a persistent transient error is retried up to the limit before raising."""
        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("boom")

            with pytest.raises(requests.exceptions.ConnectionError, match="boom"):
                api.make_api_call("http://test_url.com")

            # 1 initial attempt + 3 retries
            assert mock_get.call_count == 4


@patch.object(MetadataAPIManager, "make_api_call")
class TestPaginatedAPICall:
    host_url = "test_url.com"

    @pytest.mark.asyncio
    def test_no_pagination_required_limit_field_not_present(
        self, mock_metadata_api: MagicMock, api: MetadataAPIManager
    ) -> None:
        """Check the entire API response is returned if no pagination is required."""
        expected_response = {"meta": {}, "items": [{"key_1": "value_1"}]}
        mock_api_data = {self.host_url: expected_response}
        mock_metadata_api.side_effect = MockMetadataAPI(api_data=mock_api_data)

        result = api.make_paginated_api_call(self.host_url)

        assert result == expected_response
        assert mock_metadata_api.call_count == 1

    @pytest.mark.asyncio
    def test_no_pagination_required_num_items_below_limit(
        self, mock_metadata_api: MagicMock, api: MetadataAPIManager
    ) -> None:
        """Check no extra pagination calls are made if fewer items than the page size limit are returned."""
        expected_response = {"meta": {"limit": 2}, "items": [{"key_1": "value_1"}]}
        mock_api_data = {self.host_url: expected_response}
        mock_metadata_api.side_effect = MockMetadataAPI(api_data=mock_api_data)

        result = api.make_paginated_api_call(self.host_url)

        assert result == expected_response
        assert mock_metadata_api.call_count == 1

    @pytest.mark.asyncio
    def test_pagination_required(self, mock_metadata_api: MagicMock, api: MetadataAPIManager) -> None:
        """Check the pagination logic is called when required."""
        first_page = {
            "meta": {"limit": 2},
            "items": [{"key_1": "value_1"}, {"key_2": "value_2"}],
        }
        second_page = {
            "meta": {"limit": 2},
            "items": [{"key_3", "value_3"}],
        }
        third_page = {
            "meta": {"limit": 2},
            "items": [],
        }

        expected_response = {
            "meta": {"limit": 2},
            "items": [{"key_1": "value_1"}, {"key_2": "value_2"}, {"key_3", "value_3"}],
        }

        mock_metadata_api.side_effect = [first_page, second_page, third_page]
        result = api.make_paginated_api_call(self.host_url, page_size=2)

        assert result == expected_response
        assert mock_metadata_api.call_count == 3
