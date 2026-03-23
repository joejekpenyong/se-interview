import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import get_exchange_rate, get_weather, build_agent
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


# ── Currency converter tests ──────────────────────────────────────────────────

class TestExchangeRateTool(unittest.TestCase):

    def setUp(self):
        os.environ["EXCHANGE_RATE_API_KEY"] = "test_api_key"

    def tearDown(self):
        if "EXCHANGE_RATE_API_KEY" in os.environ:
            del os.environ["EXCHANGE_RATE_API_KEY"]

    @patch('agent.requests.get')
    def test_successful_exchange_rate(self, mock_get):
        """Test successful currency conversion returns correct structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": "success",
            "conversion_rate": 1.5,
            "time_last_update_utc": "Mon, 23 Mar 2026 12:00:00 +0000"
        }
        mock_get.return_value = mock_response

        result = get_exchange_rate.func("USD", "EUR")

        self.assertEqual(result["base_currency"], "USD")
        self.assertEqual(result["target_currency"], "EUR")
        self.assertEqual(result["exchange_rate"], 1.5)
        self.assertIn("last_updated", result)
        self.assertIn("disclaimer", result)
        mock_get.assert_called_once_with(
            f"https://v6.exchangerate-api.com/v6/test_api_key/pair/USD/EUR"
        )

    @patch('agent.requests.get')
    def test_exchange_rate_api_error(self, mock_get):
        """Test that API errors are handled gracefully."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": "error",
            "error-type": "invalid-key"
        }
        mock_get.return_value = mock_response

        result = get_exchange_rate.func("USD", "INVALID")

        self.assertIn("error", result)
        self.assertIn("invalid-key", result["error"])

    @patch('agent.requests.get')
    def test_exchange_rate_uppercases_currencies(self, mock_get):
        """Test that currency codes are uppercased regardless of input."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": "success",
            "conversion_rate": 1.5,
            "time_last_update_utc": "Mon, 23 Mar 2026 12:00:00 +0000"
        }
        mock_get.return_value = mock_response

        result = get_exchange_rate.func("usd", "eur")

        self.assertEqual(result["base_currency"], "USD")
        self.assertEqual(result["target_currency"], "EUR")

    @patch('agent.requests.get')
    def test_exchange_rate_missing_api_key(self, mock_get):
        """Test behavior when API key is missing."""
        del os.environ["EXCHANGE_RATE_API_KEY"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": "error",
            "error-type": "missing-key"
        }
        mock_get.return_value = mock_response

        result = get_exchange_rate.func("USD", "EUR")

        mock_get.assert_called_once()


# ── Weather tool tests ────────────────────────────────────────────────────────

class TestWeatherTool(unittest.TestCase):

    def setUp(self):
        os.environ["OPENWEATHER_API_KEY"] = "test_weather_api_key"

    def tearDown(self):
        if "OPENWEATHER_API_KEY" in os.environ:
            del os.environ["OPENWEATHER_API_KEY"]

    @patch('agent.requests.get')
    def test_successful_weather_lookup(self, mock_get):
        """Test successful weather lookup returns correct structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 72
            },
            "weather": [{"description": "light rain"}],
            "wind": {"speed": 3.5}
        }
        mock_get.return_value = mock_response

        result = get_weather.func("London")

        self.assertEqual(result["city"], "London")
        self.assertEqual(result["country"], "GB")
        self.assertEqual(result["temperature_celsius"], 15.5)
        self.assertEqual(result["feels_like_celsius"], 14.2)
        self.assertEqual(result["condition"], "light rain")
        self.assertEqual(result["humidity_percent"], 72)
        self.assertEqual(result["wind_speed_mps"], 3.5)
        mock_get.assert_called_once_with(
            f"https://api.openweathermap.org/data/2.5/weather?q=London&appid=test_weather_api_key&units=metric"
        )

    @patch('agent.requests.get')
    def test_weather_city_not_found(self, mock_get):
        """Test that invalid cities return an error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "message": "city not found"
        }
        mock_get.return_value = mock_response

        result = get_weather.func("InvalidCity")

        self.assertIn("error", result)
        self.assertIn("city not found", result["error"])

    @patch('agent.requests.get')
    def test_weather_missing_api_key(self, mock_get):
        """Test behavior when API key is missing."""
        del os.environ["OPENWEATHER_API_KEY"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "TestCity",
            "sys": {"country": "US"},
            "main": {"temp": 20.0, "feels_like": 19.0, "humidity": 50},
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 2.0}
            }
        mock_get.return_value = mock_response

        result = get_weather.func("TestCity")

        mock_get.assert_called_once()


# ── Agent structure tests ─────────────────────────────────────────────────────

class TestAgentStructure(unittest.TestCase):

    def test_build_agent_returns_compiled_graph(self):
        """Test that build_agent returns a runnable compiled graph."""
        agent = build_agent()
        self.assertIsNotNone(agent)

    def test_agent_has_correct_nodes(self):
        """Test that the agent graph contains the expected nodes."""
        agent = build_agent()
        nodes = list(agent.get_graph().nodes.keys())
        self.assertIn("llm_call", nodes)
        self.assertIn("tool_node", nodes)


# ── API endpoint tests ────────────────────────────────────────────────────────

class TestAPIEndpoints(unittest.TestCase):

    def test_health_endpoint(self):
        """Test that health endpoint returns ok status."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_endpoint_rejects_missing_message(self):
        """Test that chat endpoint returns 422 when message is missing."""
        response = client.post("/chat", json={})
        self.assertEqual(response.status_code, 422)

    def test_chat_endpoint_uses_default_thread_id(self):
        """Test that chat endpoint accepts requests without thread_id."""
        with patch("api.agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content="Test response")]
            }
            response = client.post("/chat", json={"message": "Hello"})
            self.assertEqual(response.status_code, 200)
            call_args = mock_agent.invoke.call_args
            config = call_args[0][1]
            self.assertEqual(
                config["configurable"]["thread_id"], "default"
            )


if __name__ == '__main__':
    unittest.main()