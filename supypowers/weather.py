# /// script
# dependencies = ["pydantic", "httpx"]
# ///
import httpx
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """Input for weather forecast."""
    location: str = Field(..., description="City name or location")
    days: int = Field(default=7, description="Number of days to forecast (1-14)")

class WeatherOutput(BaseModel):
    """Output for weather forecast."""
    ok: bool
    location: str | None = None
    forecast: list[dict] | None = None
    error: str | None = None

def get_weather(input: WeatherInput) -> WeatherOutput:
    """
    Get weather forecast for a location using Open-Meteo API (free, no API key needed).
    
    Examples:
        >>> get_weather({"location": "London", "days": 7})
    """
    try:
        # First, geocode the location to get coordinates
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={input.location}&count=1&language=en&format=json"
        geo_resp = httpx.get(geo_url, timeout=30)
        geo_data = geo_resp.json()

        if "results" not in geo_data or not geo_data["results"]:
            return WeatherOutput(ok=False, error=f"Location '{input.location}' not found")

        result = geo_data["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        location_name = f"{result['name']}, {result.get('country', '')}"

        # Get weather forecast
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,"
            f"precipitation_probability_mean,weather_code&timezone=auto&forecast_days={input.days}"
        )
        weather_resp = httpx.get(weather_url, timeout=30)
        weather_data = weather_resp.json()

        daily = weather_data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precip_probs = daily.get("precipitation_probability_mean", [])
        weather_codes = daily.get("weather_code", [])

        # Weather code mapping (WMO codes)
        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Rain showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail"
        }

        forecast = []
        for i in range(len(dates)):
            code = weather_codes[i] if i < len(weather_codes) else 0
            forecast.append({
                "date": dates[i],
                "max_temp_c": max_temps[i] if i < len(max_temps) else None,
                "min_temp_c": min_temps[i] if i < len(min_temps) else None,
                "precipitation_chance": precip_probs[i] if i < len(precip_probs) else None,
                "condition": weather_descriptions.get(code, f"Code {code}")
            })

        return WeatherOutput(ok=True, location=location_name, forecast=forecast)

    except Exception as e:
        return WeatherOutput(ok=False, error=str(e))
