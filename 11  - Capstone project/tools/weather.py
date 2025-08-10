from typing import Dict, Any
import requests

def get_weather_for_city(city: str) -> Dict[str, Any]:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "ru", "format": "json"},
        timeout=10
    ).json()
    if not geo.get("results"):
        return {"ok": False, "error": "Город не найден"}

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "forecast_days": 1, "timezone": "auto"},
        timeout=10
    ).json()
    hours = wx.get("hourly", {}).get("time", [])
    temps = wx.get("hourly", {}).get("temperature_2m", [])
    sample = [{"time": t, "temp_c": temp} for t, temp in list(zip(hours, temps))[:6]]
    return {"ok": True, "city": city, "sample": sample}
