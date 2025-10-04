# app.py
# Weather Odds (NASA POWER) — Streamlit + matplotlib
# requirements: pip install streamlit requests pandas numpy geopy matplotlib

import streamlit as st
import requests
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from datetime import datetime, date
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Odds from NASA POWER", layout="centered")

# ----------------- Helpers -----------------

def _init_state():
    if "place" not in st.session_state:
        st.session_state.place = "Rzeszow"
    if "lat" not in st.session_state:
        st.session_state.lat = 50.0375
    if "lon" not in st.session_state:
        st.session_state.lon = 22.0047

_init_state()

def c_to_f(x): return x * 9/5 + 32
def f_to_c(x): return (x - 32) * 5/9

@st.cache_data(show_spinner=False, ttl=3600)
def pull_power(lat, lon, y1, y2):
    """Fetch daily time series for a point from NASA POWER."""
    params = "T2M,T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR"
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={params}&start={y1}0101&end={y2}1231"
        f"&latitude={lat:.4f}&longitude={lon:.4f}&community=RE&format=JSON"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    param = js["properties"]["parameter"]
    dates = sorted(param["T2M"].keys())
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "T2M": [param["T2M"][d] for d in dates],
        "T2M_MAX": [param["T2M_MAX"][d] for d in dates],
        "T2M_MIN": [param["T2M_MIN"][d] for d in dates],
        "RH2M": [param["RH2M"][d] for d in dates],
        "WS10M": [param["WS10M"][d] for d in dates],
        "PRECTOTCORR": [param["PRECTOTCORR"][d] for d in dates],
    })
    return df, url

def compute_heat_index_c(t_c, rh):
    """Approx. Heat Index in °C; inputs T(°C), RH(%)."""
    t_f = t_c * 9/5 + 32
    HI_f = (-42.379 + 2.04901523*t_f + 10.14333127*rh - 0.22475541*t_f*rh
            - 0.00683783*t_f*t_f - 0.05481717*rh*rh
            + 0.00122874*t_f*t_f*rh + 0.00085282*t_f*rh*rh
            - 0.00000199*t_f*t_f*rh*rh)
    HI_c = (HI_f - 32) * 5/9
    return np.where(t_c < 26, t_c, HI_c)

def filter_by_doy_window(df, target_date, half_window):
    """Filter by day-of-year window with 29 Feb handling and year wrap."""
    df = df.copy()
    d = target_date
    if isinstance(d, datetime):
        d = d.date()
    if d.month == 2 and d.day == 29:
        d = d.replace(day=28)
    tgt = datetime(d.year, d.month, d.day).timetuple().tm_yday
    df["doy"] = df["date"].dt.dayofyear.clip(upper=365)

    def in_win(x):
        return min(abs(x - tgt), abs(x + 365 - tgt), abs(x - 365 - tgt)) <= half_window

    return df[df["doy"].apply(in_win)]

def geocode_place(q):
    geolocator = Nominatim(user_agent="weather_odds_app")
    loc = geolocator.geocode(q, timeout=10)
    if loc:
        return loc.latitude, loc.longitude, loc.address
    return None, None, None

# ----------------- UI -----------------

st.title("Weather Odds from NASA POWER (historical probabilities)")
st.caption(
    "Pick a location and day of year to see the odds of 'very hot / very cold / very windy / very wet / very uncomfortable'. "
    "Data source: NASA POWER (Langley Research Center)."
)

with st.sidebar:
    st.header("Inputs")
    st.text_input("Place (city, address)", key="place")

    unit = st.radio("Temperature unit", options=["°C", "°F"], index=0, horizontal=True)

    colA, _ = st.columns(2)
    with colA:
        if st.button("Geocode"):
            try:
                lat, lon, addr = geocode_place(st.session_state.place)
                if lat is not None:
                    st.session_state.lat = float(lat)
                    st.session_state.lon = float(lon)
                    st.success(f"Found: {addr}")
                else:
                    st.warning("Not found — please enter coordinates manually.")
            except Exception as e:
                st.warning(f"Geocoder not available: {e}")

    st.number_input("Latitude (°)", key="lat", format="%.6f")
    st.number_input("Longitude (°)", key="lon", format="%.6f")

    the_date = st.date_input("Day of year", value=date.today())
    window_days = st.slider("Window ± days around date", min_value=0, max_value=30, value=7)

    st.markdown("---")
    st.subheader("Adjustable thresholds")

    # Display thresholds in the selected unit; convert back to °C for calc
    if unit == "°C":
        th_hot_ui  = st.number_input("Very hot: T2M_MAX > [°C]", value=32)
        th_cold_ui = st.number_input("Very cold: T2M_MIN < [°C]", value=0)
        th_hi_ui   = st.number_input("Very uncomfortable (Heat Index) ≥ [°C]", value=32)
    else:
        th_hot_ui  = st.number_input("Very hot: T2M_MAX > [°F]", value=90)
        th_cold_ui = st.number_input("Very cold: T2M_MIN < [°F]", value=32)
        th_hi_ui   = st.number_input("Very uncomfortable (Heat Index) ≥ [°F]", value=90)

    th_wind = st.number_input("Very windy: WS10M ≥ [m/s]", value=10)
    th_rain = st.number_input("Very wet: PRECTOTCORR ≥ [mm/day]", value=10)

    # Convert UI thresholds back to °C for calculations
    th_hot_c  = th_hot_ui if unit == "°C" else f_to_c(th_hot_ui)
    th_cold_c = th_cold_ui if unit == "°C" else f_to_c(th_cold_ui)
    th_hi_c   = th_hi_ui if unit == "°C" else f_to_c(th_hi_ui)

    st.markdown("---")
    year_now = datetime.now().year
    start_year = st.number_input("Start year", value=1995, min_value=1981, max_value=year_now)
    end_year   = st.number_input("End year", value=max(1995, year_now-1),
                                 min_value=start_year, max_value=year_now)

run = st.button("Compute probabilities")

# ----------------- Logic -----------------

if run:
    try:
        with st.spinner("Fetching NASA POWER & computing…"):
            df, api_url = pull_power(st.session_state.lat, st.session_state.lon, int(start_year), int(end_year))
            sub = filter_by_doy_window(df, the_date, int(window_days))

            st.caption("NASA POWER endpoint used:")
            st.code(api_url, language="text")

            if sub.empty:
                st.error("No data in the selected window.")
                st.stop()

            sub["HI"] = compute_heat_index_c(sub["T2M"], sub["RH2M"])
            n = len(sub)

            probs = {
                "Very hot": (sub["T2M_MAX"] > th_hot_c).mean()*100,
                "Very cold": (sub["T2M_MIN"] < th_cold_c).mean()*100,
                "Very windy": (sub["WS10M"] >= th_wind).mean()*100,
                "Very wet": (sub["PRECTOTCORR"] >= th_rain).mean()*100,
                "Very uncomfortable": (sub["HI"] >= th_hi_c).mean()*100,
            }

            # --- TABLE + BAR CHART (matplotlib) ---
            out = pd.DataFrame({
                "Condition": list(probs.keys()),
                "Probability [%]": list(probs.values())
            })
            out["Probability [%]"] = pd.to_numeric(out["Probability [%]"], errors="coerce").fillna(0.0).round(1)

            st.subheader("Historical odds")
            st.dataframe(out, use_container_width=True)

            labels = out["Condition"].tolist()
            vals = out["Probability [%]"].astype(float).tolist()
            ymax = max(10.0, max(vals) * 1.2 if vals else 10.0)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(labels, vals)
            ax.set_ylabel("Probability [%]")
            ax.set_ylim(0, ymax)
            for i, v in enumerate(vals):
                ax.text(i, v + ymax*0.02, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            # --- YEARLY TREND: "VERY WET" (matplotlib) ---
            sub["year"] = sub["date"].dt.year
            vw_by_year = (
                sub.groupby("year")["PRECTOTCORR"]
                   .apply(lambda s: (s >= th_rain).mean() * 100.0)
                   .reset_index()
                   .rename(columns={"PRECTOTCORR": "Very wet [%]"})
            )
            vw_by_year["Very wet [%]"] = pd.to_numeric(vw_by_year["Very wet [%]"], errors="coerce").fillna(0.0)

            st.caption("Yearly trend: share of 'Very wet' days in the ±N-day window (%)")

            years = vw_by_year["year"].astype(int).tolist()
            vals2 = vw_by_year["Very wet [%]"].astype(float).tolist()
            ymax2 = max(10.0, max(vals2) * 1.2 if vals2 else 10.0)

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(years, vals2, marker="o")
            ax2.set_xlabel("year")
            ax2.set_ylabel("Very wet [%]")
            ax2.set_ylim(0, ymax2)
            for x, y in zip(years, vals2):
                ax2.text(x, y + ymax2*0.02, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)

            # --- EXPORT WITH METADATA ---
            sub_export = sub.copy()
            sub_export["SOURCE"] = "NASA POWER"
            sub_export["API_URL"] = api_url
            sub_export["LAT"] = st.session_state.lat
            sub_export["LON"] = st.session_state.lon
            sub_export["YEARS"] = f"{start_year}-{end_year}"
            sub_export["TEMP_UNIT_UI"] = unit  # record which unit the user selected

            csv = sub_export.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (subset + metadata)", data=csv,
                               file_name="nasa_power_subset.csv", mime="text/csv")

            st.success(f"Sample size: {n} days (years {start_year}–{end_year}).")

            if any(v == 0 for v in probs.values()):
                st.info("Seeing many 0% values? Increase the ±day window, relax thresholds, or widen the year range.")

    except Exception as e:
        st.error(f"Oops — something went wrong: {e}")

# ----------------- Footer -----------------
st.caption(
    "Note: this is historical statistics from NASA's gridded data (not a forecast). "
    "Wind is daily mean (not gusts)."
)
