# app.py
# Weather Odds (NASA POWER) — Streamlit + matplotlib (clean visuals, better geocoding)
# requirements: pip install streamlit requests pandas numpy geopy matplotlib

import streamlit as st
import requests
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from datetime import datetime, date
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

st.set_page_config(page_title="Weather Odds from NASA POWER", layout="centered")

# ----------------- Helpers -----------------

def _init_state():
    ss = st.session_state
    ss.setdefault("place", "Rzeszow")
    ss.setdefault("lat", 50.0375)
    ss.setdefault("lon", 22.0047)
    ss.setdefault("candidates", [])
    ss.setdefault("cand_index", 0)

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
    t_f = c_to_f(t_c)
    HI_f = (-42.379 + 2.04901523*t_f + 10.14333127*rh - 0.22475541*t_f*rh
            - 0.00683783*t_f*t_f - 0.05481717*rh*rh
            + 0.00122874*t_f*t_f*rh + 0.00085282*t_f*rh*rh
            - 0.00000199*t_f*t_f*rh*rh)
    HI_c = (HI_f - 32) * 5/9
    return np.where(t_c < 26, t_c, HI_c)

def filter_by_doy_window(df, target_date, half_window):
    """Filter by day-of-year window with 29 Feb handling and year wrap."""
    df = df.copy()
    d = target_date.date() if isinstance(target_date, datetime) else target_date
    if d.month == 2 and d.day == 29:
        d = d.replace(day=28)
    tgt = datetime(d.year, d.month, d.day).timetuple().tm_yday
    df["doy"] = df["date"].dt.dayofyear.clip(upper=365)

    def in_win(x):
        return min(abs(x - tgt), abs(x + 365 - tgt), abs(x - 365 - tgt)) <= half_window

    return df[df["doy"].apply(in_win)]

# ----- Improved geocoding -----
def geocode_place(q, country_code=None, limit=7):
    """
    Return a list of candidate locations for query q.
    Prefer only place types (city/town/village/hamlet), fall back to any.
    """
    geolocator = Nominatim(user_agent="weather_odds_app")
    results = geolocator.geocode(
        q, timeout=10, exactly_one=False, limit=limit,
        country_codes=country_code, language="en", addressdetails=True
    ) or []

    def is_place(r):
        raw = getattr(r, "raw", {})
        return raw.get("class") == "place" and raw.get("type") in {
            "city", "town", "village", "hamlet", "municipality", "locality"
        }

    place_like = [r for r in results if is_place(r)]
    return place_like or results

# ----------------- UI -----------------

st.title("Weather Odds from NASA POWER (historical probabilities)")
st.caption(
    "Pick a location and day of year to see the odds of 'very hot / very cold / very windy / very wet / very uncomfortable'. "
    "Data source: NASA POWER (Langley Research Center)."
)

with st.sidebar:
    st.header("Inputs")
    st.text_input("Place (city, address)", key="place")

    # Better geocoding controls
    country_bias = st.text_input("Country bias (ISO-2, optional e.g. pk, pl, de)", value="")
    if st.button("Geocode"):
        try:
            cands = geocode_place(st.session_state.place, (country_bias.strip().lower() or None))
            st.session_state.candidates = [
                {
                    "label": f"{c.address}  ({c.latitude:.4f}, {c.longitude:.4f})",
                    "lat": float(c.latitude), "lon": float(c.longitude), "addr": c.address,
                } for c in cands
            ]
            if not st.session_state.candidates:
                st.warning("No results — try e.g. 'Turbat, Pakistan' or set ISO-2 code above.")
        except Exception as e:
            st.warning(f"Geocoder not available: {e}")

    if st.session_state.candidates:
        st.caption("Pick a geocoding result:")
        labels = [c["label"] for c in st.session_state.candidates]
        st.session_state.cand_index = st.selectbox(
            "Candidates", range(len(labels)), index=min(st.session_state.cand_index, len(labels)-1),
            format_func=lambda i: labels[i]
        )
        if st.button("Use selected"):
            sel = st.session_state.candidates[st.session_state.cand_index]
            st.session_state.lat = sel["lat"]
            st.session_state.lon = sel["lon"]
            st.success(f"Selected: {sel['addr']}\nLat/Lon: {sel['lat']:.4f}, {sel['lon']:.4f}")

    unit = st.radio("Temperature unit", options=["°C", "°F"], index=0, horizontal=True)

    st.number_input("Latitude (°)", key="lat", format="%.6f")
    st.number_input("Longitude (°)", key="lon", format="%.6f")

    the_date = st.date_input("Day of year", value=date.today())
    window_days = st.slider("Window ± days around date", min_value=0, max_value=30, value=7)

    st.markdown("---")
    st.subheader("Adjustable thresholds")
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

    # Convert UI thresholds back to °C for data calc
    th_hot_c  = th_hot_ui if unit == "°C" else f_to_c(th_hot_ui)
    th_cold_c = th_cold_ui if unit == "°C" else f_to_c(th_cold_ui)
    th_hi_c   = th_hi_ui if unit == "°C" else f_to_c(th_hi_ui)

    st.markdown("---")
    year_now = datetime.now().year
    start_year = st.number_input("Start year", value=1995, min_value=1981, max_value=year_now)
    end_year   = st.number_input("End year", value=max(1995, year_now-1),
                                 min_value=start_year, max_value=year_now)

    st.markdown("---")
    st.subheader("Chart options")
    show_bar_labels = st.checkbox("Show bar value labels", value=True)
    smooth_trend = st.checkbox("Add rolling average (yearly trend)", value=True)
    smooth_window = st.slider("Rolling window (years)", 2, 7, 3) if smooth_trend else 0
    trend_metric = st.selectbox("Trend metric", ["Wet", "Hot", "Cold", "Windy", "Uncomfort."])

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
                "Hot": (sub["T2M_MAX"] > th_hot_c).mean()*100,
                "Cold": (sub["T2M_MIN"] < th_cold_c).mean()*100,
                "Windy": (sub["WS10M"] >= th_wind).mean()*100,
                "Wet": (sub["PRECTOTCORR"] >= th_rain).mean()*100,
                "Uncomfort.": (sub["HI"] >= th_hi_c).mean()*100,
            }

            # --- TABLE ---
            out = pd.DataFrame({
                "Condition": list(probs.keys()),
                "Probability [%]": list(probs.values())
            })
            out["Probability [%]"] = pd.to_numeric(out["Probability [%]"], errors="coerce").fillna(0.0).round(1)

            st.subheader("Historical odds")
            st.dataframe(out, use_container_width=True)

            # --- CLEAN BAR CHART ---
            labels = out["Condition"].tolist()
            vals = out["Probability [%]"].astype(float).tolist()
            ymax = max(10.0, max(vals) * 1.25 if vals else 10.0)

            fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=150)
            bars = ax.bar(labels, vals)

            ax.set_ylabel("Probability [%]", fontsize=11)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelrotation=0, labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

            if show_bar_labels:
                for rect, v in zip(bars, vals):
                    if v >= 0.05:
                        ax.text(rect.get_x() + rect.get_width()/2, v + ymax*0.015,
                                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)

            # --- YEARLY TREND (selectable metric) ---
            sub["year"] = sub["date"].dt.year
            metric_map = {
                "Wet":        (sub["PRECTOTCORR"] >= th_rain),
                "Hot":        (sub["T2M_MAX"] >  th_hot_c),
                "Cold":       (sub["T2M_MIN"] <  th_cold_c),
                "Windy":      (sub["WS10M"]   >= th_wind),
                "Uncomfort.": (sub["HI"]      >= th_hi_c),
            }
            sel_mask = metric_map[trend_metric]

            trend_by_year = (
                sub.assign(flag=sel_mask)
                   .groupby("year")["flag"].mean()
                   .mul(100.0)
                   .reset_index()
                   .rename(columns={"flag": f"{trend_metric} [%]"})
            )

            st.caption(f"Yearly trend: share of '{trend_metric}' days in the ±N-day window (%)")

            years = trend_by_year["year"].astype(int)
            vals2 = trend_by_year[f"{trend_metric} [%]"].astype(float)

            if not years.empty:
                ymax2 = max(10.0, vals2.max() * 1.25)
                fig2, ax2 = plt.subplots(figsize=(8.5, 4.0), dpi=150)
                ax2.plot(years, vals2, marker="o", linewidth=1.8, markersize=4, label="Yearly value")

                if smooth_trend and len(vals2) >= smooth_window:
                    smoothed = vals2.rolling(window=smooth_window, center=True, min_periods=1).mean()
                    ax2.plot(years, smoothed, linewidth=2.2, alpha=0.8, label=f"Rolling {smooth_window}-yr avg")

                ax2.set_xlabel("year", fontsize=11)
                ax2.set_ylabel(f"{trend_metric} [%]", fontsize=11)
                ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                ax2.set_ylim(0, ymax2)
                ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
                ax2.tick_params(axis="both", labelsize=10)
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                ax2.scatter([years.iloc[-1]], [vals2.iloc[-1]], s=46, zorder=3)
                ax2.legend(frameon=False, fontsize=9, loc="upper left")
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("No yearly data to plot.")

            # --- EXPORT WITH METADATA ---
            sub_export = sub.copy()
            sub_export["SOURCE"] = "NASA POWER"
            sub_export["API_URL"] = api_url
            sub_export["LAT"] = st.session_state.lat
            sub_export["LON"] = st.session_state.lon
            sub_export["YEARS"] = f"{start_year}-{end_year}"
            sub_export["TEMP_UNIT_UI"] = unit

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
