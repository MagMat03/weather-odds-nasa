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

@st.cache_data(show_spinner=False, ttl=3600)
def pull_power(lat, lon, y1, y2):
    """Pobierz dzienne szeregi dla punktu z NASA POWER."""
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
    """Przybliżony Heat Index (°C) z T(°C), RH(%)."""
    t_f = t_c * 9/5 + 32
    HI_f = (-42.379 + 2.04901523*t_f + 10.14333127*rh - 0.22475541*t_f*rh
            - 0.00683783*t_f*t_f - 0.05481717*rh*rh
            + 0.00122874*t_f*t_f*rh + 0.00085282*t_f*rh*rh
            - 0.00000199*t_f*t_f*rh*rh)
    HI_c = (HI_f - 32) * 5/9
    return np.where(t_c < 26, t_c, HI_c)

def filter_by_doy_window(df, target_date, half_window):
    """Okno względem dnia roku z obsługą 29 II i zawijania roku."""
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
st.caption("Wybierz lokalizację i dzień roku, a dostaniesz szanse na ‘very hot/cold/windy/wet/uncomfortable’. "
           "Źródło danych: NASA POWER (Langley Research Center).")

with st.sidebar:
    st.header("Wejście")
    st.text_input("Miejsce (miasto, adres)", key="place")

    colA, _ = st.columns(2)
    with colA:
        if st.button("Geokoduj"):
            try:
                lat, lon, addr = geocode_place(st.session_state.place)
                if lat is not None:
                    st.session_state.lat = float(lat)
                    st.session_state.lon = float(lon)
                    st.success(f"Znaleziono: {addr}")
                else:
                    st.warning("Nie znaleziono miejsca – podaj współrzędne ręcznie.")
            except Exception as e:
                st.warning(f"Geokoder nie działa: {e}")

    st.number_input("Szerokość (°)", key="lat", format="%.6f")
    st.number_input("Długość (°)", key="lon", format="%.6f")

    the_date = st.date_input("Dzień roku", value=date.today())
    window_days = st.slider("Okno ± dni wokół daty", min_value=0, max_value=30, value=7)

    st.markdown("---")
    st.subheader("Progi zjawisk (zmienialne)")
    th_hot  = st.number_input("Very hot: T2M_MAX > [°C]", value=32)
    th_cold = st.number_input("Very cold: T2M_MIN < [°C]", value=0)
    th_wind = st.number_input("Very windy: WS10M ≥ [m/s]", value=10)
    th_rain = st.number_input("Very wet: PRECTOTCORR ≥ [mm/d]", value=10)
    th_hi   = st.number_input("Very uncomfortable (Heat Index) ≥ [°C]", value=32)

    st.markdown("---")
    year_now = datetime.now().year
    start_year = st.number_input("Rok początkowy", value=1995, min_value=1981, max_value=year_now)
    end_year   = st.number_input("Rok końcowy", value=max(1995, year_now-1),
                                 min_value=start_year, max_value=year_now)

run = st.button("Policz prawdopodobieństwa")

# ----------------- Logic -----------------

if run:
    try:
        with st.spinner("Pobieram NASA POWER i liczę statystyki…"):
            df, api_url = pull_power(st.session_state.lat, st.session_state.lon, int(start_year), int(end_year))
            sub = filter_by_doy_window(df, the_date, int(window_days))

            st.caption("Wywołany endpoint NASA POWER:")
            st.code(api_url, language="text")

            if sub.empty:
                st.error("Brak danych w wybranym oknie.")
                st.stop()

            sub["HI"] = compute_heat_index_c(sub["T2M"], sub["RH2M"])
            n = len(sub)

            probs = {
                "Very hot": (sub["T2M_MAX"] > th_hot).mean()*100,
                "Very cold": (sub["T2M_MIN"] < th_cold).mean()*100,
                "Very windy": (sub["WS10M"] >= th_wind).mean()*100,
                "Very wet": (sub["PRECTOTCORR"] >= th_rain).mean()*100,
                "Very uncomfortable": (sub["HI"] >= th_hi).mean()*100,
            }

            # --- TABELA + WYKRES SŁUPKOWY (matplotlib) ---
            out = pd.DataFrame({
                "Condition": list(probs.keys()),
                "Probability [%]": list(probs.values())
            })
            out["Probability [%]"] = pd.to_numeric(out["Probability [%]"], errors="coerce").fillna(0.0).round(1)

            st.subheader("Szanse wystąpienia (historyczne)")
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

            # --- TREND ROCZNY "VERY WET" (matplotlib) ---
            sub["year"] = sub["date"].dt.year
            vw_by_year = (
                sub.groupby("year")["PRECTOTCORR"]
                   .apply(lambda s: (s >= th_rain).mean() * 100.0)
                   .reset_index()
                   .rename(columns={"PRECTOTCORR": "Very wet [%]"})
            )
            vw_by_year["Very wet [%]"] = pd.to_numeric(vw_by_year["Very wet [%]"], errors="coerce").fillna(0.0)

            st.caption("Trend roczny: odsetek dni ‘Very wet’ w oknie ±N dni (w %)")

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

            # --- Eksport z metadanymi ---
            sub_export = sub.copy()
            sub_export["SOURCE"] = "NASA POWER"
            sub_export["API_URL"] = api_url
            sub_export["LAT"] = st.session_state.lat
            sub_export["LON"] = st.session_state.lon
            sub_export["YEARS"] = f"{start_year}-{end_year}"

            csv = sub_export.to_csv(index=False).encode("utf-8")
            st.download_button("Pobierz CSV (podzbiór + metadane)", data=csv,
                               file_name="nasa_power_subset.csv", mime="text/csv")

            st.success(f"Liczba dni w próbie: {n} (lata {start_year}–{end_year}).")

            if any(v == 0 for v in probs.values()):
                st.info("Widzisz 0%? Zwiększ okno ± dni, poluzuj progi lub poszerz zakres lat.")

    except Exception as e:
        st.error(f"Ups – coś poszło nie tak: {e}")

# ----------------- Footer -----------------
st.caption("Uwaga: to statystyka historyczna z siatki NASA (nie prognoza). Wiatr to średnia dobowa, nie porywy.")
