Weather Odds from NASA POWER
An interactive Streamlit app that shows historical odds of selected weather conditions for a given location and day-of-year. It uses NASA’s POWER daily datasets to estimate the probability of:
Hot, Cold, Windy, Wet, and Uncomfortable (Heat Index) days.
⚠️ This is not a forecast. It summarizes historical climatology around the chosen date.

✨ Features
🗺️ Geocoding with country bias & candidate picker (Nominatim)
Enter “City, Country” or use ISO-2 code (e.g., pl, pk) to get accurate coordinates.
🔁 Custom day-of-year window (±N days) to increase sample size.
🎛 Adjustable thresholds for hot/cold/windy/wet/heat-index.
🌡 Temperature unit toggle (°C / °F) — UI values convert automatically.
📈 Clean charts
Bar chart of probabilities (with optional labels)
Yearly trend chart with optional rolling average and metric selector
⬇️ CSV export (subset + metadata: lat/lon, years, API URL, chosen units).
⚡ Caching of NASA POWER calls for snappy interactions.

🚀 Live Demo
Once deployed on Streamlit Community Cloud, your public link will look like:
https://weather-odds-nasagit-d4pjuutu7zkum9gjeuqq5i.streamlit.app/

🧑‍🏫 How to Use
Enter a place (e.g., Warsaw, Poland or Turbat, Pakistan).
(Optional) Set Country bias (ISO-2, e.g., pl, pk) and click Geocode.
Pick the correct candidate from the list and Use selected
Adjust date, ±day window, thresholds, and units (°C/°F).
Pick a trend metric (Wet/Hot/Cold/Windy/Uncomfort.) and (optionally) enable rolling average.
Click Compute probabilities.
Review the bar chart and trend chart.
Click Download CSV for the subset + metadata.

📊 What’s Under the Hood
Data source: NASA POWER API (daily point time series)
Endpoint pattern:
https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR&start=<YYYY>0101&end=<YYYY>1231&latitude=<lat>&longitude=<lon>&community=RE&format=JSON
Variables used
T2M, T2M_MAX, T2M_MIN (°C)
RH2M (%)
WS10M (m/s)
PRECTOTCORR (mm/day)
Heat Index: computed from T (°C) & RH(%) via a standard approximation (internally converted to/from °F as needed).
Windowing: select historical days within ±N days of the target date (day-of-year), across chosen years, with leap-day handling.
Probabilities: share (%) of days meeting each threshold within the selected sample.
