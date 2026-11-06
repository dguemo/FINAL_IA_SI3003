# Plan B: datos crudos versionados

`python -m data.download_siata` fusiona aquí lo descargado de SIATA, del geoportal AMVA y de
Open-Meteo:

- `pm25_hourly.csv.gz`: PM2.5 horario (codigo, fecha, pm25, calidad)
- `stations.csv`: catálogo de estaciones (codigo, nombre, nombre_corto, municipio, latitud, longitud)
- `meteo_daily.csv`: meteorología diaria del valle
- `pm25_daily_recent.csv`: últimos 30 promedios diarios oficiales del geoportal

**Después de la primera descarga exitosa, haga commit de estos tres archivos.** Si una fuente
falla después, el pipeline usará estos archivos (`python pipeline.py --offline`).

## Meteorología a mano (si Open-Meteo responde 429)

Abra estas URLs en el navegador, guarde cada respuesta como .json y ejecute
`python -m data.download_siata --meteo-json historico.json reciente.json`:

- https://archive-api.open-meteo.com/v1/archive?latitude=6.2518&longitude=-75.5636&timezone=America/Bogota&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,boundary_layer_height&start_date=2023-07-31&end_date=2024-07-31
- https://api.open-meteo.com/v1/forecast?latitude=6.2518&longitude=-75.5636&timezone=America/Bogota&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,boundary_layer_height&past_days=92&forecast_days=1
