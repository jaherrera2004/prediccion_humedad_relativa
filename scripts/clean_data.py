import pandas as pd

SEP = "=" * 60

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def report_nulls(df, label):
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print(f"  [{label}] Sin nulos.")
    else:
        for col, n in nulls.items():
            pct = n / len(df) * 100
            print(f"  [{label}] {col}: {n} nulos ({pct:.2f}%)")

def report_bad_values(df, label):
    issues = []

    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        n_neg = (df[col] < 0).sum()
        if col in ("Wind Speed (km/h)", "Humidity", "Visibility (km)") and n_neg > 0:
            issues.append(f"  [{label}] {col}: {n_neg} valores negativos")

    # Humedad fuera de rango [0, 1]
    if "Humidity" in df.columns:
        out = ((df["Humidity"] < 0) | (df["Humidity"] > 1)).sum()
        if out > 0:
            issues.append(f"  [{label}] Humidity: {out} valores fuera de [0, 1]")

    # Presión = 0 (dato corrupto)
    if "Pressure (millibars)" in df.columns:
        zeros = (df["Pressure (millibars)"] == 0).sum()
        if zeros > 0:
            issues.append(f"  [{label}] Pressure (millibars): {zeros} valores = 0 (corruptos)")

    # Loud Cover (siempre 0 — columna vacía de facto)
    if "Loud Cover" in df.columns:
        unique = df["Loud Cover"].nunique()
        issues.append(f"  [{label}] Loud Cover: {unique} valor(es) único(s) — columna sin información")

    if not issues:
        print(f"  [{label}] Sin datos problemáticos detectados.")
    else:
        for msg in issues:
            print(msg)


# ── Carga ────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("data/weatherHistory.csv")

section("ANTES — Estado del dataset crudo")
print(f"  Filas: {len(df_raw):,}  |  Columnas: {df_raw.shape[1]}")
print(f"  Duplicados: {df_raw.duplicated().sum():,}")
print()
report_nulls(df_raw, "ANTES")
print()
report_bad_values(df_raw, "ANTES")

print("\n  Tipos de dato:")
for col, dtype in df_raw.dtypes.items():
    print(f"    {col}: {dtype}")

print("\n  Estadísticas numéricas básicas:")
print(df_raw.describe().to_string())


# ── Limpieza ─────────────────────────────────────────────────────────────────
df = df_raw.copy()

df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
df = df.drop_duplicates()
df["Precip Type"] = df["Precip Type"].fillna("rain")
df = df.drop(columns=["Loud Cover", "Daily Summary", "Summary", "Precip Type"])

df = df.rename(columns={
    "Formatted Date":           "formatted_date",
    "Temperature (C)":          "temperature",
    "Apparent Temperature (C)": "apparent_temperature",
    "Humidity":                 "humidity",
    "Wind Speed (km/h)":        "wind_speed",
    "Wind Bearing (degrees)":   "wind_bearing",
    "Visibility (km)":          "visibility",
    "Pressure (millibars)":     "pressure",
})

numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].round(2)


# ── Verificación post-limpieza ────────────────────────────────────────────────
section("DESPUÉS — Dataset limpio")
print(f"  Filas: {len(df):,}  |  Columnas: {df.shape[1]}")
print(f"  Filas eliminadas (duplicados): {len(df_raw) - len(df):,}")
print()
report_nulls(df, "DESPUÉS")
print()

print("  Columnas restantes:")
for col in df.columns:
    print(f"    {col}")

print("\n  Estadísticas numéricas post-limpieza:")
print(df.describe().to_string())


# ── Guardar ───────────────────────────────────────────────────────────────────
assert df.isnull().sum().sum() == 0, "Quedaron nulos inesperados"

df.to_csv("data/weatherHistory_clean.csv", index=False)

section("RESULTADO")
print(f"  Guardado en data/weatherHistory_clean.csv")
print(f"  Filas finales: {len(df):,}")
print(f"  Decimales: máximo 2 en todas las columnas numéricas")
