# 📖 Public (HDB) Data Dictionary

Schema and column definitions for the datasets the public HDB pipeline reads and
writes. The private counterpart is
[`pipeline_private/DATA_DICTIONARY.md`](../pipeline_private/DATA_DICTIONARY.md).

Everything here is downloaded by `01_fetch_data.py` or derived by a later stage;
nothing needs to be placed by hand.

---

## 1. Source Dataset: `resale_flat_prices_2017_present.csv`

Location: `pipeline_public/data/resale_flat_prices_2017_present.csv`
Source: Data.gov.sg dataset [`d_8b84c4ee58e3cfc0ece0d773c8ca6abc`](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) — Housing & Development Board.
Coverage: 239,977 transactions, January 2017 to September 2026. Not committed (~23 MB); fetched on demand.

This is the training target. Every one of the 11 columns is either a model
feature or the label.

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `month` | `string` | `2024-03` | Transaction month, `YYYY-MM`. Split into `sale_year` and `sale_month` features. |
| `town` | `string` | `ANG MO KIO` | HDB town, one of 26. One-hot encoded into 26 features. |
| `flat_type` | `string` | `4 ROOM` | Flat size class: `1 ROOM` … `5 ROOM`, `EXECUTIVE`, `MULTI-GENERATION`. One-hot encoded. |
| `block` | `string` | `406` | Block number. Not a model feature — used to join MRT distance and to key the map. |
| `street_name` | `string` | `ANG MO KIO AVE 10` | Street. Joined with `block` to identify a building. |
| `storey_range` | `string` | `10 TO 12` | Floor band in 3-storey increments. Parsed to `storey_avg`, the band midpoint. |
| `floor_area_sqm` | `float` | `92.0` | Floor area in square metres. Used directly, and to derive unit price. |
| `flat_model` | `string` | `Improved` | HDB design model, 21 values (`Model A`, `Maisonette`, `DBSS`, …). One-hot encoded. |
| `lease_commence_date` | `int` | `1979` | Year the 99-year lease began. Combined with sale year to give `remaining_lease_years`. |
| `remaining_lease` | `string` | `54 years 03 months` | HDB's own text form. **Not used** — the numeric feature is derived from `lease_commence_date` instead, which avoids parsing this. |
| `resale_price` | `float` | `532000.0` | Transacted price in SGD. **This is the prediction target.** |

---

## 2. Building Metadata: `hdb_property_info.csv`

Location: `pipeline_public/data/hdb_property_info.csv`
Source: Data.gov.sg dataset [`d_17f5382f26140b1fdae0ba2ef6239d2f`](https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view) — HDB. 13,357 blocks.

Published with **24 columns and no location at all**. `02c_enrich_property_info.py`
appends 10 more (marked *derived* below) by joining building footprints and
resolving postal codes through OneMap, which is what lets the map plot one pin
per block.

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `blk_no` | `string` | `406` | Block number. |
| `street` | `string` | `ANG MO KIO AVE 10` | Street name, in HDB's abbreviated form. |
| `max_floor_lvl` | `int` | `12` | Topmost storey in the block. |
| `year_completed` | `int` | `1979` | Year construction completed. |
| `residential` | `string` | `Y` | Whether the block has residential units. |
| `commercial` | `string` | `N` | Whether it has commercial units. |
| `market_hawker` | `string` | `N` | Whether it houses a market or hawker centre. |
| `miscellaneous` | `string` | `N` | Other non-residential use. |
| `multistorey_carpark` | `string` | `N` | Whether it has a multi-storey carpark. |
| `precinct_pavilion` | `string` | `N` | Whether it has a precinct pavilion. |
| `bldg_contract_town` | `string` | `AMK` | HDB's internal town code (`AMK`, `BD`, `TAP`, …), not the full town name. |
| `total_dwelling_units` | `int` | `142` | Total residential units in the block. |
| `1room_sold` … `studio_apartment_sold` | `int` | `98` | Units sold, by flat type. Eight columns. |
| `1room_rental` … `other_room_rental` | `int` | `0` | Rental units, by flat type. Four columns. |
| `postal` | `string` | `560406` | *Derived.* 6-digit postal code, from the building footprint. |
| `full_address` | `string` | `406 ANG MO KIO AVE 10 SINGAPORE 560406` | *Derived.* |
| `lon` / `lat` | `float` | `103.84358` / `1.36694` | *Derived.* WGS84 (EPSG:4326), from OneMap where available, otherwise the footprint centroid. |
| `x` / `y` | `float` | `29537.7` / `39012.4` | *Derived.* SVY21 (EPSG:3414), Singapore's national grid. |
| `geom_4326` / `geom_3414` | `string` | `POINT (103.84358 1.36694)` | *Derived.* WKT forms of the two coordinate pairs. |
| `updated_at` | `date` | `2026-09-09` | *Derived.* When enrichment last ran. |
| `canonical_street` | `string` | `ANG MO KIO AVE 10` | *Derived.* Street normalised to HDB abbreviations so it joins against the resale CSV. |

---

## 3. Derived Lookup: `hdb_postal_data.json`

Location: `pipeline_public/data/hdb_postal_data.json` — **committed**, 9,818 entries.
Built by: `02_build_postal_data.py`.

Read by the browser so typing a postal code fills in town and lease year.

```json
{ "050004": [ { "town": "CENTRAL AREA", "lease_commence_date": "1974" } ] }
```

| Field | Type | Description |
| :--- | :--- | :--- |
| *(key)* | `string` | 6-digit postal code. |
| `town` | `string` | HDB town, matching the resale dataset's spelling. |
| `lease_commence_date` | `string` | Year the lease began, as text. |

The value is an array because a postal code can in principle cover more than one
record; the app reads the first entry.

---

## 4. Derived Lookup: `hdb_mrt_proximity.json`

Location: `pipeline_public/data/hdb_mrt_proximity.json` — **committed**, ~0.4 MB.
Built by: `02b_build_mrt_proximity.py` from `LTAMRTStationExitGEOJSON.geojson` and `HDBExistingBuilding.geojson`, both downloaded into `pipeline_public/data/` by the fetch stage and read only by this pipeline.

Supplies the `mrt_distance_km` and `mrt_walkable` model features, and the app's
per-postal lookup.

| Field | Type | Description |
| :--- | :--- | :--- |
| `unit` | `string` | Always `km`. |
| `description` | `string` | What the distances measure. |
| `median_km` | `float` | Island-wide median, used as the fallback when a block cannot be resolved. |
| `distances` | `object` | `"BLOCK\|TOWN"` → km. 9,222 entries. What the trainer joins on. |
| `by_postal` | `object` | 6-digit postal → km. 12,847 entries. What the browser looks up. |

Distance is straight-line from the block centroid to the nearest of 563 station
exits. See [docs/methodology.md](../docs/methodology.md) for the known limitation
on block-number collisions.

---

## 5. Resolver Cache: `.onemap_postal_cache.json`

Location: `pipeline_public/data/.onemap_postal_cache.json` — **committed**, ~1.3 MB, 12,847 entries.
Built by: `02c_enrich_property_info.py`.

**Public pipeline only.** Every entry is an HDB block, because the postal codes come from `HDBExistingBuilding.geojson`. There is zero overlap with the private non-landed postal lookup, and the private pipeline neither reads nor writes this file — it resolves projects to postal codes from the URA dwelling-units layer instead, which already carries them.

```json
{ "050004": { "blk": "4", "street": "SAGO LANE",
              "lon": 103.8429081594949, "lat": 1.281800622422574 } }
```

| Field | Type | Description |
| :--- | :--- | :--- |
| *(key)* | `string` | 6-digit postal code. |
| `blk` | `string` | Block number, as OneMap records it. |
| `street` | `string` | Authoritative street name. This is the field that disambiguates blocks — block 32 exists in Chinatown and in Eunos. |
| `lon` / `lat` | `float` | WGS84 address point, preferred over the footprint centroid. |

Committed because rebuilding it means roughly 12,800 rate-limited OneMap requests over about four hours. It holds only public address data and no credentials; the endpoint needs no API key.

---

## 6. Market Index: `models/hdb_price_index.json`

Location: `models/hdb_price_index.json` — **committed**.
Source: Data.gov.sg dataset [`d_14f63e595975691e7c24a27ae4c07c79`](https://data.gov.sg/datasets/d_14f63e595975691e7c24a27ae4c07c79/view) — HDB Resale Price Index (1Q2009 = 100), quarterly from 1990-Q1.

```json
{ "2024-Q3": 192.9, "2024-Q4": 196.2 }
```

Used only to derive the annual growth rate applied to predictions beyond the
training cutoff. The model itself never sees it — gradient-boosted trees cannot
extrapolate, so time is handled outside the model.
