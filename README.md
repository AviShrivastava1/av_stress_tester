# av_stress_tester

Finds the minimum perturbation to a real Waymo Open Motion Dataset scenario that
causes a collision — a search for how close ordinary driving already sits to
catastrophe.

## API

A read-only HTTP layer over the scored results. Nothing here starts a
computation: the optimizer runs offline and this reads what it wrote.

```bash
PGDATABASE=av_stress PGUSER=$(whoami) uvicorn src.api.main:app --port 8000
```

Interactive docs at `/docs`.

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness, including a real database round-trip and the PostGIS version |
| `GET /stats` | Corpus counts — total, stress-tested, collisions found, robustly safe |
| `GET /scenarios` | Ranked by fragility, most fragile first. Keyset-paginated via `limit`, `cursor`, `stress_tested_only` |
| `GET /scenarios/{id}` | One scenario, including its raw 4-D perturbation vector |
| `GET /scenarios/{id}/trajectories` | Every agent's logged path, with per-vertex timesteps and headings |
| `GET /scenarios/{id}/perturbed` | The challenger's logged path beside its minimally-perturbed one |

**The API needs no Waymo package and no `.tfrecord` shard.** It reads only
PostgreSQL/PostGIS, so it starts anywhere — trajectory geometry is exported ahead
of time by `src/scoring/export_geometry.py`, which is the only module that touches
a shard. That split is deliberate: the Waymo dependency is Colab-only, and an API
that could not start without it would not be deployable.

**Coordinates are local planar metres, not longitude/latitude.** WOMD scenarios use
a local metric frame. Treating these values as lon/lat puts every scenario off the
coast of West Africa and makes every distance meaningless.
