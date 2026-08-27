// The committed station-facts asset (viz/public/station_facts.json), generated
// by scripts/gen_station_facts.py from Wikidata + data.ny.gov. Keep these types
// in lockstep with that script's assembly step — it is the producer.
//
// Reference facts about a stop (opening year, a Commons photo, landmark
// designation, public-art credits, annual ridership rank), joined at build time
// and carried as a static asset. No live status and no generation timestamp: it
// changes only when the upstream data does, so it's fetched once per page load,
// cached, and never polled — the same contract as diagram.json.

/** A Commons photo of the station, carried with its license + attribution. */
export interface StationPhoto {
  title: string;
  source: string; // Commons file description page
  image_url: string; // stable Special:FilePath URL (full image)
  thumb_url: string | null;
  artist: string | null;
  license: string | null;
  license_url: string | null;
  credit: string | null;
  attribution_required: boolean;
}

/** One permanent-art piece from the MTA catalog. No image exists upstream, only
 * the credits and a link to the MTA collection page. */
export interface StationArt {
  artist: string | null;
  title: string | null;
  year: string | null;
  material: string | null;
  link: string | null;
}

/** Annual ridership standing, keyed by complex, so every stop of a complex
 * shares it. `rank` is 1-based across the `of` complexes that cover the full
 * twelve-month window; a complex short of the window carries no ridership. */
export interface StationRidership {
  rank: number;
  of: number;
  annual_entries: number;
  complex_id: string | null;
}

/** How this stop was tied to its Wikidata item, and how much to trust it.
 * `transitland_onestop` is the authoritative Onestop-ID → GTFS stop_id join;
 * `onestop_geohash` and `coordinate` are spatial fallbacks (see the generator).
 * `osm_agrees` is an independent OpenStreetMap-node cross-check, null when the
 * item names no node or the lookup was skipped/failed. */
export interface StationMatch {
  method: "transitland_onestop" | "onestop_geohash" | "coordinate";
  authoritative: boolean;
  confidence: number;
  distance_m: number | null;
  onestop_id: string | null;
  osm_node: string | null;
  osm_agrees: boolean | null;
  osm_distance_m: number | null;
}

/** Facts for one stop. Every field is optional: a stop may have ridership but no
 * Wikidata match, art but no photo, and so on. */
export interface StationFactsEntry {
  wikidata_qid?: string;
  wikidata_name?: string | null;
  opened_year?: number | null;
  opened_date?: string | null;
  heritage?: string[];
  photo?: StationPhoto | null;
  art?: StationArt[];
  ridership?: StationRidership;
  match?: StationMatch;
}

export interface StationFacts {
  sources: Record<string, string | null>;
  ridership_window: { start: string; end: string };
  counts: Record<string, number | Record<string, number>>;
  stations: Record<string, StationFactsEntry>;
}

// Module-scoped singleton: the asset is fetched and parsed exactly once per load
// and shared. Failures are not cached, so a transient error doesn't wedge the
// hook for the rest of the session.
let factsPromise: Promise<StationFacts> | null = null;

export function fetchStationFacts(): Promise<StationFacts> {
  if (!factsPromise) {
    // Default cache semantics (not force-cache): a browser holding an older copy
    // must be able to revalidate against a regenerated asset. The module-scoped
    // promise is what keeps it to one fetch per load.
    factsPromise = fetch("/station_facts.json")
      .then((res) => {
        if (!res.ok) throw new Error(`station facts fetch failed: ${res.status}`);
        return res.json() as Promise<StationFacts>;
      })
      .then((doc) => {
        if (doc.stations === undefined) {
          throw new Error(
            "station_facts.json is missing `stations` — the cached copy predates " +
              "the current shape. Hard-reload to pick up the new one.",
          );
        }
        return doc;
      })
      .catch((e: unknown) => {
        factsPromise = null;
        throw e;
      });
  }
  return factsPromise;
}
