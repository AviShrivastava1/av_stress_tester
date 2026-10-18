import file from '../../waymo_derived_outputs.json';

/**
 * The one switch for everything that exports Waymo-derived imagery or geometry (an image, an SVG, a GIF or a video of a scene).
 *
 * The Waymo licence reply is outstanding (email of 2026-10-09 to open-dataset@waymo.com), so the exporters are built, tested and left OFF.
 * The value comes from waymo_derived_outputs.json at the repository root, {"enabled": false}; src/waymo_derived.py is the other reader of that
 * file, and there is no second constant. Only the boolean `true` turns it on: anything else in the file reads as off here (the Python reader
 * is stricter and raises).
 *
 * With it off, no export control renders anywhere (scene/export/SceneExportControls), and the module that produces the files is never
 * loaded (ScenePanel imports it lazily, behind this value). Every exporter also takes the value as an argument and refuses when it is false,
 * so tests exercise them with it on without editing the file.
 */
export const WAYMO_DERIVED_OUTPUTS_ENABLED: boolean = (file as { enabled?: unknown }).enabled === true;
