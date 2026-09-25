// Стиль подложки карты: только дороги, вода, парки и подписи. Без зданий и POI.
// Данные — OpenFreeMap (свежий OpenStreetMap, обновляется еженедельно, ключ не нужен).
window.App = window.App || {};

App.basemapStyle = function () {
  const C = {
    bg: "#0d131f",
    water: "#0f2135",
    park: "#122219",
    minor: "#1c2536",
    street: "#232e42",
    major: "#2c384f",
    motorway: "#36435c",
    rail: "#1f2839",
    label: "#6f7c93",
    labelHalo: "#0d131f",
    place: "#8391a7",
  };
  // Толщина дороги в зависимости от масштаба
  const w = (z10, z16) => ["interpolate", ["exponential", 1.6], ["zoom"], 10, z10, 16, z16];
  const name = ["coalesce", ["get", "name:ru"], ["get", "name"]];

  return {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {
      omt: { type: "vector", url: "https://tiles.openfreemap.org/planet" },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": C.bg } },
      { id: "park", type: "fill", source: "omt", "source-layer": "park", paint: { "fill-color": C.park, "fill-opacity": 0.8 } },
      {
        id: "wood", type: "fill", source: "omt", "source-layer": "landcover",
        filter: ["in", ["get", "class"], ["literal", ["wood", "grass"]]],
        paint: { "fill-color": C.park, "fill-opacity": 0.5 },
      },
      { id: "water", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-color": C.water } },
      {
        id: "waterway", type: "line", source: "omt", "source-layer": "waterway",
        paint: { "line-color": C.water, "line-width": w(1, 4) },
      },
      {
        id: "rail", type: "line", source: "omt", "source-layer": "transportation", minzoom: 11,
        filter: ["in", ["get", "class"], ["literal", ["rail", "transit"]]],
        paint: { "line-color": C.rail, "line-width": w(0.6, 2), "line-dasharray": [3, 3] },
      },
      {
        id: "road-minor", type: "line", source: "omt", "source-layer": "transportation", minzoom: 12,
        filter: ["in", ["get", "class"], ["literal", ["minor", "service"]]],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": C.minor, "line-width": w(0.3, 4) },
      },
      {
        id: "road-street", type: "line", source: "omt", "source-layer": "transportation",
        filter: ["in", ["get", "class"], ["literal", ["secondary", "tertiary"]]],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": C.street, "line-width": w(0.8, 8) },
      },
      {
        id: "road-major", type: "line", source: "omt", "source-layer": "transportation",
        filter: ["in", ["get", "class"], ["literal", ["primary", "trunk"]]],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": C.major, "line-width": w(1.2, 11) },
      },
      {
        id: "road-motorway", type: "line", source: "omt", "source-layer": "transportation",
        filter: ["==", ["get", "class"], "motorway"],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": C.motorway, "line-width": w(1.5, 12) },
      },
      {
        id: "road-names", type: "symbol", source: "omt", "source-layer": "transportation_name", minzoom: 13,
        layout: {
          "symbol-placement": "line",
          "text-field": name,
          "text-font": ["Noto Sans Regular"],
          "text-size": ["interpolate", ["linear"], ["zoom"], 13, 10, 17, 13],
        },
        paint: { "text-color": C.label, "text-halo-color": C.labelHalo, "text-halo-width": 1.4 },
      },
      {
        id: "water-names", type: "symbol", source: "omt", "source-layer": "water_name", minzoom: 11,
        layout: { "text-field": name, "text-font": ["Noto Sans Italic"], "text-size": 12 },
        paint: { "text-color": "#3f5f82", "text-halo-color": C.labelHalo, "text-halo-width": 1 },
      },
      {
        id: "districts", type: "symbol", source: "omt", "source-layer": "place", minzoom: 11, maxzoom: 15,
        filter: ["in", ["get", "class"], ["literal", ["suburb", "quarter", "neighbourhood"]]],
        layout: {
          "text-field": name,
          "text-font": ["Noto Sans Regular"],
          "text-size": 11,
          "text-transform": "uppercase",
          "text-letter-spacing": 0.08,
        },
        paint: { "text-color": C.place, "text-opacity": 0.55, "text-halo-color": C.labelHalo, "text-halo-width": 1 },
      },
    ],
  };
};
