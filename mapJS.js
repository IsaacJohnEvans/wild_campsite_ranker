


var map = L.map("map", {
    center: [53.098275, -1.813959],
    zoom: 15,
    maxZoom: 15});

$.getJSON('https://geoportal1-ons.opendata.arcgis.com/datasets/b216b4c8a4e74f6fb692a1785255d777_0.geojson?outSR={%22latestWkid%22:27700,%22wkid%22:27700}').then(function(geoJSON) {
  var osm = new L.TileLayer.BoundaryCanvas("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    boundary: geoJSON,
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors, UK shape <a href="https://github.com/johan/world.geo.json">johan/word.geo.json</a>'
  });
  map.addLayer(osm);
  var ukLayer = L.geoJSON(geoJSON);
  map.fitBounds(ukLayer.getBounds());
});