// Pure view helpers shared by the page and node tests. No data fetching here.
(function (root) {
  function dayNum(d) { return Math.round(Date.UTC(+d.slice(0, 4), +d.slice(5, 7) - 1, +d.slice(8, 10)) / 86400000); }
  // 14/30/60 = calendar days ending at (and including) the evaluation cutoff. View crop only.
  function cropPoints(points, cutoff, days) {
    var end = dayNum(cutoff), start = end - (days - 1);
    return points.filter(function (p) { var n = dayNum(p.date); return n >= start && n <= end; });
  }
  // Segments only between consecutive saved results; an unknown value breaks the line.
  function segments(points, trait) {
    var segs = [], prev = null;
    points.forEach(function (p) {
      var v = p.traits[trait] ? p.traits[trait].display : null;
      if (v === null || v === undefined) { prev = null; return; }
      if (prev) segs.push({ from: prev, to: { date: p.date, v: v }, gapDays: dayNum(p.date) - dayNum(prev.date) });
      prev = { date: p.date, v: v };
    });
    return segs;
  }
  var api = { dayNum: dayNum, cropPoints: cropPoints, segments: segments };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.ViewLogic = api;
})(this);
