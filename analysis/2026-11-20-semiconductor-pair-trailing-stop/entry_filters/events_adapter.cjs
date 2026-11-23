// Input rows must already be completed, positive-volume five-minute bars.
// Output index i identifies a confirmation at bar i's END, never its backfilled start.
'use strict';

const fs = require('node:fs');
const events = require('../../2026-11-17-candle-center-area-inspector/events.js');
const { rapidWindow } = require('../../2026-11-17-candle-center-area-inspector/timeline_shape.js');

function chartConfirmations({ times, closes }) {
  const milliseconds = times.map((seconds, i) => {
    if (!Number.isSafeInteger(seconds) || seconds % 300 !== 0 || (i && seconds <= times[i - 1])) {
      throw new Error('Expected ordered, distinct five-minute bar ends in Unix seconds.');
    }
    return seconds * 1000;
  });
  const settings = { ...events.defaults(), direction: 'up' };
  const flags = {};
  for (const symbol of ['SOXL', 'SOXS']) {
    const prices = closes[symbol];
    const logClose = Float64Array.from(prices, price => 100 * Math.log(price));
    const visible = new Array(times.length).fill(0);
    // CandleEvents.run resets its window on each missing slot or scheduled closure.
    for (const event of events.run(prices, milliseconds, [settings]).events) {
      if (!rapidWindow(logClose.subarray(event.a, event.i + 1))) visible[event.i] = 1;
    }
    flags[symbol] = visible;
  }
  return flags;
}

if (require.main === module) {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  process.stdout.write(JSON.stringify(chartConfirmations(input)));
}

module.exports = { chartConfirmations };
