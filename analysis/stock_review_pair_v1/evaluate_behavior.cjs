const fs=require('fs'),path=require('path');
const B=require('../stock_behavior_description_v3/core.js');
const out=process.argv[2],input=JSON.parse(fs.readFileSync(path.join(out,'results/behavior_input.json')));
const end=input.prices.as_of,start=B.shifts(end,-59);
const records=input.records.map(r=>B.describe(input.prices,r.code,start,end));
fs.writeFileSync(path.join(out,'results/behavior.json'),JSON.stringify({version:B.VERSION,data_run_id:input.data_run_id,run_id:input.run_id,start,end,records}));
