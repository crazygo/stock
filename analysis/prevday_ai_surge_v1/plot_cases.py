"""Export inspectable scientific plots of discovery cases and frozen-rule transfer."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    events=pd.read_csv(HERE/'discovery_events.csv')
    events=events[events.close8.eq(1)].sort_values(['date','close_return'],ascending=[True,False])
    n=len(events)
    if n:
        fig,axes=plt.subplots(math.ceil(n/3),3,figsize=(15,3.5*math.ceil(n/3)),squeeze=False)
        for ax,(_,r) in zip(axes.flat,events.iterrows()):
            d=pd.read_parquet(HERE/f'data_cache/us_5m/{r.symbol}/2026.parquet')
            d=d[d.session_date.eq(r.prev_date)&d.session_type.eq('regular')].sort_values('start_at')
            x=np.arange(len(d))*5+5;ret=(d.close.to_numpy()/d.open.iloc[0]-1)*100
            ax.axhline(0,color='#aaa',lw=.7)
            ax.plot(x,ret,color='#146b67',lw=1.8)
            ax.fill_between(x,(d.low/d.open.iloc[0]-1)*100,(d.high/d.open.iloc[0]-1)*100,color='#146b67',alpha=.12)
            ax2=ax.twinx();ax2.bar(x,d.volume,color='#9dabae',alpha=.25,width=4);ax2.set_ylim(0,d.volume.max()*5);ax2.set_yticks([])
            ax.set_xlim(0,390);ax.set_xticks([0,90,210,330,390],['09:30','11:00','13:00','15:00','16:00'])
            ax.set_ylabel('T-1 return from open (%)')
            ax.set_title(f'{r.symbol} | T={r.date}: {r.close_return:+.1%}\nT-1={r.prev_date}; volume {r.prev_relvol:.2f}x; close location {r.prev_location:.0%}',fontsize=10)
            ax.grid(axis='y',alpha=.2)
        for ax in list(axes.flat)[n:]:ax.set_visible(False)
        fig.suptitle('Previous-session 5-minute paths of all 6 primary eligible >=8% cases\nGray bars: volume. Corporate-action cases excluded. Every price precedes the target day.',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(HERE/'prevday_cases.png',dpi=165);plt.close(fig)
    frozen=json.loads((HERE/'frozen_rules.json').read_text())
    v=pd.read_csv(HERE/'validation.csv');v=v[v.target.eq('close8')]
    windows=['discovery_5','prior_5','prior_15','prior_40']
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    for ax,r in zip(axes.flat,frozen['rules']):
        a=v[v.rule.eq(r['id'])].set_index('window').reindex(windows)
        ax.bar(np.arange(4),a.rate*100,color=['#c17f36','#146b67','#146b67','#146b67'],alpha=.8)
        ax.plot(np.arange(4),a.baseline*100,'o--',color='#777',label='Eligible baseline')
        for i,row in enumerate(a.itertuples()):
            if np.isfinite(row.rate):ax.text(i,row.rate*100+.8,f'{row.k}/{row.n}',ha='center',fontsize=10)
            else:ax.text(i,.8,'No triggers',ha='center',fontsize=9,color='#777')
        ax.set_ylim(0,32)
        ax.set_xticks(range(4),['Recent 5\nDiscovery','Earlier 5','Earlier 15','Earlier 40'])
        ax.set_title(r['id']+' '+r['english_name'],fontsize=11);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Next-day close >=8% frequency (%)')
    fig.suptitle('Frozen rules: recent discovery vs. earlier, disjoint dates\nNumerator / triggered stock-days. Reverse-time transfer is not prospective validation.',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(HERE/'rule_transfer.png',dpi=165);plt.close(fig)


if __name__=='__main__':main()
