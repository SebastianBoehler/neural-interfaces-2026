"""Source-only BIDS information inventory; no EEG or excluded-person event reads."""
import collections
import csv
import hashlib
import json
from pathlib import Path

FIT=('P04','P05','P06','P07','P08','P10')
ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'data/neural_compet/Dreyer2026Proteus/download/nm000290'


def main():
    files=[];tasks=collections.Counter();events=collections.Counter();columns=set();rates=set();counts=set()
    seconds=collections.Counter();full_task=0.;official_task=0.
    for person in FIT:
        for path in sorted((RAW/f'sub-{person}').glob('ses-*/eeg/*_eeg.json')):
            meta=json.loads(path.read_text());task=meta['TaskName'];tasks[task]+=1
            rates.add(meta['SamplingFrequency']);counts.add(tuple(meta.get(k,0) for k in ('EEGChannelCount','EOGChannelCount','EMGChannelCount')))
            seconds[task]+=float(meta['RecordingDuration'])
            event=path.with_name(path.name.replace('_eeg.json','_events.tsv'))
            with event.open() as f:
                reader=csv.DictReader(f,delimiter='\t');columns.update(reader.fieldnames)
                for row in reader:
                    events[row['trial_type']]+=1
                    if row['trial_type'] in ('mi','sub','word'):
                        duration=float(row['duration']);full_task+=duration;official_task+=min(4.,max(0.,duration-1.))
            files.append(dict(path=str(path.relative_to(RAW)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                events_sha256=hashlib.sha256(event.read_bytes()).hexdigest()))
    result=dict(source_people=FIT,excluded_event_reads=['P01','P02','P03','P09'],EEG_loaded=False,
        n_source_recordings=len(files),sampling_rates=sorted(rates),EEG_EOG_EMG_counts=sorted(counts),
        task_recordings=dict(tasks),task_recording_seconds=dict(seconds),event_columns=sorted(columns),
        source_event_types=dict(events),cued_task_seconds=full_task,official_task_seconds=official_task,
        nominal_other_cued_task_seconds=full_task-official_task,
        limitation='duration sums overlap trials only as reported; not exported/certified independent crops; metadata not model inputs',files=files)
    out=ROOT/'evidence/next_wave_16nov/direct346_inventory.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':main()
