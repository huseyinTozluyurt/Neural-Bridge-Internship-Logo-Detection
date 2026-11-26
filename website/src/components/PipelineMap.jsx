const STAGES = [
  { id: 'datasets', n: '01', name: 'Acquire', tool: 'download_dataset.py', note: 'FlickrLogos-32 · OpenLogo · Roboflow exports' },
  { id: 'convert', n: '02', name: 'Convert', tool: 'convert_*.py', note: 'VOC / COCO → YOLO txt labels' },
  { id: 'unify', n: '03', name: 'Unify', tool: 'unify_logo_datasets.py', note: 'One class list, leakage-aware splits' },
  { id: 'audit', n: '04', name: 'Audit', tool: 'audit_dataset.py', note: 'Label correctness, balance, duplicates' },
  { id: 'config', n: '05', name: 'Configure', tool: 'data.yaml · hyp · cfg', note: 'Classes, paths, hyper-parameters' },
  { id: 'training', n: '06', name: 'Train', tool: 'train_yolo.py', note: 'YOLOv7 transfer learning, guarded' },
  { id: 'evaluation', n: '07', name: 'Evaluate', tool: 'evaluate_yolo.py', note: 'mAP, precision, recall, F1, errors' },
  { id: 'gui', n: '08', name: 'Deploy', tool: 'video_detection_gui.py', note: 'PyQt5 image and video detection' },
];

export default function PipelineMap() {
  return (
    <ol className="pipeline" aria-label="Pipeline stages">
      {STAGES.map((s) => (
        <li key={s.id}>
          <a href={`#${s.id}`}>
            <span className="stage-n">{s.n}</span>
            <span className="stage-name">{s.name}</span>
            <code>{s.tool}</code>
            <span className="stage-note">{s.note}</span>
          </a>
        </li>
      ))}
    </ol>
  );
}
