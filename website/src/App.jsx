import { Fragment, useEffect, useState } from 'react';
import Section from './components/Section.jsx';
import CodeBlock from './components/CodeBlock.jsx';
import Figure, { imageSrc } from './components/Figure.jsx';
import Callout from './components/Callout.jsx';
import FileTag from './components/FileTag.jsx';
import PipelineMap from './components/PipelineMap.jsx';
import IoUPlayground from './components/IoUPlayground.jsx';
import PRCalculator from './components/PRCalculator.jsx';
import LabelChecker from './components/LabelChecker.jsx';
import MetricsTable from './components/MetricsTable.jsx';
import TrainingDynamics from './components/TrainingDynamics.jsx';
import QualitativeGallery from './components/QualitativeGallery.jsx';
import { ref, SAMPLES } from './components/qualitativeData.js';
import Reveal from './components/Reveal.jsx';
import ScrollProgress from './components/ScrollProgress.jsx';
import CountUp from './components/CountUp.jsx';
import useAutoReveal from './components/useAutoReveal.js';

const REPO_URL = 'https://github.com/huseyinTozluyurt/Neural-Bridge-Internship-Logo-Detection';

const NAV = [
  { id: 'overview', label: 'Overview' },
  { id: 'setup', label: 'Setup' },
  { id: 'datasets', label: 'Datasets' },
  { id: 'convert', label: 'Format conversion' },
  { id: 'unify', label: 'Dataset unification' },
  { id: 'audit', label: 'Data audit' },
  { id: 'config', label: 'YAML configuration' },
  { id: 'training', label: 'Training' },
  { id: 'evaluation', label: 'Evaluation' },
  { id: 'metrics', label: 'Understanding metrics' },
  { id: 'inference', label: 'Image & video inference' },
  { id: 'qualitative', label: 'Qualitative results' },
  { id: 'gui', label: 'PyQt5 application' },
  { id: 'beyond', label: 'Beyond detection' },
  { id: 'recipe', label: 'Retraining recipe' },
  { id: 'cite', label: 'Cite' },
];

function useScrollSpy(ids) {
  const [active, setActive] = useState(ids[0]);
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: '-15% 0px -70% 0px' },
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [ids]);
  return active;
}

const KPIS = [
  { label: 'mAP@0.5', value: '94.75%' },
  { label: 'mAP@0.5:0.95', value: '82.04%' },
  { label: 'Precision', value: '85.71%' },
  { label: 'Recall', value: '92.88%' },
  { label: 'F1-score', value: '89.15%' },
];

export default function App() {
  const active = useScrollSpy(NAV.map((n) => n.id));
  const [menuOpen, setMenuOpen] = useState(false);
  useAutoReveal();

  return (
    <>
      <ScrollProgress />
      <header className="topbar">
        <a className="brand" href="#top">
          <span className="brand-mark" aria-hidden="true" />
          Logo Detection Pipeline
        </a>
        <button type="button" className="menu-btn" onClick={() => setMenuOpen((o) => !o)} aria-expanded={menuOpen}>
          Contents
        </button>
        <a className="gh" href={REPO_URL} target="_blank" rel="noreferrer">
          View on GitHub
        </a>
      </header>

      <div className="layout">
        <nav className={menuOpen ? 'sidebar open' : 'sidebar'} aria-label="Sections">
          <ol>
            {NAV.map((n, i) => (
              <li key={n.id}>
                <a href={`#${n.id}`} className={active === n.id ? 'active' : ''} onClick={() => setMenuOpen(false)}>
                  <span>{String(i + 1).padStart(2, '0')}</span>
                  {n.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <main id="top">
          <Hero />
          <Overview />
          <Setup />
          <Datasets />
          <Convert />
          <Unify />
          <Audit />
          <Config />
          <Training />
          <Evaluation />
          <Metrics />
          <Inference />
          <Qualitative />
          <Gui />
          <Beyond />
          <Recipe />
          <Cite />
          <footer className="footer">
            <p>
              Built with React and Vite from the code in{' '}
              <a href={REPO_URL} target="_blank" rel="noreferrer">
                Neural-Bridge-Internship-Logo-Detection
              </a>
              . Detector framework: YOLOv7 by WongKinYiu (vendored and patched in <code>yolov7/</code>).
            </p>
          </footer>
        </main>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */

function Hero() {
  return (
    <section className="hero">
      <div className="hero-text">
        <div className="kicker">Computer vision · YOLOv7 · PyTorch · PyQt5</div>
        <h1 className="hero-title">
          {'Brand logo detection, from raw datasets to a desktop app'.split(' ').map((w, i) => (
            <Fragment key={i}><span style={{ animationDelay: `${120 + i * 70}ms` }}>{w}</span>{' '}</Fragment>
          ))}
        </h1>
        <p className="lead">
          A guided tour of the full pipeline in this repository: converting and unifying FlickrLogos-32 and OpenLogo, auditing labels
          for correctness, configuring and training a YOLOv7 detector, scoring it with precision, recall and mAP, and running it on
          images and videos from a PyQt5 application.
        </p>
        <details className="abstract">
          <summary>Abstract</summary>
          <p>
            We describe an end-to-end pipeline for closed-set brand-logo detection. Heterogeneous public datasets (FlickrLogos-32 and
            OpenLogo) are converted to a common YOLO format and merged into one taxonomy through an explicit alias map. Splits are built
            from perceptual-hash duplicate groups to prevent train/test leakage, and every label is checked by a preflight validator before
            training. A COCO-pretrained YOLOv7 detector fine-tuned on FlickrLogos-32 reaches 94.75% mAP@0.5 and 82.04% mAP@0.5:0.95
            on a held-out test split. Qualitative analysis on unseen advertisement footage shows strong robustness to clutter, colour
            casts and archival degradation. The dominant failure modes are confusion between brands with similar colour schemes and
            loss of the colour cue in monochrome renderings, which motivates the combined-dataset model.
          </p>
        </details>
        <div className="hero-actions">
          <a className="btn" href="#overview">Start the tour</a>
          <a className="btn ghost" href="#qualitative">See qualitative results</a>
        </div>
      </div>
      <div className="hero-visual">
        <img className="hero-img" src={imageSrc('qual_paulaner_beergarden.jpg')} alt="Paulaner emblem detected at 0.97 confidence in a beer-garden advertisement" />
        <svg className="hero-overlay" viewBox="0 0 100 57" preserveAspectRatio="none" aria-hidden="true">
          <rect className="hero-box" x="39.6" y="9.9" width="20.3" height="20.4" pathLength="1" />
        </svg>
        <span className="hero-tag">paulaner 0.97</span>
        <span className="hero-scan" aria-hidden="true" />
      </div>
      <div className="kpis" aria-label="Published test-split results">
        {KPIS.map((k) => (
          <div key={k.label} className="kpi">
            <div className="kpi-value"><CountUp value={k.value} /></div>
            <div className="kpi-label">{k.label}</div>
          </div>
        ))}
        <div className="kpi-note">Current published checkpoint: 32-class FlickrLogos-32 model, 224 held-out test images, 323 logos.</div>
      </div>
    </section>
  );
}

function Overview() {
  return (
    <Section
      id="overview"
      kicker="01 · Overview"
      title="The pipeline at a glance"
      lead="Every stage is a script in scripts/. Each one reads the previous stage's output, never modifies its inputs, and writes a report you can inspect before moving on."
    >
      <PipelineMap />
      <h3>How the two layers relate</h3>
      <p>
        <FileTag path="scripts/">scripts/</FileTag> is the project-specific layer and <FileTag path="yolov7/">yolov7/</FileTag> is the
        detector framework it drives. Training and evaluation never call <code>yolov7/train.py</code> or <code>yolov7/test.py</code>{' '}
        directly. They go through <FileTag path="scripts/train_yolo.py" /> and <FileTag path="scripts/evaluate_yolo.py" />, which add
        dataset validation, path normalisation, class-order checks and reporting on top.
      </p>
      <div className="two-col">
        <div className="card">
          <h4>Data side</h4>
          <ul>
            <li><b>Convert</b> each source to YOLO text labels (one row per box).</li>
            <li><b>Unify</b> class names across sources with an alias map, then split without leakage.</li>
            <li><b>Audit</b> every label row, class balance and duplicate images before a single GPU minute is spent.</li>
          </ul>
        </div>
        <div className="card">
          <h4>Model side</h4>
          <ul>
            <li><b>Train</b> YOLOv7 from COCO-pretrained weights with guards on class count and validation support.</li>
            <li><b>Evaluate</b> the checkpoint on the untouched test split: AP, precision, recall, F1 and error CSVs.</li>
            <li><b>Deploy</b> the checkpoint through the CLI or the PyQt5 desktop viewer.</li>
          </ul>
        </div>
      </div>
      <h3>Repository layout</h3>
      <CodeBlock
        lang="text"
        title="repository tree"
        code={`
.
├── config.py                   Shared project configuration
├── test_setup.py               Verifies CUDA / dependency availability
├── docs/                       Evaluation charts, sample detections, per-class CSV
├── website/                    This documentation site (React + Vite)
├── yolov7/                     Vendored YOLOv7, patched for this project
│   ├── train.py, test.py, detect.py
│   ├── models/                 yolo.py (architecture), common.py, experimental.py
│   ├── cfg/training/yolov7.yaml
│   └── data/hyp.scratch.p5.no_ota.yaml
└── scripts/
    ├── download_dataset.py           Roboflow / FlickrLogos-32 download helpers
    ├── convert_voc_to_yolo.py        OpenLogo (Pascal VOC) → YOLO
    ├── convert_flickrlogos32.py      FlickrLogos-32 → COCO
    ├── convert_coco_to_yolo.py       COCO → YOLO (name-stable class IDs)
    ├── unify_logo_datasets.py        Merge sources into one taxonomy + splits
    ├── logo_class_map_openlogo_flickr.yaml
    ├── dataset_validation.py         Preflight label checks (used by training)
    ├── audit_dataset.py              Class/box stats, duplicates, previews
    ├── prepare_dataset_splits.py     Leakage-aware re-splitting
    ├── train_yolo.py                 Training wrapper
    ├── evaluate_yolo.py              Evaluation wrapper + reports
    ├── detect_yolo.py                Inference wrapper
    ├── video_detection_gui.py        PyQt5 desktop app
    └── requirements-gui.txt`}
      />
    </Section>
  );
}

function Setup() {
  return (
    <Section
      id="setup"
      kicker="02 · Setup"
      title="Environment and dependencies"
      lead="One virtual environment holds PyTorch, the YOLOv7 requirements and the optional GUI dependency."
    >
      <CodeBlock
        code={`
git clone ${REPO_URL}.git
cd Neural-Bridge-Internship-Logo-Detection

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate

pip install -r yolov7/requirements.txt
pip install -r scripts/requirements-gui.txt   # PyQt5, only for the desktop app

# COCO-pretrained YOLOv7 weights: the starting point for transfer learning
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -P yolov7/

# Confirm CUDA and the GPU are visible
python test_setup.py`}
      />
      <Callout title="Hardware this project was tuned on">
        A 4 GB GTX 1650. That is why the recorded commands use <code>--batch-size 4 --img-size 416 --workers 0</code>. With more VRAM,
        raise the batch size first, then the image size.
      </Callout>
      <p>
        Large artefacts stay out of git on purpose: <code>datasets/</code>, <code>runs/</code>, <code>reports/</code> and all{' '}
        <code>*.pt</code> weights are listed in <FileTag path=".gitignore" />. Each command below recreates them locally.
      </p>
    </Section>
  );
}

function Datasets() {
  return (
    <Section
      id="datasets"
      kicker="03 · Datasets"
      title="The two source datasets"
      lead="The new model combines FlickrLogos-32 and OpenLogo. They overlap in brands but disagree on file formats, class naming and split conventions, which is exactly what the next three stages fix."
    >
      <div className="two-col">
        <div className="card">
          <h4>FlickrLogos-32</h4>
          <p>
            32 brands (Adidas, Aldi, Apple, Becks, BMW, Carlsberg, Coca-Cola, DHL, Heineken, Pepsi, Shell, Starbucks, UPS and more)
            photographed in real-world Flickr scenes. Clean, consistent boxes; small per-class support. This is the dataset behind the
            current published checkpoint.
          </p>
          <p className="muted">Format: original annotations → COCO via <FileTag path="scripts/convert_flickrlogos32.py" /> → YOLO.</p>
        </div>
        <div className="card">
          <h4>OpenLogo</h4>
          <p>
            A much broader brand vocabulary in Pascal VOC XML. Many brands are annotated twice: once for the figurative mark and once
            for the word mark (for example <code>adidas</code> and <code>adidas_text</code>, <code>marlboro_fig</code> and{' '}
            <code>marlboro_text</code>).
          </p>
          <p className="muted">Format: Pascal VOC → YOLO via <FileTag path="scripts/convert_voc_to_yolo.py" />.</p>
        </div>
      </div>
      <h3>Downloading from Roboflow</h3>
      <p>
        <FileTag path="scripts/download_dataset.py" /> fetches Roboflow-hosted versions when an API key is set. Export Roboflow versions in{' '}
        <b>YOLOv7 PyTorch</b> format and they can go straight to the unification step; the unifier accepts Roboflow&apos;s{' '}
        <code>valid</code> split name as well as <code>val</code>.
      </p>
      <CodeBlock
        code={`
export ROBOFLOW_API_KEY='your_key'
python scripts/download_dataset.py`}
      />
      <Callout tone="warn" title="Keep the key out of git">
        Store the key in an environment variable or a local <code>.env</code> file. <code>.env</code> is already ignored by git.
      </Callout>
    </Section>
  );
}

function Convert() {
  return (
    <Section
      id="convert"
      kicker="04 · Format conversion"
      title="Everything becomes YOLO text labels"
      lead="YOLOv7 reads one .txt file per image with one row per box. All converters write into a fresh output folder and leave the source untouched."
    >
      <div className="two-col">
        <div>
          <h3>The YOLO label format</h3>
          <CodeBlock lang="text" title="labels/train/img_0001.txt" code={`
# class  x_center  y_center  width   height   (all normalised 0–1)
12       0.512     0.430     0.210   0.118
3        0.250     0.700     0.090   0.060`} />
          <p>
            Coordinates are fractions of the image width and height, measured from the box centre. An empty file means the image is a
            confirmed negative (no logos). Class IDs index into the <code>names</code> list in <code>data.yaml</code>, so an ID only means
            something together with that list.
          </p>
        </div>
        <div>
          <h3>Expected folder structure</h3>
          <CodeBlock lang="text" title="dataset layout" code={`
datasets/<name>/
├── data.yaml
├── train/
│   ├── images/   *.jpg
│   └── labels/   *.txt   (same stem as the image)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/`} />
        </div>
      </div>
      <h3>OpenLogo: Pascal VOC → YOLO</h3>
      <p>
        Every VOC <code>object/name</code> is kept as its own class, so no brand information is lost here; merging subtype names happens
        later and explicitly. The output is staged and only moved into place after the whole conversion succeeds.
      </p>
      <CodeBlock code={`
python scripts/convert_voc_to_yolo.py \\
    --source /path/to/openlogo \\
    --output datasets/openlogo_yolo_multiclass \\
    --seed 42`} />
      <h3>FlickrLogos-32: original → COCO → YOLO</h3>
      <CodeBlock code={`
python scripts/convert_flickrlogos32.py /path/to/FlickrLogos-32 --output-dir datasets/flickrlogos_coco
python scripts/convert_coco_to_yolo.py --input datasets/flickrlogos_coco --output datasets/flickrlogos_yolo`} />
      <Callout title="Why the COCO converter remaps by name">
        COCO exports can number categories differently in train, valid and test. <code>convert_coco_to_yolo.py</code> builds one
        category-name → ID mapping across all splits and rejects malformed or out-of-bounds boxes, so a split-local category ID can
        never silently turn one brand into another.
      </Callout>
    </Section>
  );
}

function Unify() {
  return (
    <Section
      id="unify"
      kicker="05 · Dataset unification"
      title="Merging FlickrLogos-32 and OpenLogo into one taxonomy"
      lead="unify_logo_datasets.py turns several YOLO datasets into one: a single canonical class list, remapped class IDs, and train/val/test splits that keep near-identical images together."
    >
      <h3>Step 1 · Decide what a class is: the alias map</h3>
      <p>
        Names are normalised automatically (case, spaces, punctuation), but the script never guesses that two different words are the
        same brand. Those decisions live in a YAML alias map. For this pair of datasets the repository ships{' '}
        <FileTag path="scripts/logo_class_map_openlogo_flickr.yaml" />, which merges OpenLogo&apos;s <code>*_text</code>,{' '}
        <code>*_fig</code> and numbered variants into the base brand and reconciles spelling differences between the two sources.
      </p>
      <CodeBlock lang="yaml" title="scripts/logo_class_map_openlogo_flickr.yaml (excerpt)" code={`
classes: auto          # union of all source classes after aliasing
aliases:
  adidas:   [adidas1, adidas_text]
  cocacola: [coke]
  guinness: [guiness]          # FlickrLogos-32 spelling
  heineken: [heineken_text]
  marlboro: [marlboro_fig, marlboro_text]
  cvs:      [cvspharmacy]`} />
      <div className="two-col">
        <div className="card">
          <h4><code>classes: auto</code></h4>
          <p>The class list is the union of every source class after aliasing. Good for a first broad model.</p>
        </div>
        <div className="card">
          <h4>Explicit <code>classes: [...]</code></h4>
          <p>
            Train a selected taxonomy. Any label outside it is an error by default. <code>--unmapped-labels exclude-image</code> drops
            the whole image rather than one box, because dropping a single box would leave a real logo unlabelled and teach the model
            that it is background.
          </p>
        </div>
      </div>

      <h3>Step 2 · Group near-duplicates so they cannot leak across splits</h3>
      <p>
        If the same photo (or a resized copy of it) lands in both train and test, the test score measures memory, not generalisation.
        The unifier prevents that in four steps:
      </p>
      <ol className="steps">
        <li><b>Fingerprint</b> each image with SHA-256 (exact bytes) and a 64-bit <i>difference hash</i> (dHash) that survives resizing and recompression.</li>
        <li><b>Search</b> a BK-tree for every pair of hashes within <code>--duplicate-hamming</code> bits (default 4).</li>
        <li><b>Merge</b> matching pairs into connected components with union-find. Each component is one duplicate group.</li>
        <li><b>Split groups, not images.</b> A stratified group split assigns whole groups to train, val or test, then checks every class still has training examples.</li>
      </ol>
      <Callout title="Source test sets stay test">
        Images that were already in a source&apos;s test split stay in the unified test split by default, and any duplicate group touching
        one goes to test with it. Use <code>--repartition-source-test</code> only when you deliberately want to reshuffle them.
      </Callout>

      <h3>Step 3 · Run it</h3>
      <CodeBlock code={`
python scripts/unify_logo_datasets.py \\
    --data datasets/flickrlogos_yolo/data.yaml \\
           datasets/openlogo_yolo_multiclass/data.yaml \\
    --class-map scripts/logo_class_map_openlogo_flickr.yaml \\
    --output datasets/logos_unified_v1 \\
    --ratios 0.8 0.1 0.1 \\
    --seed 42`} />
      <table className="plain">
        <thead><tr><th>Flag</th><th>What it controls</th></tr></thead>
        <tbody>
          <tr><td><code>--ratios 0.8 0.1 0.1</code></td><td>Train/val/test proportions for images not pinned to a source test split.</td></tr>
          <tr><td><code>--seed 42</code></td><td>Same inputs + same seed = identical file names and split assignment.</td></tr>
          <tr><td><code>--duplicate-hamming N</code></td><td>Perceptual-hash distance for grouping; <code>0</code> means exact-byte duplicates only.</td></tr>
          <tr><td><code>--min-images-per-class N</code></td><td>Require N independent duplicate groups for every class in every split.</td></tr>
          <tr><td><code>--allow-missing-labels-as-negative</code></td><td>Treat images with no label file as background. Off by default: a missing file is an error.</td></tr>
          <tr><td><code>--unmapped-labels exclude-image</code></td><td>With an explicit class list, skip images containing a label outside it.</td></tr>
        </tbody>
      </table>
      <h3>What it writes</h3>
      <ul>
        <li><code>data.yaml</code> with the new <code>nc</code> and <code>names</code>, ready for training.</li>
        <li><code>train/ val/ test/</code> image and label folders with remapped IDs.</li>
        <li><code>unification_manifest.json</code>: every output file, its source, SHA-256 and difference hash.</li>
        <li><code>unification_report.json</code>: per-class support per split, duplicate groups, requested vs. actual ratios, warnings.</li>
      </ul>
      <Callout tone="warn" title="New class IDs, new head">
        The unified class IDs are newly assigned. A checkpoint trained on FlickrLogos-32 alone is not compatible unless names and order
        match exactly, which is why the combined model is trained from <code>yolov7.pt</code>, not from the old <code>best.pt</code>.
      </Callout>
    </Section>
  );
}

function Audit() {
  return (
    <Section
      id="audit"
      kicker="06 · Data audit"
      title="Proving the labels are correct before training"
      lead="Two layers of checks. dataset_validation.py is a hard preflight gate that training runs automatically; audit_dataset.py produces a human-readable report with statistics and previews."
    >
      <h3>Preflight validation: what is checked</h3>
      <div className="two-col">
        <div className="card">
          <h4>Errors (training refuses to start)</h4>
          <ul className="tight">
            <li><code>nc</code> missing or not positive; <code>names</code> length ≠ <code>nc</code>; empty names</li>
            <li><code>train</code> or <code>val</code> split missing, or their image folders empty</li>
            <li>A label row without exactly 5 values, or with non-numeric values</li>
            <li>Class ID outside <code>0 … nc−1</code></li>
            <li>Coordinates not normalised, zero-size boxes, or a box extending outside the image</li>
            <li>Label files with no matching image; duplicate image stems</li>
            <li>The same image file in two splits</li>
            <li>A class with no training annotations; train or val with no valid boxes</li>
          </ul>
        </div>
        <div className="card">
          <h4>Warnings (reported, not blocking)</h4>
          <ul className="tight">
            <li>Images without a label file (treated as background negatives)</li>
            <li>Classes with no validation or test annotations: their per-class score cannot be measured</li>
            <li>Classes with fewer than 5 validation objects: their per-class metrics will be noisy</li>
          </ul>
          <p className="muted">
            Pass <code>--min-val-objects-per-class 5</code> to <code>train_yolo.py</code> to turn that last warning into a hard stop.
          </p>
        </div>
      </div>
      <LabelChecker />
      <h3>The audit report</h3>
      <CodeBlock code={`
python scripts/audit_dataset.py \\
    --data datasets/logos_unified_v1/data.yaml \\
    --output reports/logos_unified_v1_audit`} />
      <table className="plain">
        <thead><tr><th>Output</th><th>Use it to…</th></tr></thead>
        <tbody>
          <tr><td><code>audit_report.json</code></td><td>See every finding in one place, including the preflight validator&apos;s result.</td></tr>
          <tr><td><code>class_counts.csv</code></td><td>Spot class imbalance: brands with a handful of boxes will have weak, noisy recall.</td></tr>
          <tr><td><code>box_statistics.csv</code></td><td>Check box sizes and aspect ratios; tiny boxes argue for a larger <code>--img-size</code>.</td></tr>
          <tr><td><code>duplicate_images.csv</code></td><td>Confirm no exact duplicate crosses splits (skip with <code>--skip-hashes</code>).</td></tr>
          <tr><td>Annotated previews</td><td>Eyeball up to 20 images per split with boxes drawn (<code>--preview-count</code>).</td></tr>
        </tbody>
      </table>
      <h3>Manual review and re-splitting</h3>
      <CodeBlock code={`
# One labelled contact sheet per split for a quick visual pass
python scripts/review_dataset_samples.py \\
    --data datasets/logos_unified_v1/data.yaml --output reports/dataset_review

# Re-split train/val while keeping the held-out test images fixed
python scripts/prepare_dataset_splits.py \\
    --data datasets/logos_yolo/data.yaml --output datasets/logos_yolo_multiclass_ready \\
    --ratios 0.8 0.1 0.1 --seed 42 --preserve-test`} />
      <Callout title="The audit reports; it never edits">
        None of these scripts modify labels. Fix problems in the source, re-run conversion and unification, then audit again.
      </Callout>
    </Section>
  );
}

function Config() {
  return (
    <Section
      id="config"
      kicker="07 · YAML configuration"
      title="Three YAML files define a training run"
      lead="The dataset file says what to learn, the model file says which network learns it, and the hyper-parameter file says how."
    >
      <h3>1 · Dataset: <code>data.yaml</code></h3>
      <CodeBlock lang="yaml" title="datasets/logos_unified_v1/data.yaml (generated)" code={`
path: datasets/logos_unified_v1   # optional root for the split paths
train: train/images
val:   val/images
test:  test/images
nc: 32                             # must equal len(names)
names: [HP, adidas, aldi, apple, becks, bmw, carlsberg, chimay, cocacola, ...]`} />
      <p>
        <code>names</code> may also be a mapping (<code>{'{0: HP, 1: adidas, …}'}</code>); the validator requires its keys to be contiguous
        and start at 0. The order of <code>names</code> is the meaning of every class ID, and the evaluator checks that the
        checkpoint&apos;s embedded names match it.
      </p>
      <h3>2 · Model: <FileTag path="yolov7/cfg/training/yolov7.yaml" /></h3>
      <CodeBlock lang="yaml" title="yolov7/cfg/training/yolov7.yaml (excerpt)" code={`
nc: 80                 # overridden at train time by data.yaml's nc
depth_multiple: 1.0
width_multiple: 1.0
anchors:
  - [12,16, 19,36, 40,28]          # P3/8   small logos
  - [36,75, 76,55, 72,146]         # P4/16  medium
  - [142,110, 192,243, 459,401]    # P5/32  large
# backbone: stacked ELAN blocks → SPPCSPC
# head:     PANet-style top-down + bottom-up fusion → IDetect(P3, P4, P5)`} />
      <p>
        Only the final detection layer depends on the number of classes. When training starts from <code>yolov7.pt</code>, every
        compatible layer is transferred from COCO and the head is rebuilt for your <code>nc</code>. Auto-anchor checks whether the anchors
        fit your box shapes and re-fits them if not.
      </p>
      <h3>3 · Hyper-parameters: <FileTag path="yolov7/data/hyp.scratch.p5.no_ota.yaml" /></h3>
      <CodeBlock lang="yaml" title="yolov7/data/hyp.scratch.p5.no_ota.yaml (excerpt)" code={`
lr0: 0.01            # initial SGD learning rate
lrf: 0.1             # final LR = lr0 × lrf (one-cycle schedule)
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
box: 0.05            # loss gains
cls: 0.3
obj: 0.7
hsv_h: 0.015         # colour jitter
hsv_s: 0.7
hsv_v: 0.4
translate: 0.2
scale: 0.9
fliplr: 0.5          # horizontal flip probability
mosaic: 1.0          # 4-image mosaic
mixup: 0.15
loss_ota: 0          # OTA assignment off: avoids NaN losses at batch size 4`} />
      <Callout tone="warn" title="A note on horizontal flips for logos">
        <code>fliplr: 0.5</code> mirrors half the training images. Mirrored text logos do not occur naturally, so if you see the model
        confusing word marks, try a run with <code>fliplr: 0.0</code> and compare the per-class AP.
      </Callout>
    </Section>
  );
}

function Training() {
  return (
    <Section
      id="training"
      kicker="08 · Training"
      title="Training YOLOv7 with guard rails"
      lead="train_yolo.py validates the dataset, normalises paths and then launches yolov7/train.py. If the data does not match what you expect, it stops before training rather than after."
    >
      <h3>How YOLOv7 learns to find logos</h3>
      <div className="three-col">
        <div className="card">
          <h4>Backbone</h4>
          <p>Stacked ELAN (Efficient Layer Aggregation) blocks turn the image into feature maps at strides 8, 16 and 32, ending in an SPPCSPC block that pools context at several scales.</p>
        </div>
        <div className="card">
          <h4>Neck</h4>
          <p>A PANet-style path mixes coarse, semantic features with fine, spatial ones, so a small logo on a bottle and a large logo on a billboard both get useful features.</p>
        </div>
        <div className="card">
          <h4>Head</h4>
          <p>For every grid cell and each of 3 anchors per scale, predicts box offsets, an objectness score and one probability per brand. NMS removes overlapping duplicates.</p>
        </div>
      </div>
      <p>
        The loss has three parts that match the curves below: a <b>box</b> loss (CIoU between predicted and true box), an{' '}
        <b>objectness</b> loss (is there a logo here at all?) and a <b>classification</b> loss (which brand?). Training starts from COCO
        weights, so the network already knows edges, textures and shapes and only has to learn brands.
      </p>
      <h3>The command</h3>
      <CodeBlock code={`
python scripts/train_yolo.py \\
    --data datasets/logos_unified_v1/data.yaml \\
    --expect-classes <nc from data.yaml> \\
    --min-val-objects-per-class 5 \\
    --weights yolov7/yolov7.pt \\
    --hyp yolov7/data/hyp.scratch.p5.no_ota.yaml \\
    --epochs 100 --img-size 416 --batch-size 4 --workers 0 --device 0 \\
    --name logos_unified_v1`} />
      <table className="plain">
        <thead><tr><th>Flag</th><th>Why it is there</th></tr></thead>
        <tbody>
          <tr><td><code>--expect-classes</code></td><td>Aborts unless <code>data.yaml</code> has exactly this many classes. Stops you training on the wrong dataset.</td></tr>
          <tr><td><code>--min-val-objects-per-class</code></td><td>Aborts if any brand has too few validation boxes to measure.</td></tr>
          <tr><td><code>--weights yolov7/yolov7.pt</code></td><td>COCO-pretrained transfer learning. Never warm-start from a single-class checkpoint: its head has one output.</td></tr>
          <tr><td><code>--hyp …no_ota.yaml</code></td><td>Disables OTA label assignment, which produced NaN losses at batch size 4 (seen on the 291-class combined run).</td></tr>
          <tr><td><code>--img-size 416 --batch-size 4</code></td><td>Fits a 4 GB GPU. Remember the image size; evaluation and inference must use the same value.</td></tr>
          <tr><td><code>--single-cls</code></td><td>Leave off. It collapses every brand into one “logo” class.</td></tr>
        </tbody>
      </table>
      <h3>Resuming an interrupted run</h3>
      <CodeBlock code={`
python scripts/train_yolo.py ...same flags... \\
    --resume runs/train/logos_unified_v1/weights/best.pt`} />
      <p>
        Resume from <code>best.pt</code>, not <code>last.pt</code>. <code>best.pt</code> is only overwritten when validation fitness
        improves, so a bad epoch can never corrupt it; <code>last.pt</code> is overwritten every epoch.
      </p>
      <h3>Watching a run: the console log</h3>
      <Figure
        wide
        name="training_log_epochs_3_12.png"
        alt="YOLOv7 console log for epochs 3 to 12 showing losses, precision, recall and mAP per epoch"
        caption="Fig. T1. YOLOv7 console output for epochs 3–12 of a 100-epoch run (416 px, batch 4, 2.04 GB GPU memory). Each epoch has two rows: training losses (box, obj, cls, total) over 446 iterations, then validation on 232 images with 373 boxes."
      />
      <TrainingDynamics />
      <div className="two-col">
        <Reveal className="card">
          <h4>What the first epochs show</h4>
          <ul className="tight">
            <li>mAP@0.5 rises from 0.08 to 0.49 in nine epochs while the total loss falls steadily from 0.106 to 0.081. The detector is still far from converged at this point.</li>
            <li>Precision swings (0.06 → 0.38 → 0.29 → 0.46) while recall climbs more steadily. Early on, confidence scores are poorly calibrated: the single operating point YOLOv7 reports moves around even when the underlying ranking (mAP) improves monotonically.</li>
            <li>The gap between mAP@0.5 and mAP@0.5:0.95 stays near 0.14. Boxes are found before they are tight; the strict IoU levels catch up later in training.</li>
          </ul>
        </Reveal>
        <Reveal className="card" delay={100}>
          <h4>A systems observation</h4>
          <p>
            Epoch time grows from 7 min 25 s (epochs 4–8) to about 34.5 min (epochs 10–12), and validation slows from 22 s to 1 min 47 s.
            Training and validation slow by the same 4–5×, which points to the machine rather than the model: most likely GPU thermal
            or power throttling, or another process competing for the GPU. With <code>--workers 0</code> data loading also runs on the
            main process, so CPU contention shows up directly.
          </p>
          <p className="muted">Worth checking with <code>nvidia-smi -q -d PERFORMANCE,TEMPERATURE</code> during the next run.</p>
        </Reveal>
      </div>
      <Figure
        wide
        name="training_curves.png"
        alt="Training and validation loss curves with precision, recall and mAP over 100 epochs"
        caption="The FlickrLogos-32 run: box, objectness and classification losses fall steadily on both train and validation, while mAP@0.5 plateaus near 0.95 by epoch 40 and mAP@0.5:0.95 keeps climbing to about 0.8."
      />
      <Callout title="Reading the curves">
        Validation losses tracking training losses means no overfitting. If validation loss turns upward while training loss keeps
        falling, stop early and use <code>best.pt</code>. Record every run with <FileTag path="scripts/record_experiment.py" /> so the
        command, environment, git revision and dataset hash are saved under <code>reports/experiments/</code>.
      </Callout>
    </Section>
  );
}

function Evaluation() {
  return (
    <Section
      id="evaluation"
      kicker="09 · Evaluation"
      title="Scoring the checkpoint on data it has never seen"
      lead="evaluate_yolo.py runs YOLOv7's test routine on the held-out test split and adds the reports you need to decide whether a model is good enough."
    >
      <CodeBlock code={`
python scripts/evaluate_yolo.py \\
    --data datasets/logos_unified_v1/data.yaml \\
    --weights runs/train/logos_unified_v1/weights/best.pt \\
    --img-size 416 --device 0`} />
      <div className="two-col">
        <div className="card">
          <h4>Safety checks before scoring</h4>
          <ul className="tight">
            <li>The checkpoint&apos;s embedded class count and name order must match <code>data.yaml</code>. Otherwise scores would be computed against the wrong brands.</li>
            <li><code>--no-half</code> is always passed: FP16 inference silently produced NaN results on some GPUs.</li>
            <li><code>--img-size</code> should match training; YOLOv7 defaults to 640.</li>
          </ul>
        </div>
        <div className="card">
          <h4>Reports in <code>reports/checkpoint_evaluation/&lt;run&gt;/</code></h4>
          <ul className="tight">
            <li><code>evaluation_report.json</code>: overall and per-class numbers</li>
            <li><code>per_class_metrics.csv</code>: AP50, AP50-95, precision, recall, F1</li>
            <li><code>false_positives.csv</code>, <code>missed_logos.csv</code> + annotated examples</li>
            <li>Confusion matrix, PR curve and F1 curve images</li>
          </ul>
        </div>
      </div>
      <p>
        Pass several checkpoints after <code>--weights</code> to compare runs on identical data. Error lists use confidence 0.25 and IoU 0.5
        by default (<code>--report-conf</code>, <code>--match-iou</code>).
      </p>
      <h3>Results of the current checkpoint</h3>
      <div className="kpis inline">
        {KPIS.map((k) => (
          <div key={k.label} className="kpi">
            <div className="kpi-value"><CountUp value={k.value} /></div>
            <div className="kpi-label">{k.label}</div>
          </div>
        ))}
        <div className="kpi-note">TP 300 · FP 50 · missed 23, of 323 ground-truth logos in 224 test images (FlickrLogos-32, 32 classes).</div>
      </div>
      <div className="gallery two">
        <Figure name="eval_confusion_matrix.png" alt="Confusion matrix" caption="Confusion matrix. The diagonal is correct; most mix-ups are between visually similar beer labels." />
        <div>
          <Figure name="eval_pr_curve.png" alt="Precision-recall curve" caption="Precision-recall curve per class and averaged." />
          <Figure name="eval_f1_curve.png" alt="F1 versus confidence" caption="F1 against confidence threshold: pick the confidence where the curve peaks." />
        </div>
      </div>
      <MetricsTable />
      <Callout title="Where the model is weakest">
        Carlsberg, HP, UPS and Tsingtao have the lowest strict AP. They combine few test boxes with visual overlap (Carlsberg and Heineken
        share green-and-gold shield labels). Adding OpenLogo samples for these brands is the main motivation for the combined model.
      </Callout>
    </Section>
  );
}

function Metrics() {
  return (
    <Section
      id="metrics"
      kicker="10 · Understanding metrics"
      title="What precision, recall and mAP actually measure"
      lead="Every number in the report starts from one decision: does a predicted box match a ground-truth box of the same class closely enough?"
    >
      <IoUPlayground />
      <PRCalculator />
      <table className="plain">
        <thead><tr><th>Term</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><b>Confidence</b></td><td>objectness × class probability. Predictions below <code>--conf</code> are discarded. Raise it for fewer false alarms, lower it to catch more logos.</td></tr>
          <tr><td><b>IoU</b></td><td>Intersection over union of two boxes. A prediction is a true positive when IoU ≥ 0.5 with an unmatched ground-truth box of the same class.</td></tr>
          <tr><td><b>Precision</b></td><td>Of everything predicted, how much was right: TP ÷ (TP + FP). Low precision means false alarms.</td></tr>
          <tr><td><b>Recall</b></td><td>Of every real logo, how many were found: TP ÷ (TP + FN). Low recall means missed logos.</td></tr>
          <tr><td><b>F1-score</b></td><td>Harmonic mean of precision and recall. Its peak on the F1 curve is the best single confidence threshold.</td></tr>
          <tr><td><b>AP@0.5</b></td><td>Area under one class&apos;s precision-recall curve as confidence sweeps 0 → 1, with matches at IoU ≥ 0.5.</td></tr>
          <tr><td><b>AP@0.5:0.95</b></td><td>AP averaged over ten IoU thresholds (0.50, 0.55 … 0.95). Rewards tight boxes, not just roughly right ones.</td></tr>
          <tr><td><b>mAP</b></td><td>The mean of AP over all classes. Every brand counts equally, however many examples it has.</td></tr>
        </tbody>
      </table>
    </Section>
  );
}

function Inference() {
  return (
    <Section
      id="inference"
      kicker="11 · Image & video inference"
      title="Running the detector on new media"
      lead="detect_yolo.py wraps yolov7/detect.py. It accepts an image, a video, or a folder of either, and writes annotated media plus YOLO-format label files with confidences."
    >
      <CodeBlock code={`
mkdir -p test_images test_videos     # not committed: add your own media

python scripts/detect_yolo.py \\
    --weights runs/train/logos_unified_v1/weights/best.pt \\
    --source test_images/ \\
    --img-size 416 --conf 0.4 --iou 0.45

python scripts/detect_yolo.py \\
    --weights runs/train/logos_unified_v1/weights/best.pt \\
    --source test_videos/ \\
    --img-size 416 --conf 0.4`} />
      <p>
        Output goes to <code>runs/detect/&lt;name&gt;/</code>. Useful extras: <code>--classes</code> to keep only some brands,{' '}
        <code>--device cpu</code> without a GPU, <code>--nosave</code> to write labels only.
      </p>
      <h3>Detections on unseen advertisement footage</h3>
      <div className="gallery two">
        <Figure name="detection_pepsi.png" alt="Pepsi detected" caption="Pepsi, 0.95 confidence" />
        <Figure name="detection_stellaartois.png" alt="Stella Artois detected" caption="Stella Artois, 0.89 confidence" />
        <Figure name="detection_paulaner_multi.png" alt="Several Paulaner logos" caption="Paulaner, four logos in one busy frame" />
        <Figure name="detection_cocacola.png" alt="Coca-Cola on a truck" caption="Coca-Cola, 0.70: small, angled logo" />
      </div>
    </Section>
  );
}

function Gui() {
  return (
    <Section
      id="gui"
      kicker="13 · PyQt5 application"
      title="Logo Detection Studio: the desktop app"
      lead="video_detection_gui.py puts the detection pipeline behind a PyQt5 window: choose media, choose a checkpoint, run a batch and play back the annotated results without touching the terminal."
    >
      <CodeBlock code={`
pip install -r scripts/requirements-gui.txt
python scripts/video_detection_gui.py`} />
      <Figure wide name="PyQT_GUI.png" alt="The PyQt5 Logo Detection Viewer playing back an annotated video" caption="A batch in progress, with a finished video's annotated output playing inline (Heineken boxed at bottom right)." />
      <div className="three-col">
        <div className="card">
          <h4>1 · Select images or videos</h4>
          <p>Toggle between <code>test_videos/</code> and <code>test_images/</code>, or browse to any folder. Tick files or whole folders (selected recursively), or drag them in from your file manager.</p>
        </div>
        <div className="card">
          <h4>2 · Model and settings</h4>
          <p>Pick the <code>best.pt</code> checkpoint and output folder, then set confidence (default 0.4), NMS IoU (0.45), image size and device (GPU 0 or CPU). Optionally save label files with scores.</p>
        </div>
        <div className="card">
          <h4>3 · Run and review</h4>
          <p>Each item runs as a separate process with a progress bar and live log. Cancel at any time. Select a finished item to view the image or play the video in the window.</p>
        </div>
      </div>
      <Callout tone="warn" title="Two settings to change for a new model">
        The weights field defaults to an older checkpoint path, and the image size defaults to 640. Point it at the new{' '}
        <code>runs/train/&lt;run&gt;/weights/best.pt</code> and set the image size to the value you trained with (416 in the commands above).
      </Callout>
      <p>
        Every batch gets a timestamped folder with one YOLO output folder per media file plus <code>summary.csv</code> and{' '}
        <code>summary.json</code> recording each item&apos;s status. Under the hood the window uses <code>QProcess</code> to call the same
        detection code as the CLI, so GUI and CLI results are identical for the same settings. The terminal picker{' '}
        <FileTag path="scripts/batch_video_detect.py" /> offers the same batch flow without a window.
      </p>
      <Figure wide name="gui_screenshot.png" alt="Annotated Coca-Cola billboard frame" caption="Annotated playback of a video frame: Coca-Cola detected at 0.89 on a roadside billboard." />
    </Section>
  );
}

function Beyond() {
  return (
    <Section
      id="beyond"
      kicker="14 · Beyond detection"
      title="What else is built on the detector"
      lead="The trained checkpoint also feeds the repository's research tools. They are optional for detection but share the same weights."
    >
      <div className="two-col">
        <div className="card">
          <h4>Logo flipping pipeline</h4>
          <p><FileTag path="scripts/logo_flipping_pipeline.py" />: YOLO (+ optional EasyOCR) → MobileSAM / GrabCut segmentation → symmetry line and perspective check → reflect → inpaint (LaMa or OpenCV Telea) → blend back.</p>
        </div>
        <div className="card">
          <h4>Video tracking</h4>
          <p><FileTag path="scripts/video_logo_tracking.py" />: Hungarian matching on CLIP appearance + motion cost, Kalman filter and optical flow, MediaPipe hand-occlusion handling with cached results.</p>
        </div>
        <div className="card">
          <h4>Unknown logos</h4>
          <p><FileTag path="scripts/detect_unknown_logos.py" /> compares detected regions with a CLIP gallery of reference logos and labels low-similarity crops <code>unknown</code>. Thresholds are tuned with <code>calibrate_unknown_logos.py</code>.</p>
        </div>
        <div className="card">
          <h4>Benchmarks and reproducibility</h4>
          <p><code>benchmark_video_detection.py</code> compares checkpoint speed (FPS) on the same videos; <code>evaluate_tracking.py</code> scores tracks; <code>record_experiment.py</code> logs every run.</p>
        </div>
      </div>
    </Section>
  );
}

const RECIPE = [
  ['Convert OpenLogo', 'python scripts/convert_voc_to_yolo.py --source /path/to/openlogo --output datasets/openlogo_yolo_multiclass --seed 42'],
  ['Convert FlickrLogos-32', 'python scripts/convert_flickrlogos32.py /path/to/FlickrLogos-32 --output-dir datasets/flickrlogos_coco\npython scripts/convert_coco_to_yolo.py --input datasets/flickrlogos_coco --output datasets/flickrlogos_yolo'],
  ['Unify both', 'python scripts/unify_logo_datasets.py --data datasets/flickrlogos_yolo/data.yaml datasets/openlogo_yolo_multiclass/data.yaml --class-map scripts/logo_class_map_openlogo_flickr.yaml --output datasets/logos_unified_v1 --ratios 0.8 0.1 0.1 --seed 42'],
  ['Audit', 'python scripts/audit_dataset.py --data datasets/logos_unified_v1/data.yaml --output reports/logos_unified_v1_audit'],
  ['Read nc', 'grep "^nc:" datasets/logos_unified_v1/data.yaml'],
  ['Train', 'python scripts/record_experiment.py --name logos_unified_v1 --data datasets/logos_unified_v1/data.yaml -- \\\n  python scripts/train_yolo.py --data datasets/logos_unified_v1/data.yaml --expect-classes <nc> --min-val-objects-per-class 5 --weights yolov7/yolov7.pt --hyp yolov7/data/hyp.scratch.p5.no_ota.yaml --epochs 100 --img-size 416 --batch-size 4 --workers 0 --device 0 --name logos_unified_v1'],
  ['Evaluate', 'python scripts/evaluate_yolo.py --data datasets/logos_unified_v1/data.yaml --weights runs/train/logos_unified_v1/weights/best.pt --img-size 416 --device 0'],
  ['Run the app', 'python scripts/video_detection_gui.py   # weights: runs/train/logos_unified_v1/weights/best.pt, image size 416'],
];

function Recipe() {
  const [done, setDone] = useState(() => new Set());
  const toggle = (i) =>
    setDone((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  return (
    <Section
      id="recipe"
      kicker="15 · Retraining recipe"
      title="Training the combined FlickrLogos-32 + OpenLogo model"
      lead="The whole pipeline in order, for replacing the outdated checkpoint. Tick steps off as you go; the checklist lives only in this browser tab."
    >
      <ol className="recipe">
        {RECIPE.map(([title, cmd], i) => (
          <li key={title} className={done.has(i) ? 'done' : ''}>
            <label>
              <input type="checkbox" checked={done.has(i)} onChange={() => toggle(i)} />
              <span className="recipe-n">{i + 1}</span>
              <b>{title}</b>
            </label>
            <CodeBlock code={cmd} />
          </li>
        ))}
      </ol>
      <Callout title="Before you call the model done">
        Compare the new <code>per_class_metrics.csv</code> against the FlickrLogos-32 table above for the 32 shared brands. The combined
        model should hold or improve those, and its overall mAP is only comparable to the old one on the same test images. Review the
        unification report&apos;s duplicate groups and classes with sparse validation support before trusting per-class numbers.
      </Callout>
    </Section>
  );
}

const FAILURE_MODES = [
  {
    mode: 'Inter-class confusion',
    examples: `${ref('n')} Heineken → carlsberg 0.66; ${ref('m')} Esso badge → stellaartois 0.79, paulaner 0.76`,
    cause: 'Brands that share colour scheme and emblem shape; at small scale the lettering that separates them is lost.',
    remedy: 'More examples of the confusable pairs (OpenLogo adds both); review the confusion matrix per pair after retraining.',
  },
  {
    mode: 'Loss of colour cue',
    examples: `${ref('o')} grayscale Esso → bmw 0.50; ${ref('p')} white-on-black Esso print → ups 0.75`,
    cause: 'The detector leans on brand colours; monochrome renderings leave only shape, which other brands share.',
    remedy: 'Add grayscale/desaturation augmentation and monochrome or printed-on-fabric training examples.',
  },
  {
    mode: 'Correct but low confidence',
    examples: `${ref('h')} pepsi 0.41 under a paw; ${ref('coke_toy')} cocacola 0.48 on a tiny, blurred toy truck; ${ref('f')} pepsi 0.52 on a re-captured screen`,
    cause: 'The brand is right, but occluded or tiny logos give weaker evidence, so scores fall towards the threshold.',
    remedy: 'Not an error at 0.40, but a high threshold hides these. In video, tracking carries the identity through such frames.',
  },
  {
    mode: 'Missed tiny background logos',
    examples: `${ref('m')} Esso logos on background trucks in the 5×5 video wall`,
    cause: 'Logos only a few pixels tall are under-represented in training.',
    remedy: 'Train at a larger --img-size if VRAM allows; mosaic augmentation already creates some truncation.',
  },
];

function Qualitative() {
  return (
    <Section
      id="qualitative"
      kicker="12 · Qualitative results"
      title="Qualitative results on unseen advertisement footage"
      lead="The test-split numbers say how often the detector is right. These frames show where and why it is right or wrong, which is what guides the next round of data collection."
    >
      <Reveal className="callout">
        <strong>Protocol</strong>
        <div>
          {SAMPLES.length} frames were captured from the PyQt5 viewer while it ran <code>detect_yolo.py</code> with the 32-class FlickrLogos-32 checkpoint at
          confidence 0.4 and NMS IoU 0.45. The footage is commercial advertising video that played no part in training, validation or
          testing. These frames have no ground-truth annotations, so each verdict is a visual judgement against the brand on screen, and the
          set was chosen to show both strengths and failure modes, not to estimate accuracy. Move the threshold slider to see which boxes
          survive each setting. Click any image to enlarge it; use ← and → to step through.
        </div>
      </Reveal>
      <h3>Choosing a stable confidence threshold</h3>
      <p>
        Every box in these frames carries a confidence score, and <code>--conf</code> decides which ones are shown. The F1 curve from the
        test split (Fig. Q0) is the principled place to pick it. It peaks at <b>F1 = 0.93 at confidence 0.774</b>, and it forms a broad
        plateau: anywhere from about 0.45 to 0.85, F1 stays within a few points of that peak. A value on the plateau is{' '}
        <i>stable</i>: small shifts in lighting or scale move scores a little, but they do not tip the operating point off a cliff.
      </p>
      <div className="two-col">
        <Figure name="eval_f1_curve.png" alt="F1 versus confidence threshold, peaking at 0.93 at 0.774" caption="Fig. Q0. F1 against confidence on the 224-image test split. Grey: individual classes. Blue: all classes, best F1 0.93 at 0.774." />
        <div>
          <table className="plain">
            <thead><tr><th>Threshold</th><th>Correct kept</th><th>Wrong kept</th><th>Use it for</th></tr></thead>
            <tbody>
              <tr><td><code>0.40</code></td><td>39 / 39</td><td>5 / 5</td><td>Maximum recall. Every logo in Fig. Q1 is shown, including occluded and tiny ones.</td></tr>
              <tr><td><code>0.50</code></td><td>37 / 39</td><td>5 / 5</td><td>Video tracking default; tracking recovers the few frames it drops.</td></tr>
              <tr className="recommended"><td><code>0.77</code> ★</td><td>29 / 39</td><td>1 / 5</td><td>Recommended for an image demo. Measured optimum on 323 labelled logos; removes 4 of the 5 misclassifications.</td></tr>
              <tr><td><code>0.80</code></td><td>27 / 39</td><td>0 / 5</td><td>Zero misclassifications on these frames, at the cost of hiding small and occluded logos.</td></tr>
            </tbody>
          </table>
          <p className="muted">
            Counts are over the 44 boxes in Fig. Q1 (39 correct, 5 wrong). These frames were picked by hand, so treat them as an
            illustration; the 0.774 optimum comes from the labelled test split.
          </p>
        </div>
      </div>
      <Callout title="Recommendation">
        Run the image demo at <code>--conf 0.77</code>, the test-split optimum. For video, use <code>0.50</code>: a logo that drops
        below threshold for one frame usually reappears in the next, so recall matters more than per-frame precision. Try both with
        the slider below.
      </Callout>

      <h3>Fig. Q1 · Detections across scenes, scales and degradations</h3>
      <QualitativeGallery />

      <h3>Failure analysis</h3>
      <p>
        The failures fall into four recurring modes. None of them is random noise: each points to a specific gap in the training data or a
        specific setting, which makes them actionable for the combined FlickrLogos-32 + OpenLogo model.
      </p>
      <div className="table-wrap plain-wrap">
        <table className="plain">
          <thead><tr><th>Failure mode</th><th>Observed in</th><th>Likely cause</th><th>Remedy</th></tr></thead>
          <tbody>
            {FAILURE_MODES.map((f) => (
              <tr key={f.mode}>
                <td><b>{f.mode}</b></td><td>{f.examples}</td><td>{f.cause}</td><td>{f.remedy}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="three-col">
        <Reveal className="card finding">
          <span className="finding-n">1</span>
          <h4>Robust to scene conditions</h4>
          <p>Colour casts, film grain, snow cover, curved glass, overlapping objects, AI-generated imagery and busy backgrounds do not break detection; frontal logos stay at 0.87–0.97.</p>
        </Reveal>
        <Reveal className="card finding" delay={100}>
          <span className="finding-n">2</span>
          <h4>Colour is a crutch</h4>
          <p>Both high-confidence misclassifications of Esso happen when the logo appears without its colours. Grayscale augmentation is the cheapest experiment to run next.</p>
        </Reveal>
        <Reveal className="card finding" delay={200}>
          <span className="finding-n">3</span>
          <h4>Errors match the metrics</h4>
          <p>The Heineken/Carlsberg confusion seen here is the same pair the confusion matrix and the per-class AP table flag. Offline metrics predict field behaviour.</p>
        </Reveal>
      </div>
    </Section>
  );
}

const BIBTEX = `@misc{tozluyurt2026logodetection,
  author       = {Tozluyurt, H{\\"u}seyin},
  title        = {Brand Logo Detection with {YOLOv7}: Dataset Unification,
                  Label Auditing, Training and Evaluation},
  year         = {2026},
  howpublished = {\\url{https://github.com/huseyinTozluyurt/Neural-Bridge-Internship-Logo-Detection}},
  note         = {Neural Bridge internship project}
}`;

function Cite() {
  return (
    <Section id="cite" kicker="16 · Cite" title="Citing this work" lead="If this pipeline or its analysis helps your research, please cite the repository.">
      <CodeBlock lang="bibtex" title="BibTeX" code={BIBTEX} />
      <p className="muted">
        Please also cite YOLOv7 (Wang, Bochkovskiy and Liao, CVPR 2023), FlickrLogos-32 (Romberg et al., ICMR 2011) and OpenLogo (Su, Gong
        and Zhu, 2018) when you use those components.
      </p>
    </Section>
  );
}
