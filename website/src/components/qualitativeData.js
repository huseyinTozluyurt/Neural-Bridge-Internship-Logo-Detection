// Qualitative results on unseen advertisement footage, captured from the
// PyQt5 viewer (detect_yolo.py, conf 0.4, IoU 0.45). Verdicts are judged
// by eye against the visible brand; there is no ground truth for these frames.
const RAW = [
  {
    key: 'a', file: 'qual_pepsi_can.jpg', verdict: 'correct', brand: 'Pepsi',
    dets: [[0.95, true]],
    pred: 'pepsi 0.95',
    title: 'Canonical frontal view',
    text: 'A frontal, well-lit can is the easiest case: the box hugs the globe emblem tightly and confidence is near saturation. Condensation and specular highlights on the can do not disturb the detection, and the black redesigned livery, which differs from the older blue cans in FlickrLogos-32, is still recognised from the globe alone.',
  },
  {
    key: 'b', file: 'qual_paulaner_logo.jpg', verdict: 'correct', brand: 'Paulaner',
    dets: [[0.96, true]],
    pred: 'paulaner 0.96',
    title: 'Emblem versus wordmark',
    text: 'The network localises the circular monk emblem and leaves the large PAULANER wordmark below it outside the box. The model has learned the roundel as "the logo", so on brands with separate figurative and text marks it will report only one of them. This is the distinction OpenLogo encodes with its *_text classes, which the alias map merges into the base brand.',
  },
  {
    key: 'c', file: 'qual_paulaner_beergarden.jpg', verdict: 'correct', brand: 'Paulaner',
    dets: [[0.97, true]],
    pred: 'paulaner 0.97',
    title: 'Overlay on a cluttered scene',
    text: 'A graphic overlay on a busy beer-garden shot with people, foliage and a large caption. Background clutter produces no spurious boxes, and the emblem is detected with the highest confidence in the set.',
  },
  {
    key: 'paul_mugs', file: 'qual_paulaner_mugs.jpg', verdict: 'correct', brand: 'Paulaner',
    dets: [[0.81, true], [0.87, true]],
    pred: 'paulaner 0.81 · 0.87 (+2 at frame edge)',
    title: 'Small emblems on stoneware-style mugs',
    text: 'The emblems on two one-litre mugs are found at 0.81 and 0.87, even though each covers well under 1% of the frame and the beer behind the glass changes the background colour. Two more boxes sit on the lantern at the top-right frame edge. Their labels are cut off by the border, but the boxes cover the Paulaner emblems printed on the lantern, so there are four correct detections in one frame.',
  },
  {
    key: 'paul_glass', file: 'qual_paulaner_glass.jpg', verdict: 'correct', brand: 'Paulaner',
    dets: [[0.92, true]],
    pred: 'paulaner 0.92',
    title: 'Emblem on a Weißbier glass',
    text: 'The emblem on the Weißbier glass is detected at 0.92 with a tight box, in front of a softly blurred background and a hand. As in {b}, the box covers the monk roundel rather than the PAULANER wordmark printed below it, so the model consistently treats the emblem as the logo.',
  },
  {
    key: 'paul_toast', file: 'qual_paulaner_toast.jpg', verdict: 'correct', brand: 'Paulaner',
    dets: [[0.82, true], [0.9, true], [0.9, true]],
    pred: 'paulaner 0.82 · 0.90 · 0.90',
    title: 'Crowded toast: overlapping glasses',
    text: 'Four glasses meet in a toast. The three whose emblem faces the camera are detected (0.82–0.90) with separate, tight boxes, even though the glasses overlap and hands cover their lower halves. The fourth glass, seen from behind, shows no emblem and correctly gets no box. Overlapping same-class objects are where non-maximum suppression can wrongly merge detections, and here it does not.',
  },
  {
    key: 'd', file: 'qual_erdinger.jpg', verdict: 'correct', brand: 'Erdinger',
    dets: [[0.79, true], [0.84, true], [0.84, true]],
    pred: 'erdinger 0.79 · 0.84 · 0.84',
    title: 'Multiple small instances',
    text: 'Three instances: the print on the glass and the labels of two adjacent bottles, each under a tenth of the frame height. Non-maximum suppression keeps the neighbouring bottles as separate detections. The glass print is partly hidden by foam and receives the lowest score.',
  },
  {
    key: 'stella', file: 'qual_stellaartois_chalice.jpg', verdict: 'correct', brand: 'Stella Artois',
    dets: [[0.89, true]],
    pred: 'stellaartois 0.89',
    title: 'Label on a curved chalice in a dim bar',
    text: 'The Stella Artois label is wrapped around a curved, condensation-covered chalice under warm, low bar lighting. Shallow depth of field blurs the people and other glasses behind it. The detection is confident and the box closely fits the red label and crest.',
  },
  {
    key: 'e', file: 'qual_pepsi_stadium.jpg', verdict: 'correct', brand: 'Pepsi',
    dets: [[0.75, true], [0.82, true]],
    pred: 'pepsi 0.75 · 0.82',
    title: 'Small logos under a strong colour cast',
    text: 'Two small logos above a stage in a night-time aerial shot dominated by blue light. Both are found, which suggests the HSV colour augmentation in training (hsv_h 0.015, hsv_s 0.7, hsv_v 0.4) gives useful robustness to scene lighting.',
  },
  {
    key: 'f', file: 'qual_pepsi_hologram.jpg', verdict: 'correct', brand: 'Pepsi',
    dets: [[0.52, true], [0.8, true]],
    pred: 'pepsi 0.52 · 0.80',
    title: 'Logos inside a screen inside the scene',
    text: 'The cans appear on a giant display filmed at a concert, so they are re-captured, rotated and partly covered by paws. Both are detected. The more occluded can drops to 0.52, close to the 0.4 operating threshold.',
  },
  {
    key: 'g', file: 'qual_pepsi_bears.jpg', verdict: 'correct', brand: 'Pepsi',
    dets: [[0.66, true]],
    pred: 'pepsi 0.66',
    title: 'Hand-held, rotated, motion-blurred',
    text: 'The can held between the two figures is correctly identified at 0.66 despite rotation, motion blur and the paw wrapped around it. The can fills only a small patch of the frame, so the network has to recognise the brand from a partly covered globe emblem.',
  },
  {
    key: 'h', file: 'qual_pepsi_bear_occluded.jpg', verdict: 'correct', brand: 'Pepsi',
    dets: [[0.41, true]],
    pred: 'pepsi 0.41',
    title: 'Correct under heavy occlusion',
    text: 'About half of the can is covered by a paw and the logo occupies a tiny part of the frame, yet the brand is still identified correctly. The low score (0.41) shows how confidence reacts to occlusion rather than to doubt about the brand. Drag the threshold slider above 0.41 to watch this detection disappear: this is the trade-off the threshold controls.',
  },
  {
    key: 'coke_bottle', file: 'qual_cocacola_bottle.jpg', verdict: 'correct', brand: 'Coca-Cola',
    dets: [[0.87, true]],
    pred: 'cocacola 0.87',
    title: 'Script logo on a contour bottle',
    text: 'The Coca-Cola script on a glass contour bottle, framed by a miniature snow village and a red costume sleeve. Strong red elements elsewhere in the frame (the sleeve, lit windows) do not produce false detections, so the model is keying on the script shape, not just the brand red. This commercial is labelled on screen as AI-generated (see the toy-truck frame), so {coke_bottle}–{coke_billboard} also test generalisation to synthetic imagery.',
  },
  {
    key: 'coke_toy', file: 'qual_cocacola_toytruck.jpg', verdict: 'correct', brand: 'Coca-Cola',
    dets: [[0.48, true]],
    pred: 'cocacola 0.48',
    title: 'Tiny, defocused logo on a toy truck',
    text: 'A miniature truck held by a hand, with the script logo only a few dozen pixels tall, defocused and cut off by the trailer edge. The detection is correct, at a modest 0.48. Like {h}, this frame shows that confidence falls with scale and blur even when the class is right, which is why the choice of threshold matters for a demo.',
  },
  {
    key: 'coke_convoy', file: 'qual_cocacola_convoy.jpg', verdict: 'correct', brand: 'Coca-Cola',
    dets: [[0.7, true]],
    pred: 'cocacola 0.70',
    title: 'Strong perspective on a truck trailer',
    text: 'The script on the trailer side is seen at a steep angle and compressed horizontally to roughly a third of its frontal width, under night-time headlights and fairy lights. It is still found at 0.70 with a box that follows the trailer face.',
  },
  {
    key: 'coke_bridge', file: 'qual_cocacola_bridge.jpg', verdict: 'correct', brand: 'Coca-Cola',
    dets: [[0.67, true], [0.89, true], [0.9, true], [0.76, true]],
    pred: 'cocacola 0.67 · 0.89 · 0.90 · 0.76',
    title: 'A row of instances at different scales',
    text: 'Four trucks cross a bridge, each trailer at a different distance and angle. All four logos are detected, including the right-most one, which is cut off by the frame border (0.76). Confidence is lowest on the farthest truck (0.67), in line with its smaller apparent size, and highest on the two closest (0.89, 0.90).',
  },
  {
    key: 'coke_billboard', file: 'qual_cocacola_billboard.jpg', verdict: 'correct', brand: 'Coca-Cola',
    dets: [[0.89, true]],
    pred: 'cocacola 0.89',
    title: 'Vintage script on an illuminated billboard',
    text: 'A roadside billboard in the classic Coca-Cola advertising style, framed by bulbs and seen at an angle in a dark forest. The box covers only the script wordmark and ignores the illustration and the bright bulb border. The same frame appears in the PyQt5 section as an example of annotated playback.',
  },
  {
    key: 'i', file: 'qual_esso_centenary.jpg', verdict: 'correct', brand: 'Esso',
    dets: [[0.92, true]],
    pred: 'esso 0.92',
    title: 'Logo inside a composite badge',
    text: 'In the centenary badge the model localises only the Esso oval and ignores the surrounding "Celebrating 100 years" artwork and the caption, even though the caption repeats the brand name in plain text.',
  },
  {
    key: 'j', file: 'qual_esso_film_sign.jpg', verdict: 'correct', brand: 'Esso',
    dets: [[0.95, true]],
    pred: 'esso 0.95',
    title: 'Archival film with scratches and colour fading',
    text: 'A scanned film frame with dust, scratches, vignetting and faded colour, viewed at an oblique angle. The detection is confident and the box covers the sign housing. A slightly loose box like this still counts at IoU 0.5 but costs points at the stricter IoU levels of AP@.5:.95.',
  },
  {
    key: 'k', file: 'qual_esso_tank.jpg', verdict: 'correct', brand: 'Esso',
    dets: [[0.95, true]],
    pred: 'esso 0.95',
    title: 'Partial snow cover on a curved surface',
    text: 'Snow covers the top of the oval and the tank surface curves the logo. The box stops at the emblem and correctly excludes the adjacent "Diesel Efficient" text.',
  },
  {
    key: 'l', file: 'qual_esso_truck.jpg', verdict: 'correct', brand: 'Esso',
    dets: [[0.97, true]],
    pred: 'esso 0.97',
    title: 'Large, near-frontal instance',
    text: 'A large logo on a truck panel gets the highest Esso score. The box sits a little wide on the right, towards the trademark symbol, which again matters only at the strict IoU levels of AP@.5:.95.',
  },
  {
    key: 'm', file: 'qual_esso_montage.jpg', verdict: 'mixed', brand: 'Esso',
    dets: [[0.88, true], [0.86, true], [0.7, true], [0.93, true], [0.84, true], [0.82, true], [0.74, true], [0.79, true], [0.79, false], [0.76, false]],
    pred: 'esso 0.70–0.93 · stellaartois 0.79 · paulaner 0.76',
    title: 'Video wall: many small logos, two confusions',
    text: 'A 5×5 mosaic of interview clips. Most small Esso ovals are found (0.70–0.93), but the same centenary badge is twice given the wrong brand: stellaartois 0.79 (top row) and paulaner 0.76 (bottom-left). Both brands also use a round emblem with red and white elements. At this scale the network keeps the shape and colour pattern but loses the lettering that tells the brands apart. Several Esso logos on trucks in the background are missed.',
  },
  {
    key: 'n', file: 'qual_heineken_carlsberg.jpg', verdict: 'error', brand: 'Heineken',
    dets: [[0.66, false]],
    pred: 'carlsberg 0.66 (true: heineken)',
    title: 'Right box, wrong brand: green-label beers',
    text: 'An extreme close-up of a Heineken glass is labelled carlsberg. The box itself is well placed around the star and banner, so the localisation is right and only the class is wrong. Both brands use green, white and red on their labels, and the confusion matrix and the per-class table already flag this pair: Carlsberg has the lowest AP@.5:.95 of all 32 classes (0.52). The scale makes it harder still, because Flickr photos rarely show a logo filling the whole frame. The margin is thin (0.66), and the fix is more Carlsberg and Heineken examples, which the combined dataset adds.',
  },
  {
    key: 'o', file: 'qual_esso_bw_bmw.jpg', verdict: 'error', brand: 'Esso',
    dets: [[0.5, false]],
    pred: 'bmw 0.50 (true: esso)',
    title: 'Grayscale input removes the colour cue',
    text: 'A black-and-white archival photograph of an Esso sign is labelled bmw. Without the red lettering and blue ring, what is left is a ring around an inner oval, which matches the BMW roundel. The score sits right at the threshold. Low-confidence wrong labels like this cluster near the operating point, so raising --conf to 0.55 would remove this error at a small cost in recall. Grayscale augmentation during training would target the cause directly.',
  },
  {
    key: 'p', file: 'qual_esso_shirt_ups.jpg', verdict: 'error', brand: 'Esso',
    dets: [[0.75, false]],
    pred: 'ups 0.75 (true: esso)',
    title: 'Monochrome print on deforming fabric',
    text: 'A white-on-black Esso print on a shirt, distorted by folds and perspective, is labelled ups with fairly high confidence. As in {o}, the colour cue is gone, and the dark field inside a light outline resembles the UPS shield. Unlike {o}, this error would survive a higher threshold. The fix is training data: monochrome and printed-on-fabric logos, which OpenLogo adds.',
  },
];

const LETTERS = 'abcdefghijklmnopqrstuvwxyz';
const ids = Object.fromEntries(RAW.map((r, i) => [r.key, LETTERS[i]]));
// Captions cross-reference other panels as {key}; resolve to (letter).
export const SAMPLES = RAW.map((r) => ({
  ...r,
  id: ids[r.key],
  text: r.text.replace(/\{(\w+)\}/g, (_, k) => `(${ids[k]})`),
}));
// Look up the display letter of a sample by its key, for cross-references.
export const ref = (key) => `(${ids[key]})`;

export const VERDICTS = {
  correct: { label: 'Correct', icon: '✓' },
  partial: { label: 'Correct, with misses', icon: '≈' },
  mixed: { label: 'Mixed', icon: '±' },
  error: { label: 'Failure', icon: '✗' },
};
