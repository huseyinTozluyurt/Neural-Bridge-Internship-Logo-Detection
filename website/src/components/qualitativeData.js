// Qualitative results on unseen advertisement footage, captured from the
// PyQt5 viewer (detect_yolo.py, conf 0.4, IoU 0.45). Verdicts are judged
// by eye against the visible brand; there is no ground truth for these frames.
export const SAMPLES = [
  {
    id: 'a', file: 'qual_pepsi_can.jpg', verdict: 'correct', brand: 'Pepsi',
    pred: 'pepsi 0.95',
    title: 'Canonical frontal view',
    text: 'A frontal, well-lit can is the easiest case: the box hugs the globe emblem tightly and confidence is near saturation. Condensation and specular highlights on the can do not disturb the detection, and the black redesigned livery, which differs from the older blue cans in FlickrLogos-32, is still recognised from the globe alone.',
  },
  {
    id: 'b', file: 'qual_paulaner_logo.jpg', verdict: 'correct', brand: 'Paulaner',
    pred: 'paulaner 0.96',
    title: 'Emblem versus wordmark',
    text: 'The network localises the circular monk emblem and leaves the large PAULANER wordmark below it outside the box. The model has learned the roundel as "the logo", so on brands with separate figurative and text marks it will report only one of them. This is the distinction OpenLogo encodes with its *_text classes, which the alias map merges into the base brand.',
  },
  {
    id: 'c', file: 'qual_paulaner_beergarden.jpg', verdict: 'correct', brand: 'Paulaner',
    pred: 'paulaner 0.97',
    title: 'Overlay on a cluttered scene',
    text: 'A graphic overlay on a busy beer-garden shot with people, foliage and a large caption. Background clutter produces no spurious boxes, and the emblem is detected with the highest confidence in the set.',
  },
  {
    id: 'd', file: 'qual_erdinger.jpg', verdict: 'correct', brand: 'Erdinger',
    pred: 'erdinger 0.79 · 0.84 · 0.84',
    title: 'Multiple small instances',
    text: 'Three instances: the print on the glass and the labels of two adjacent bottles, each under a tenth of the frame height. Non-maximum suppression keeps the neighbouring bottles as separate detections. The glass print is partly hidden by foam and receives the lowest score.',
  },
  {
    id: 'e', file: 'qual_pepsi_stadium.jpg', verdict: 'correct', brand: 'Pepsi',
    pred: 'pepsi 0.75 · 0.82',
    title: 'Small logos under a strong colour cast',
    text: 'Two small logos above a stage in a night-time aerial shot dominated by blue light. Both are found, which suggests the HSV colour augmentation in training (hsv_h 0.015, hsv_s 0.7, hsv_v 0.4) gives useful robustness to scene lighting.',
  },
  {
    id: 'f', file: 'qual_pepsi_hologram.jpg', verdict: 'correct', brand: 'Pepsi',
    pred: 'pepsi 0.52 · 0.80',
    title: 'Logos inside a screen inside the scene',
    text: 'The cans appear on a giant display filmed at a concert, so they are re-captured, rotated and partly covered by paws. Both are detected. The more occluded can drops to 0.52, close to the 0.4 operating threshold.',
  },
  {
    id: 'g', file: 'qual_pepsi_bears.jpg', verdict: 'partial', brand: 'Pepsi',
    pred: 'pepsi 0.66',
    title: 'Hand-held, rotated, motion-blurred',
    text: 'The can held between the two figures is detected at 0.66 despite rotation and blur. Partly visible cans along the lower edge of the frame are not detected: missed detections (false negatives) on truncated objects, which lower recall.',
  },
  {
    id: 'h', file: 'qual_pepsi_bear_occluded.jpg', verdict: 'partial', brand: 'Pepsi',
    pred: 'pepsi 0.41',
    title: 'Heavy occlusion pushes confidence to the threshold',
    text: 'About half of the can is covered by a paw and the logo occupies a tiny part of the frame. The detection survives at 0.41, just above the default --conf 0.4. The video tracking pipeline uses --yolo-conf 0.5 and would discard it. This one frame shows the precision/recall trade-off of the confidence threshold.',
  },
  {
    id: 'i', file: 'qual_esso_centenary.jpg', verdict: 'correct', brand: 'Esso',
    pred: 'esso 0.92',
    title: 'Logo inside a composite badge',
    text: 'In the centenary badge the model localises only the Esso oval and ignores the surrounding "Celebrating 100 years" artwork and the caption, even though the caption repeats the brand name in plain text.',
  },
  {
    id: 'j', file: 'qual_esso_film_sign.jpg', verdict: 'correct', brand: 'Esso',
    pred: 'esso 0.95',
    title: 'Archival film with scratches and colour fading',
    text: 'A scanned film frame with dust, scratches, vignetting and faded colour, viewed at an oblique angle. The detection is confident and the box covers the sign housing. A slightly loose box like this still counts at IoU 0.5 but costs points at the stricter IoU levels of AP@.5:.95.',
  },
  {
    id: 'k', file: 'qual_esso_tank.jpg', verdict: 'correct', brand: 'Esso',
    pred: 'esso 0.95',
    title: 'Partial snow cover on a curved surface',
    text: 'Snow covers the top of the oval and the tank surface curves the logo. The box stops at the emblem and correctly excludes the adjacent "Diesel Efficient" text.',
  },
  {
    id: 'l', file: 'qual_esso_truck.jpg', verdict: 'correct', brand: 'Esso',
    pred: 'esso 0.97',
    title: 'Large, near-frontal instance',
    text: 'A large logo on a truck panel gets the highest Esso score. The box sits a little wide on the right, towards the trademark symbol, which again matters only at the strict IoU levels of AP@.5:.95.',
  },
  {
    id: 'm', file: 'qual_esso_montage.jpg', verdict: 'mixed', brand: 'Esso',
    pred: 'esso 0.70–0.93 · stellaartois 0.79 · paulaner 0.76',
    title: 'Video wall: many small logos, two confusions',
    text: 'A 5×5 mosaic of interview clips. Most small Esso ovals are found (0.70–0.93), but the same centenary badge is twice given the wrong brand: stellaartois 0.79 (top row) and paulaner 0.76 (bottom-left). Both brands also use a round emblem with red and white elements. At this scale the network keeps the shape and colour pattern but loses the lettering that tells the brands apart. Several Esso logos on trucks in the background are missed.',
  },
  {
    id: 'n', file: 'qual_heineken_carlsberg.jpg', verdict: 'error', brand: 'Heineken',
    pred: 'carlsberg 0.66 (true: heineken)',
    title: 'Right box, wrong brand: green-label beers',
    text: 'An extreme close-up of a Heineken glass is labelled carlsberg. The box itself is well placed around the star and banner, so the localisation is right and only the class is wrong. Both brands use green, white and red on their labels, and the confusion matrix and the per-class table already flag this pair: Carlsberg has the lowest AP@.5:.95 of all 32 classes (0.52). The scale makes it harder still, because Flickr photos rarely show a logo filling the whole frame. The margin is thin (0.66), and the fix is more Carlsberg and Heineken examples, which the combined dataset adds.',
  },
  {
    id: 'o', file: 'qual_esso_bw_bmw.jpg', verdict: 'error', brand: 'Esso',
    pred: 'bmw 0.50 (true: esso)',
    title: 'Grayscale input removes the colour cue',
    text: 'A black-and-white archival photograph of an Esso sign is labelled bmw. Without the red lettering and blue ring, what is left is a ring around an inner oval, which matches the BMW roundel. The score sits right at the threshold. Low-confidence wrong labels like this cluster near the operating point, so raising --conf to 0.55 would remove this error at a small cost in recall. Grayscale augmentation during training would target the cause directly.',
  },
  {
    id: 'p', file: 'qual_esso_shirt_ups.jpg', verdict: 'error', brand: 'Esso',
    pred: 'ups 0.75 (true: esso)',
    title: 'Monochrome print on deforming fabric',
    text: 'A white-on-black Esso print on a shirt, distorted by folds and perspective, is labelled ups with fairly high confidence. As in (o), the colour cue is gone, and the dark field inside a light outline resembles the UPS shield. Unlike (o), this error would survive a higher threshold. The fix is training data: monochrome and printed-on-fabric logos, which OpenLogo adds.',
  },
];

export const VERDICTS = {
  correct: { label: 'Correct', icon: '✓' },
  partial: { label: 'Correct, with misses', icon: '≈' },
  mixed: { label: 'Mixed', icon: '±' },
  error: { label: 'Failure', icon: '✗' },
};
