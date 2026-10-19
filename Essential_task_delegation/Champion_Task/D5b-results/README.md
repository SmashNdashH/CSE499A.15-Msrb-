# xbd-extra-tier3 (task D5b)

612 xBD tier3 tiles that DisasterM3's segmentation data doesn't use. All of them are DisasterM3 benchmark images (option A, chosen 19 Oct 2026), and none is one of the 645 validation images.

## Pools

```
                                     pool  tiles  buildings  destroyed_polygons tiles_with_destroyed
  new benchmark tiles (converted for R12)    612      41820                4549                  227
new tiles not in the benchmark (left out)   3122       2455                   0                    —
```

A model trained on these tiles must not answer benchmark questions (1,045 of 4,982 BDC questions use them); R12 already follows that rule. The new tiles that aren't benchmark images hold no destroyed building, so they are left out.

## Use in R12

Upload this folder together with `D5-results/`. R12 finds each by its `mapping.json` and maps the masks the same way:

```python
lut = np.array(json.load(open(D5B / 'mapping.json'))['lut'], np.uint8)
mask = lut[cv2.imread(str(D5B / m['mask']), cv2.IMREAD_GRAYSCALE)]   # 0-3, 255 = ignore
```

Images (`post_image` and `pre_image` in `manifest.json`) come from Bojan Tunguz's "xView2 Challenge Dataset - Tier 3 Data" on Kaggle, folder `tier3/images/`.

## Mapping

See `mapping_report.md`. Lookup table: `[0, 1, 2, 2, 3, 255]`.

## Per event

```
             event  tiles  destroyed_polygons
    joplin-tornado    148                3274
lower-puna-volcano     60                 173
     moore-tornado     60                 208
    nepal-flooding     60                  48
   pinery-bushfire     60                  40
 portugal-wildfire     60                  61
tuscaloosa-tornado     60                 210
      woolsey-fire     60                 374
     sunda-tsunami     44                 161
```