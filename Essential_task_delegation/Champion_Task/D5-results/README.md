# xbd-extra-1007 (task D5)

1,007 xBD training tiles that are not in DisasterM3's segmentation data, so none of them is one of the 645 validation images.

## Pools for R12

```
                                 pool  tiles  destroyed_polygons  tiles_with_destroyed
                        all new tiles   1007                3472                   163
benchmark-safe (in_benchmark = false)    560                   6                     3
```

447 tiles are DisasterM3 benchmark images (741 of 4,982 BDC questions use them). Training on them is fine for segmentation scores (O1). A model trained on them must not be used for H1/KR2 counting unless those questions are scored separately.

## Use in R12

Masks keep the raw xBD codes (0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed, 5 un-classified). Map them when loading:

```python
D5 = Path('/kaggle/input/xbd-extra-1007')   # wherever the dataset is mounted
lut = np.array(json.load(open(D5 / 'mapping.json'))['lut'], np.uint8)
mask = lut[cv2.imread(str(D5 / m['mask']), cv2.IMREAD_GRAYSCALE)]   # 0-3, 255 = ignore
```

Images (`post_image` and `pre_image` in `manifest.json`) come from xBD's `train/images` folder.

## Mapping

See `mapping_report.md`. Lookup table: `[0, 1, 2, 2, 3, 255]`.

## Per event

```
              event  tiles  in_benchmark  destroyed_polygons
         socal-fire    467            27                  81
  mexico-earthquake    121           121                   2
       palu-tsunami     82            74                2206
   midwest-flooding     65            29                   5
   hurricane-harvey     61            39                  31
  hurricane-matthew     56            38                 227
 hurricane-florence     51            33                   0
  hurricane-michael     45            35                  99
santa-rosa-wildfire     41            34                 796
  guatemala-volcano     18            17                  25
```